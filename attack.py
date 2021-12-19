import torch
import numpy as np
import os
import argparse
import librosa
import re
import json
from string import punctuation
from g2p_en import G2p

from models.StyleSpeech import StyleSpeech
from text import text_to_sequence
import audio as Audio
import utils
import soundfile as sf
from torch.autograd import Variable
import torch.nn

import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from tqdm import trange

from torch.nn.utils import clip_grad_value_
from attack_utils import attack_emb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, lexicon_path):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(lexicon_path)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(phones, ['english_cleaners']))

    return torch.from_numpy(sequence).to(device=device)


def preprocess_audio(audio_file):
    wav, sample_rate = librosa.load(audio_file, sr=None)
    if sample_rate != 16000:
        wav = librosa.resample(wav, sample_rate, 16000)
    return wav

def get_StyleSpeech(config, checkpoint_path):
    model = StyleSpeech(config).to(device=device)
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.eval()
    return model

def wav2mel(audio):
    n_fft=1024
    hop_length=256
    win_length=1024
    sampling_rate=16000
    n_mel_channels=80
    mel_fmin=0.0
    mel_fmax=None
    window = torch.hann_window(win_length).float()

    p = (n_fft - hop_length) // 2

    # audio = F.pad(audio, (p, p), "reflect").squeeze(1)

    audio = F.pad(audio, (int(n_fft / 2), int(n_fft / 2)), "reflect").squeeze(1)
    fft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
    )
    mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
    real_part, imag_part = fft.unbind(-1)
    magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
    mel_output = torch.matmul(torch.from_numpy(mel_basis), magnitude)
    log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
    return log_mel_spec




def synthesize(args, model, _stft):
    # hyperparameters
    learning_rate = 0.001
    iter = 20
    eps = 0.005
    wav = preprocess_audio(args.ref_audio)
    src = preprocess_english(args.text, args.lexicon_path).unsqueeze(0)
    src_len = torch.from_numpy(np.array([src.shape[1]])).to(device=device)

    wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
    wav = wav.detach()
    # attack
    delta = Variable(torch.zeros(wav.size()).type(torch.FloatTensor), requires_grad=True)
    optimizer = torch.optim.SGD(params=[delta], lr=learning_rate, momentum=1)
    ori_mel = wav2mel(wav).transpose(2, 1).to(device=device).detach()

    # iterative attack
    for _ in trange(iter):
        optimizer.zero_grad()
        _delta = torch.clamp(delta, -eps, eps)
        adv_wav = wav + _delta
        adv_mel = wav2mel(adv_wav)

        adv_mel = adv_mel.to(device=device).transpose(2, 1)
        loss = attack_emb(model, ori_mel, adv_mel)
        print('[INFO]  loss = ', loss.item())
        loss.backward(retain_graph=True)
        delta.grad = torch.sign(delta.grad)

        optimizer.step()

    # baseline: random noise
    base_wav = wav + eps * torch.normal(0, 1, size=delta.size()).tanh()
    base_mel = wav2mel(base_wav)
    base_mel = base_mel.to(device=device).transpose(2, 1)
    # use final delta perturbation to create adv wav
    delta = torch.clamp(delta, -eps, eps)
    adv_wav = wav + delta.detach()
    adv_mel = wav2mel(adv_wav)
    adv_mel = adv_mel.to(device=device).transpose(2, 1)
    # extact style vector
    style_vector_base = model.get_style_vector(base_mel)
    style_vector_adv = model.get_style_vector(adv_mel)
    style_vector_ori = model.get_style_vector(ori_mel)
    # voice cloning
    result_mel_ori = model.inference(style_vector_ori, src, src_len)[0]
    result_mel_ori = result_mel_ori.cpu().squeeze().transpose(0, 1).detach()
    result_mel_adv = model.inference(style_vector_adv, src, src_len)[0]
    result_mel_adv = result_mel_adv.cpu().squeeze().transpose(0, 1).detach()
    result_mel_base = model.inference(style_vector_base, src, src_len)[0]
    result_mel_base = result_mel_base.cpu().squeeze().transpose(0, 1).detach()

    # vocoder
    from melgan_neurips.mel2wav.interface import MelVocoder
    vocoder = MelVocoder(path='./melgan_neurips/pretrained/')
    out_wav_ori = vocoder.inverse(result_mel_ori.unsqueeze(0))
    out_wav_adv = vocoder.inverse(result_mel_adv.unsqueeze(0))
    out_wav_base = vocoder.inverse(result_mel_base.unsqueeze(0))

    # save file
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    sf.write('./results/synthesized_ori.wav', out_wav_ori.transpose(0, 1).cpu().numpy(), 16000)
    sf.write('./results/synthesized_adv.wav', out_wav_adv.transpose(0, 1).cpu().numpy(), 16000)
    sf.write('./results/synthesized_base.wav', out_wav_base.transpose(0, 1).cpu().numpy(), 16000)
    sf.write('./results/ori_with_adv.wav', adv_wav.squeeze(0).transpose(0, 1).cpu().numpy(), 16000)
    sf.write('./results/ori_with_base.wav', base_wav.squeeze(0).transpose(0, 1).cpu().numpy(), 16000)
    # plotting
    utils.plot_data([result_mel_ori.numpy(), result_mel_adv.numpy()],
        ['Original Synthesized', 'Adversarial Synthesized'], filename=os.path.join(save_path, 'plot.png'))
    print('Generate done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
        help="Path to the pretrained model")
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument("--save_path", type=str, default='results/')
    parser.add_argument("--ref_audio", type=str, required=True,
        help="path to an reference speech audio sample")
    parser.add_argument("--text", type=str, required=True,
        help="raw text to synthesize")
    parser.add_argument("--lexicon_path", type=str, default='lexicon/librispeech-lexicon.txt')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)

    # Get model
    model = get_StyleSpeech(config, args.checkpoint_path)
    print('model is prepared')

    _stft = Audio.stft.TacotronSTFT(
                config.filter_length,
                config.hop_length,
                config.win_length,
                config.n_mel_channels,
                config.sampling_rate,
                config.mel_fmin,
                config.mel_fmax)

    # Synthesize
    synthesize(args, model, _stft)
