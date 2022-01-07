import torch
import numpy as np
import os
import argparse
import librosa
import re
import json
import random
from string import punctuation
from g2p_en import G2p

from models.StyleSpeech import StyleSpeech
from text import text_to_sequence
import audio as Audio
import utils
import soundfile as sf
from torch.autograd import Variable
import torch.nn
from torch.utils.data import Dataset

import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from tqdm import trange, tqdm

from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from attack_utils import attack_emb, imperceptible_attack
from melgan_neurips.mel2wav.interface import MelVocoder


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

    # print("Raw Text Sequence: {}".format(text))
    # print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(phones, ['english_cleaners']))

    return torch.from_numpy(sequence).to(device=device)


def preprocess_audio(audio_file):
    wav, sample_rate = librosa.load(audio_file, sr=None)
    if sample_rate != 16000:
        wav = librosa.resample(wav, sample_rate, 16000)
    return wav

def preprocess_audio_list(audio_list):
    wavs = []
    for file in audio_list:
        wav, sample_rate = librosa.load(file, sr=None)
        if sample_rate != 16000:
            wav = librosa.resample(wav, sample_rate, 16000)
        wavs.append(wav)
    wavs = np.concatenate(wavs)
    return wavs


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


# class AudioDataset(Dataset):
#     def __init__(self, data_root='/hdd0/CORPORA/VCTK-Corpus-0.92'):
#         self.wav_dir = os.path.join(data_root, 'wav48_silence_trimmed')
#         self.txt_dir = os.path.join(data_root, 'txt')

#         remove_tags = ['p280', 'p362']
#         self.speaker_list = sorted(os.listdir(self.txt_dir))
#         self.speaker_list = [e for e in self.speaker_list if e not in remove_tags]

#     def __getitem__(self, idx):
#         '''
#         Returns:
#             tuple: ``(target_audio, target_text, gt_audio)``
#         '''
#         speaker = self.speaker_list[idx]

#         idx_list = [e.split('.')[0].split('_')[1] for e in sorted(os.listdir(os.path.join(self.txt_dir, speaker)))]
#         idx_01, idx_02 = random.sample(idx_list, 2)

#         target_audio = os.path.join(self.wav_dir, speaker, f'{speaker}_{idx_01}_mic2.flac')
#         gt_audio = os.path.join(self.wav_dir, speaker, f'{speaker}_{idx_02}_mic2.flac')

#         with open(os.path.join(self.txt_dir, speaker, f'{speaker}_{idx_02}.txt')) as f:
#             target_text = f.read()

#         return target_audio, target_text, gt_audio

#     def __len__(self):
#         return len(self.speaker_list)

# for battleship
class AudioDataset(Dataset):
    def __init__(self, data_root='VCTK-Corpus'):
        self.wav_dir = os.path.join(data_root, 'wav48')
        self.txt_dir = os.path.join(data_root, 'txt')

        remove_tags = ['p280'] + [f'p{i}' for i in range(300, 376 + 1)]
        self.speaker_list = sorted(os.listdir(self.txt_dir))
        self.speaker_list = [e for e in self.speaker_list if e not in remove_tags]
        self.num_sent = 3

    def __getitem__(self, idx):
        '''
        Returns:
            tuple: ``(target_audio, target_text, gt_audio)``
        '''
        speaker = self.speaker_list[idx]

        idx_list = [e.split('.')[0].split('_')[1] for e in sorted(os.listdir(os.path.join(self.txt_dir, speaker)))]
        # idx_01, idx_02 = random.sample(idx_list, 2)
        samples = random.sample(idx_list, self.num_sent)
        duration = sum([librosa.get_duration(filename=os.path.join(self.wav_dir, speaker, f'{speaker}_{idx}.wav')) for idx in samples])
        idx_02 = random.sample(idx_list, 1)[0]
        while duration < 10:
            samples = random.sample(idx_list, self.num_sent)
            duration = sum([librosa.get_duration(filename=os.path.join(self.wav_dir, speaker, f'{speaker}_{idx}.wav')) for idx in samples])
        # idx_01 is cloned speech (speech only)
        # idx_02 is ground truth speech (both speech and text)
        target_audio = [ os.path.join(self.wav_dir, speaker, f'{speaker}_{idx}.wav') for idx in samples ]
        gt_audio = os.path.join(self.wav_dir, speaker, f'{speaker}_{idx_02}.wav')

        with open(os.path.join(self.txt_dir, speaker, f'{speaker}_{idx_02}.txt')) as f:
            target_text = f.read()

        return target_audio, target_text, gt_audio

    def __len__(self):
        return len(self.speaker_list)


class DummyDataset(Dataset):
    def __init__(self, args):
        self.audio_dir = '/hdd0/CORPORA/VCTK/clean_testset_wav'
        self.audio_filenames = sorted(os.listdir(self.audio_dir))

    def __getitem__(self, idx):
        target_audio = os.path.join(self.audio_dir, self.audio_filenames[idx])
        target_text = 'Some random text.'
        gt_audio = os.path.join(self.audio_dir, self.audio_filenames[idx])

        return target_audio, target_text, gt_audio

    def __len__(self):
        return len(self.audio_filenames)


def write_wav(audio_path, audio_tensor, sr=16000):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    sf.write(audio_path, audio_tensor.transpose(0, 1).cpu().numpy(), sr)


def synthesize(args, model, target_model, vocoder, _stft, target_audio, target_text, gt_audio):
    # hyperparameters
    learning_rate = args.learning_rate
    iter = args.iteration
    eps = args.epsilon

    wav = preprocess_audio_list(target_audio)
    src = preprocess_english(target_text, args.lexicon_path).unsqueeze(0)
    src_len = torch.from_numpy(np.array([src.shape[1]])).to(device=device)

    wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
    wav = wav.detach()

    # attack
    if args.random_start:
        delta = Variable(torch.empty_like(wav).uniform_(-eps, eps), requires_grad=True)
    else:
        delta = Variable(torch.zeros(wav.size()).type(torch.FloatTensor), requires_grad=True)

    optimizer = torch.optim.SGD(params=[delta], lr=learning_rate, momentum=args.momentum)
    ori_mel = wav2mel(wav).transpose(2, 1).to(device=device).detach()

    if not args.imp_loss:
        # iterative attack
        for _ in range(iter):
            optimizer.zero_grad()
            _delta = torch.clamp(delta, -eps, eps)
            adv_wav = wav + _delta
            adv_mel = wav2mel(adv_wav)

            adv_mel = adv_mel.to(device=device).transpose(2, 1)
            loss = attack_emb(model, ori_mel, adv_mel)
            loss.backward(retain_graph=True)
            delta.grad = torch.sign(delta.grad)

            optimizer.step()
    else:
        # imp attack
        first_iter = 20
        second_iter = 300
        alpha = 0.02
        threshold = 0.2

        optimizer1 = torch.optim.SGD(params=[delta], lr=args.learning_rate1, momentum=args.momentum)
        ori_mel = wav2mel(wav).transpose(2, 1).to(device=device).detach()
        imp_attack = imperceptible_attack()
        # 1st stage
        for _ in trange(first_iter):
            optimizer1.zero_grad()
            _delta = torch.clamp(delta, -eps, eps)
            adv_wav = wav + _delta
            adv_mel = wav2mel(adv_wav)

            adv_mel = adv_mel.to(device=device).transpose(2, 1)
            attack_loss = attack_emb(model, ori_mel, adv_mel)
            loss = attack_loss
            print('[INFO]  loss = ', loss.item())
            loss.backward(retain_graph=True)
            delta.grad = torch.sign(delta.grad)

            optimizer1.step()


        # 2nd stage
        delta_2 = Variable(delta, requires_grad=True)

        optimizer2 = torch.optim.Adam(params=[delta_2], lr=args.learning_rate2)
        for i in trange(second_iter):
            optimizer2.zero_grad()
            if torch.isnan(torch.sum(delta_2)):
                # delta_2 = _delta # save the last non-nan delta
                break
            _delta = delta_2.clone().detach()
            # no clamping: only bounded by psd mask
            adv_wav = wav + delta_2
            adv_mel = wav2mel(adv_wav)

            adv_mel = adv_mel.to(device=device).transpose(2, 1)
            attack_loss = attack_emb(model, ori_mel, adv_mel)
            imperceptible_loss = imp_attack.imperceptible_loss(delta_2, wav.squeeze(0).squeeze(0).cpu().numpy())
            if i % 2 and attack_loss < 0.2:
                alpha = alpha * 1.2
            elif i % 3 and attack_loss > 0.2:
                alpha = alpha * 0.8

            # print(attack_loss)
            # print(imperceptible_loss)
            loss = attack_loss + alpha * imperceptible_loss
            # print('[INFO]  loss = ', loss.item())

            loss.backward(retain_graph=True)
            # clip grad
            clip_value = 1.0
            # clip_grad_value_(delta_2, clip_value)
            clip_grad_norm_(delta_2, clip_value)

            optimizer2.step()


    # baseline: random noise
    base_wav = wav + eps * torch.normal(0, 1, size=delta.size()).tanh()
    base_mel = wav2mel(base_wav)
    base_mel = base_mel.to(device=device).transpose(2, 1)

    # use final delta perturbation to create adv wav
    if not args.imp_loss:
        delta = torch.clamp(delta, -eps, eps)
        adv_wav = wav + delta.detach()
        adv_mel = wav2mel(adv_wav)
        adv_mel = adv_mel.to(device=device).transpose(2, 1)
    else:
        adv_wav = wav + _delta.detach()
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
    out_wav_ori = vocoder.inverse(result_mel_ori.unsqueeze(0))
    out_wav_adv = vocoder.inverse(result_mel_adv.unsqueeze(0))
    out_wav_base = vocoder.inverse(result_mel_base.unsqueeze(0))

    # save file
    # filename = os.path.basename(target_audio)
    filename = os.path.basename(gt_audio)
    gt_wav = preprocess_audio(gt_audio)
    gt_wav = torch.from_numpy(gt_wav).unsqueeze(0).unsqueeze(0)
    gt_wav = gt_wav.detach()

    write_wav(os.path.join(args.save_dir, '00_gt',              filename), gt_wav.squeeze(0))
    write_wav(os.path.join(args.save_dir, '01_ori_with_adv',     filename), adv_wav.squeeze(0))
    write_wav(os.path.join(args.save_dir, '02_ori_with_base',    filename), base_wav.squeeze(0))
    write_wav(os.path.join(args.save_dir, '03_synthesized_ori',  filename), out_wav_ori)
    write_wav(os.path.join(args.save_dir, '04_synthesized_adv',  filename), out_wav_adv)
    write_wav(os.path.join(args.save_dir, '05_synthesized_base', filename), out_wav_base)

    # black box 
    target_style_vector_adv = target_model.get_style_vector(adv_mel)
    result_mel_target = model.inference(target_style_vector_adv, src, src_len)[0]
    result_mel_target = result_mel_target.cpu().squeeze().transpose(0, 1).detach()
    out_wav_target = vocoder.inverse(result_mel_target.unsqueeze(0))
    write_wav(os.path.join(args.save_dir, '06_synthesized_black',  filename), out_wav_target)

def synthesize_all(args, model, target_model, vocoder, _stft):
    audio_dataset = AudioDataset(args.data_root)
    # audio_dataset = DummyDataset(args)

    for batch in tqdm(audio_dataset, desc='Generating audios'):
        target_audio = batch[0]
        target_text = batch[1]
        gt_audio = batch[2]

        synthesize(args, model, target_model, vocoder, _stft, target_audio, target_text, gt_audio)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--blackbox_target_path", type=str, required=False, help="Path to the pretrained model")
    parser.add_argument('--config',          type=str, default='configs/config.json')
    parser.add_argument("--save_dir",        type=str, default='results/')

    parser.add_argument("--lexicon_path",    type=str, default='lexicon/librispeech-lexicon.txt')
    parser.add_argument("--vocoder_path",    type=str, default='melgan_neurips/pretrained')
    parser.add_argument("--seed",            type=int, default=0)
    parser.add_argument("--imp_loss", action='store_true')
    parser.add_argument("--random_start", action='store_true')
    parser.add_argument("--epsilon", type=float, default=0.002)
    parser.add_argument("--iteration", type=float, default=40)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--learning_rate1", type=float, default=0.001)
    parser.add_argument("--learning_rate2", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--data_root", type=str, default='VCTK-Corpus')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    random.seed(args.seed)

    # Get config
    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)

    # Get model
    model = get_StyleSpeech(config, args.checkpoint_path)
    target_model = None
    if args.blackbox_target_path is not None: 
        target_model = get_StyleSpeech(config, args.blackbox_target_path)

    vocoder = MelVocoder(path=args.vocoder_path)

    _stft = Audio.stft.TacotronSTFT(
        config.filter_length,
        config.hop_length,
        config.win_length,
        config.n_mel_channels,
        config.sampling_rate,
        config.mel_fmin,
        config.mel_fmax
    )

    # Synthesize
    synthesize_all(args, model, target_model, vocoder, _stft)


if __name__ == "__main__":
    main()

