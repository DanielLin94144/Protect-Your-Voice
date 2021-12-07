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


'''
attack method
'''
def attack_emb(model, ori_mel, adv_mel):
    ori_w = model.get_style_vector(ori_mel + torch.normal(0.0, 0.0001, size=ori_mel.size()).to(device))
    adv_w = model.get_style_vector(adv_mel)
    loss = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # loss = torch.nn.L1Loss()
    # loss = torch.nn.MSELoss()
    return loss(ori_w, adv_w)
'''
imperceptible
'''
import scipy
class imperceptible_attack():
    def __init__ (self,     
        win_length: int = 2048,
        hop_length: int = 512,
        n_fft: int = 2048,
        sample_rate = 16000, 
        ): 
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.device = device

    def compute_masking_threshold(self, x):
        """
        Compute the masking threshold and the maximum psd of the original audio.
        :param x: Samples of shape (seq_length,).
        :return: A tuple of the masking threshold and the maximum psd.
        """

        # First compute the psd matrix
        # Get window for the transformation
        window = scipy.signal.get_window("hann", self.win_length, fftbins=True)

        # Do transformation
        transformed_x = librosa.core.stft(
            y=x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window, center=False
        )
        transformed_x *= np.sqrt(8.0 / 3.0)

        psd = abs(transformed_x / self.win_length)
        original_max_psd = np.max(psd * psd)
        with np.errstate(divide="ignore"):
            psd = (20 * np.log10(psd)).clip(min=-200)
        psd = 96 - np.max(psd) + psd

        # Compute freqs and barks
        freqs = librosa.core.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        barks = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan(pow(freqs / 7500.0, 2))

        # Compute quiet threshold
        ath = np.zeros(len(barks), dtype=np.float64) - np.inf
        bark_idx = np.argmax(barks > 1)
        ath[bark_idx:] = (
            3.64 * pow(freqs[bark_idx:] * 0.001, -0.8)
            - 6.5 * np.exp(-0.6 * pow(0.001 * freqs[bark_idx:] - 3.3, 2))
            + 0.001 * pow(0.001 * freqs[bark_idx:], 4)
            - 12
        )

        # Compute the global masking threshold theta
        theta = []

        for i in range(psd.shape[1]):
            # Compute masker index
            masker_idx = scipy.signal.argrelextrema(psd[:, i], np.greater)[0]

            if 0 in masker_idx:
                masker_idx = np.delete(masker_idx, 0)

            if len(psd[:, i]) - 1 in masker_idx:
                masker_idx = np.delete(masker_idx, len(psd[:, i]) - 1)

            barks_psd = np.zeros([len(masker_idx), 3], dtype=np.float64)
            barks_psd[:, 0] = barks[masker_idx]
            barks_psd[:, 1] = 10 * np.log10(
                pow(10, psd[:, i][masker_idx - 1] / 10.0)
                + pow(10, psd[:, i][masker_idx] / 10.0)
                + pow(10, psd[:, i][masker_idx + 1] / 10.0)
            )
            barks_psd[:, 2] = masker_idx

            for j in range(len(masker_idx)):
                if barks_psd.shape[0] <= j + 1:
                    break

                while barks_psd[j + 1, 0] - barks_psd[j, 0] < 0.5:
                    quiet_threshold = (
                        3.64 * pow(freqs[int(barks_psd[j, 2])] * 0.001, -0.8)
                        - 6.5 * np.exp(-0.6 * pow(0.001 * freqs[int(barks_psd[j, 2])] - 3.3, 2))
                        + 0.001 * pow(0.001 * freqs[int(barks_psd[j, 2])], 4)
                        - 12
                    )
                    if barks_psd[j, 1] < quiet_threshold:
                        barks_psd = np.delete(barks_psd, j, axis=0)

                    if barks_psd.shape[0] == j + 1:
                        break

                    if barks_psd[j, 1] < barks_psd[j + 1, 1]:
                        barks_psd = np.delete(barks_psd, j, axis=0)
                    else:
                        barks_psd = np.delete(barks_psd, j + 1, axis=0)

                    if barks_psd.shape[0] == j + 1:
                        break

            # Compute the global masking threshold
            delta = 1 * (-6.025 - 0.275 * barks_psd[:, 0])

            t_s = []

            for m in range(barks_psd.shape[0]):
                d_z = barks - barks_psd[m, 0]
                zero_idx = np.argmax(d_z > 0)
                s_f = np.zeros(len(d_z), dtype=np.float64)
                s_f[:zero_idx] = 27 * d_z[:zero_idx]
                s_f[zero_idx:] = (-27 + 0.37 * max(barks_psd[m, 1] - 40, 0)) * d_z[zero_idx:]
                t_s.append(barks_psd[m, 1] + delta[m] + s_f)

            t_s_array = np.array(t_s)

            theta.append(np.sum(pow(10, t_s_array / 10.0), axis=0) + pow(10, ath / 10.0))

        theta = np.array(theta)

        return theta, original_max_psd

    def psd_transform(self, delta, original_max_psd):
        """
        Compute the psd matrix of the perturbation.
        :param delta: The perturbation.
        :param original_max_psd: The maximum psd of the original audio.
        :return: The psd matrix.
        """
        # import torch  # lgtm [py/repeated-import]

        # Get window for the transformation
        window_fn = torch.hann_window  # type: ignore
        delta = delta.squeeze(0)
        # Return STFT of delta
        delta_stft = torch.stft(
            delta,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
            window=window_fn(self.win_length).to(self.device),
        ).to(self.device)

        # Take abs of complex STFT results
        transformed_delta = torch.sqrt(torch.sum(torch.square(delta_stft), -1))

        # Compute the psd matrix
        psd = (8.0 / 3.0) * transformed_delta / self.win_length
        psd = psd ** 2
        psd = (
            torch.pow(torch.tensor(10.0).type(torch.float64), torch.tensor(9.6).type(torch.float64)).to(
                self.device
            )
            / torch.reshape(torch.tensor(original_max_psd).to(self.device), [-1, 1, 1])
            * psd.type(torch.float64)
        )

        return psd

    def imperceptible_loss(self, delta, ori_wav):
        theta, original_max_psd = self.compute_masking_threshold(ori_wav) # input numpy array
        
        relu = torch.nn.ReLU()
        
        psd_transform_delta = self.psd_transform(
            delta=delta.to(self.device), original_max_psd=torch.tensor(original_max_psd).to(self.device)
        )
        loss = torch.mean(relu(psd_transform_delta.squeeze(0).transpose(1, 0) - torch.tensor(theta).to(self.device)))

        return loss

def synthesize(args, model, _stft):
    # hyperparameters
    learning_rate1 = 0.001
    learning_rate2 = 5e-4
    first_iter = 20
    second_iter = 500
    eps = 0.01
    alpha = 0.02

    wav = preprocess_audio(args.ref_audio)
    src = preprocess_english(args.text, args.lexicon_path).unsqueeze(0)
    src_len = torch.from_numpy(np.array([src.shape[1]])).to(device=device)

    wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
    wav = wav.detach()
    # attack
    delta = Variable(torch.zeros(wav.size()).type(torch.FloatTensor), requires_grad=True)
    optimizer1 = torch.optim.SGD(params=[delta], lr=learning_rate1, momentum=1)
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
    
    optimizer2 = torch.optim.Adam(params=[delta_2], lr=learning_rate2)
    for i in trange(second_iter):
        optimizer2.zero_grad()
        _delta = delta_2  # no clamping: only bounded by psd mask
        adv_wav = wav + _delta
        adv_mel = wav2mel(adv_wav)

        adv_mel = adv_mel.to(device=device).transpose(2, 1)
        attack_loss = attack_emb(model, ori_mel, adv_mel)
        imperceptible_loss = imp_attack.imperceptible_loss(_delta, wav.squeeze(0).squeeze(0).cpu().numpy())
        if attack_loss < 0.4: 
            alpha = alpha * 1.2
        elif attack_loss > 0.4: 
            alpha = alpha * 0.8

        print(attack_loss)
        print(imperceptible_loss)
        loss = attack_loss + alpha * imperceptible_loss
        print('[INFO]  loss = ', loss.item())
        loss.backward(retain_graph=True)
        # clip grad
        clip_value = 1.0
        clip_grad_value_(delta_2, clip_value)

        optimizer2.step()

    # baseline: random noise
    base_wav = wav + eps * torch.normal(0, 1, size=delta.size()).tanh()
    base_mel = wav2mel(base_wav)
    base_mel = base_mel.to(device=device).transpose(2, 1)
    # use final delta perturbation to create adv wav
    adv_wav = wav + delta_2.detach()
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
    # vocoder = MelVocoder(path='./melgan_neurips/pretrained/')
    vocoder = MelVocoder(path='/home/daniel094144/Daniel/StyleSpeech/melgan_neurips/pretrained/')
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
    parser.add_argument("--save_path", type=str, default='imperceptible_results/')
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
