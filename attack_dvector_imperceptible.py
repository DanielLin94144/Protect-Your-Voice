import torch
import os
import librosa

import soundfile as sf
from torch.autograd import Variable
import torch.nn

from tqdm import trange
from torch import nn 
from torch.nn.utils import clip_grad_norm_
from attack_utils import attack_dvector_emb, imperceptible_attack
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpeakerEncoder(nn.Module):
    def __init__(self, device, loss_device, model_hidden_size = 256,
        model_embedding_size = 256, model_num_layers = 3, mel_n_channels = 40):
        super().__init__()
        self.loss_device = loss_device
        
        # Network defition
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True).to(device)
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=model_embedding_size).to(device)
        self.relu = torch.nn.ReLU().to(device)
        
        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(loss_device)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)
    
    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        
        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape 
        (batch_size, n_frames, n_channels) 
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers, 
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        # L2-normalize it
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        

        return embeds



def load_model(weights_fpath='./real_time_encoder/pretrained.pt', device=None):
    """
    Loads the model in memory. If this function is not explicitely called, it will be run on the 
    first call to embed_frames() with the default weights file.
    
    :param weights_fpath: the path to saved model weights.
    :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The 
    model will be loaded and will run on this device. Outputs will however always be on the cpu. 
    If None, will default to your GPU if it"s available, otherwise your CPU.
    """
    # TODO: I think the slow loading of the encoder might have something to do with the device it
    #   was saved on. Worth investigating.
    
    model = SpeakerEncoder(device, torch.device("cpu"))
    checkpoint = torch.load(weights_fpath, device)
    model.load_state_dict(checkpoint["model_state"])
    # model.eval()
    print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath, checkpoint["step"]))
    
    return model.to(device)


def load_audio(fpath, sampling_rate=16000):
    original_wav, source_sr = librosa.load(fpath)
    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = torch.from_numpy(librosa.resample(original_wav, source_sr, sampling_rate))
    
    return wav


def wav_to_mel_spectrogram(wav, sampling_rate=16000, mel_window_length=25, mel_window_step=10, 
    mel_n_channels = 40):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        win_length=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        # onesided=True,
        n_mels=mel_n_channels,
    )(wav)

    # return frames.astype(torch.float32).T
    return frames.T

def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * torch.log10(torch.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))


if __name__ == '__main__': 

    audio_path = './audio/1463_infer.wav'
    sampling_rate = 16000
    audio_norm_target_dBFS = -30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)

    # hyperparameters
    learning_rate1 = 0.001
    learning_rate2 = 5e-4
    first_iter = 20
    second_iter = 150
    eps = 0.01
    alpha = 0.02

    ori_wav = load_audio(audio_path, sampling_rate).detach()
    # attack
    delta = Variable(torch.zeros(ori_wav.size()).type(torch.FloatTensor), requires_grad=True)
    optimizer1 = torch.optim.SGD(params=[delta], lr=learning_rate1, momentum=1)
    wav = normalize_volume(ori_wav, audio_norm_target_dBFS, increase_only=True)
    ori_mel = wav_to_mel_spectrogram(wav).detach().unsqueeze(0)

    imp_attack = imperceptible_attack()
    # 1st stage
    for _ in trange(first_iter):
        optimizer1.zero_grad()
        _delta = torch.clamp(delta, -eps, eps)
        adv_wav = wav + _delta
        adv_wav = normalize_volume(adv_wav, audio_norm_target_dBFS, increase_only=True)
        adv_mel = wav_to_mel_spectrogram(adv_wav).unsqueeze(0)

        attack_loss = attack_dvector_emb(model, ori_mel, adv_mel)
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
        if torch.isnan(torch.sum(delta_2)): 
            # delta_2 = _delta # save the last non-nan delta
            break
        _delta = delta_2.clone().detach()
        # no clamping: only bounded by psd mask
        adv_wav = wav + delta_2
        adv_wav = normalize_volume(adv_wav, audio_norm_target_dBFS, increase_only=True)
        adv_mel = wav_to_mel_spectrogram(adv_wav).unsqueeze(0)

        attack_loss = attack_dvector_emb(model, ori_mel, adv_mel)
        imperceptible_loss = imp_attack.imperceptible_loss(delta_2, wav.squeeze(0).squeeze(0).cpu().numpy())
        if i % 2 and attack_loss < 0.1: 
            alpha = alpha * 1.2
        elif i % 3 and attack_loss > 0.1: 
            alpha = alpha * 0.8

        print(attack_loss)
        print(imperceptible_loss)
        loss = attack_loss + alpha * imperceptible_loss
        print('[INFO]  loss = ', loss.item())
        
        loss.backward(retain_graph=True)
        # clip grad
        clip_value = 1.0
        # clip_grad_value_(delta_2, clip_value)
        clip_grad_norm_(delta_2, clip_value)

        optimizer2.step()

    adv_wav = wav + _delta.detach()

    # save file
    save_path = './dvector_results_imp'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    sf.write(os.path.join(save_path, 'ori_with_adv.wav'), adv_wav.cpu().numpy(), 16000)


