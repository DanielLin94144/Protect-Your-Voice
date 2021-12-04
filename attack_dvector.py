# https://github.com/CorentinJ/Real-Time-Voice-Cloning/
import torch
from torch import nn
import librosa
import torchaudio

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
    model.eval()
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
    # frames = librosa.feature.melspectrogram(
    #     wav,
    #     sampling_rate,
    #     n_fft=int(sampling_rate * mel_window_length / 1000),
    #     hop_length=int(sampling_rate * mel_window_step / 1000),
    #     n_mels=mel_n_channels
    # )

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

    wav = load_audio(audio_path, sampling_rate)
    # attack here
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    mel = wav_to_mel_spectrogram(wav)
    print("Loaded file succesfully")
    
    # Then we derive the embedding. There are many functions and parameters that the 
    # speaker encoder interfaces. These are mostly for in-depth research. You will typically
    # only use this function (with its default parameters):
    # embed = embed_utterance(preprocessed_wav)

    emb = model(mel.unsqueeze(0).to(device))
    print(emb.shape)
    print("Created the embedding")