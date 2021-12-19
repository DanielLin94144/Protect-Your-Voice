# requirements python>=3.8, s3prl, fairseq

import os
import sys
import librosa

import torch
from s3prl.hub import mos_wav2vec2
# from s3prl.hub import mos_tera


def main(audio_dir):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'DEVICE: {device}')
    mos_predictor = mos_wav2vec2().to(device)

    files = sorted(os.listdir(audio_dir))
    files = list(filter(lambda f: f[-3:] == 'wav', files))

    wavs = []
    for f in files:
        wav, sr = librosa.load(os.path.join(audio_dir, f), sr=None)
        wav = torch.FloatTensor(wav).to(device)
        wavs.append(wav)

    with torch.no_grad():
        scores = mos_predictor(wavs)['scores']

    for i in range(len(files)):
        print(f'file = {files[i]:>20}, mos = {scores[i]:.4f}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python3 mos.py path/to/audio_dir')
    else:
        main(sys.argv[1])

