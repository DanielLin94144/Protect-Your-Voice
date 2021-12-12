# requirements resemblyzer, matplotlib>=3.0.0, umap-learn

import os
import sys
import numpy as np

from resemblyzer import preprocess_wav, VoiceEncoder


def main(ref_audio_path, audio_dir):
    ref_audio = preprocess_wav(ref_audio_path)

    files = sorted(os.listdir(audio_dir))
    files = list(filter(lambda f: f[-3:] == 'wav', files))

    audios = []
    for f in files:
        audio = preprocess_wav(os.path.join(audio_dir, f))
        audios.append(audio)

    encoder = VoiceEncoder()
    ref_embed = encoder.embed_utterance(ref_audio)

    audio_embeds = [encoder.embed_utterance(audio) for audio in audios]

    print(f'reference audio = {ref_audio_path}')
    for i, audio_embed in enumerate(audio_embeds):
        print(f'file = {files[i]:>20}, asv_score = {np.dot(ref_embed, audio_embed):.4f}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: python3 asv.py path/to/ref_audio.wav path/to/audio_dir')
    else:
        main(sys.argv[1], sys.argv[2])

