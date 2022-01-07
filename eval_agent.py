import os
import sys
import librosa
import numpy as np

import torch
from s3prl.hub import mos_wav2vec2, mos_tera, mos_apc
from resemblyzer import preprocess_wav, VoiceEncoder


class EvalAgent():
    def __init__(self, root_dir):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.root_dir = root_dir
        self.encoder = VoiceEncoder(verbose=False)
        self.mos_predictor = mos_apc().to(self.device)

    def asv_score(self, dir_01, dir_02):
        files = sorted(os.listdir(os.path.join(self.root_dir, dir_01)))

        embed = []
        for f in files:
            ref_01 = preprocess_wav(os.path.join(self.root_dir, dir_01, f))
            ref_02 = preprocess_wav(os.path.join(self.root_dir, dir_02, f))

            embed_01 = self.encoder.embed_utterance(ref_01)
            embed_02 = self.encoder.embed_utterance(ref_02)

            embed.append((embed_01, embed_02))

        scores = [np.dot(e[0], e[1]) for e in embed]

        return np.mean(scores), np.std(scores)

    def mos_score(self, dir_01):
        files = sorted(os.listdir(os.path.join(self.root_dir, dir_01)))

        wavs = []
        for f in files:
            wav, sr = librosa.load(os.path.join(self.root_dir, dir_01, f), sr=None)
            wav = torch.FloatTensor(wav).to(self.device)
            wavs.append(wav)

        self.mos_predictor.eval()
        with torch.no_grad():
            scores = self.mos_predictor(wavs)['scores'].detach().cpu().numpy()

        return np.mean(scores), np.std(scores)


def eval_all(root_dir):
    ref_dir = '00_gt'
    test_dirs = ['01_ori_with_adv', '02_ori_with_base', '03_synthesized_ori', '04_synthesized_adv', '05_synthesized_base']

    agent = EvalAgent(root_dir=root_dir)
    for test_dir in test_dirs:
        asv_score = agent.asv_score(ref_dir, test_dir)
        mos_score = agent.mos_score(test_dir)

        print(f'[ref = {ref_dir}, test_dir = {test_dir:>20}] asv = {asv_score[0]:.4f} +- {asv_score[1]:.4f}, mos = {mos_score[0]:.4f} += {mos_score[1]:.4f}')


if __name__ == '__main__':
    eval_all(sys.argv[1])

