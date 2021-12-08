import torch
import numpy as np
import librosa
import torch.nn
import scipy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def attack_dvector_emb(model, ori_mel, adv_mel):
    
    ori_w = model(ori_mel.to(device) + torch.normal(0.0, 0.0001, size=ori_mel.size()).to(device))
    adv_w = model(adv_mel.to(device))

    # loss = torch.nn.L1Loss()
    loss = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return loss(ori_w, adv_w)

'''
imperceptible
'''

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