import numpy as np
import torch.nn.functional as F
import librosa
from librosa.filters import mel
from librosa.util import pad_center
from scipy.signal import get_window
from torch.autograd import Variable
from torch import hann_window

from .constants import *
from time import time

""" Initial code was from https://github.com/jongwook/onsets-and-frames """

class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length, hop_length, win_length=None, window='hann', padding=True):
        super(STFT, self).__init__()
        if win_length is None:
            win_length = filter_length

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.padding = padding
        
    def forward(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        if self.padding == True:
            input_data = F.pad(
                input_data.unsqueeze(1),
                (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
                mode='reflect')
            input_data = input_data.squeeze(1)
        forward_transform = torch.conv1d(input_data,self.forward_basis, stride=self.hop_length, padding=0)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        return magnitude
        # phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        # return magnitude, phase


class MelSpectrogram(torch.nn.Module):
    def __init__(self, n_mels, sample_rate, filter_length, hop_length,
                 win_length=None, mel_fmin=0.0, mel_fmax=None):
        super(MelSpectrogram, self).__init__()
        self.stft = STFT(filter_length, hop_length, win_length)
        # self.torchSTFT = torchSTFT
        # self.stft = spectrogram()

        mel_basis = mel(sample_rate, filter_length, n_mels, mel_fmin, mel_fmax, htk=True)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def forward(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, T, n_mels)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)
        with torch.no_grad():            
            magnitudes = np.abs(librosa.core.stft(y.numpy()[0], n_fft=WINDOW_LENGTH, hop_length=HOP_LENGTH, center=False))
            magnitudes = torch.Tensor(magnitudes).unsqueeze(0)

            mel_output = torch.matmul(self.mel_basis, magnitudes)
            mel_output = torch.log(torch.clamp(mel_output, min=1e-5))
            return mel_output