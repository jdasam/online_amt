
import torch
import torch.nn.functional as F
from torch import nn
import random

from .mel import MelSpectrogram
from .constants import *


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features //
                      16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16,
                      output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) *
                      (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class ARmodel(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48,
                 recursive=True, n_class=5, context_len=1):
        super().__init__()
        self.recursive = recursive
        self.n_class = n_class
        self.melspectrogram = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX)
        model_size = model_complexity * 16
        self.model_size = model_size
        self.context_len = context_len
        self.embedding = nn.Embedding(n_class, 2)
        self.conv_stack = ConvStack(input_features, model_size)
        rc_dim = 88*2 if recursive else 0
        self.sequence_model = nn.LSTM(
            model_size * context_len + rc_dim, model_size, num_layers=2, batch_first=True, bidirectional=False)
        self.sequence_model.flatten_parameters()
        self.fc = nn.Linear(model_size, output_features * n_class)

    def acoustic_model(self, audio):
        batch_size = audio.shape[0]
        mel = self.melspectrogram(
            audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
        acoustic_out = self.conv_stack(mel)  # (B x T x C)
        return acoustic_out


    def lm_model_step(self, acoustic_out, hidden, prev_out):
        '''
        acoustic_out: tensor, shape of (B x T(1) x C)
        prev_out: tensor, shape of (B x T(1) x pitch)
        '''
        batch_size = acoustic_out.shape[0]
        if not self.recursive:
            sequence_out, hidden_out = self.sequence_model(acoustic_out, hidden)
        else:
            prev_embedded = self.embedding(
                prev_out).flatten(-2)  # (B x T x C)
            combined_out = torch.cat([prev_embedded, acoustic_out], dim=-1)
            sequence_out, hidden_out = self.sequence_model(combined_out, hidden)
        fc_out = self.fc(sequence_out)
        # to shape (batch, class, time, pitch)
        batch_size = acoustic_out.shape[0]
        fc_out = fc_out.view(batch_size, -1, 88, self.n_class).permute(0, 3, 1, 2)
        return F.log_softmax(fc_out, dim=1), hidden_out

    def init_hidden(self):
        # return (torch.zeros(2, 1, self.model_size).cuda(), torch.zeros(2, 1, self.model_size).cuda())
        return (torch.zeros(2, 1, self.model_size).to(next(self.parameters()).device),
         torch.zeros(2, 1, self.model_size).to(next(self.parameters()).device) )
