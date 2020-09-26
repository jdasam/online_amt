
import torch
import torch.nn.functional as F
from torch import nn
import random

from .mel import MelSpectrogram
from .constants import *

""" Initial code was from https://github.com/jongwook/onsets-and-frames """

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



class AR_Transcriber(nn.Module):
    def __init__(self, input_features, output_features,
                 model_complexity_conv=48, model_complexity_lstm=48):
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.model_complexity_conv = model_complexity_conv
        self.model_complexity_lstm = model_complexity_lstm

        model_size_conv = model_complexity_conv * 16
        model_size_lstm = model_complexity_lstm * 16
        self.language_hidden_size = model_size_lstm

        self.acoustic_model = ConvStack(input_features, model_size_conv)
        self.language_model = torch.nn.LSTM(model_size_conv + 88*2, model_size_lstm, num_layers=2, batch_first=True, bidirectional=False)   # hidden size 768, num layers 2
        self.language_model.flatten_parameters()

        self.language_post = nn.Sequential(
            torch.nn.Linear(model_size_lstm, 88 * 5)
        )

        self.class_embedding = nn.Embedding(5,2)
        self.melspectrogram = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX)
        

    def forward(self, mel, gt_label=False): # [gt_label: 1x640x88]
        acoustic_out = self.acoustic_model(mel) #[mel: 1x640x229, acoustic_out: 1x640x768]
        if not isinstance(gt_label, bool):
            prev_gt = torch.cat((torch.zeros((gt_label.shape[0], 1, gt_label.shape[2]), device=mel.device, dtype=torch.long), gt_label[:, :-1, :].type(torch.LongTensor).to(mel.device)), dim=1)
            # prev_gt = torch.cat((torch.zeros((gt_label.shape[0], 1, gt_label.shape[2]), device=mel.device, dtype=torch.long), gt_label[:, :-1, :]), dim=1)
            concated_data = torch.cat((acoustic_out, self.class_embedding(prev_gt).view(mel.shape[0], -1, 88 * 2)), dim=2) # [1 640 944]
            # self.language_model.flatten_parameters()
            result, _= self.language_model(concated_data) # [1, 640, 1536], [1, 1, 1536], [1, 1, 1536]
            total_result = self.language_post(result).view(mel.shape[0], -1, 88, 5) # [1, 640, 88, 5]
            # total_result = torch.log_softmax(total_result, dim=3)
            # total_result = torch.argmax(result, dim=3)
        else:
            h, c= self.init_lstm_hidden(mel.shape[0], mel.device)
            prev_out =  torch.zeros((mel.shape[0], 1, 88*2)).to(mel.device)
            total_result = torch.zeros((mel.shape[0], mel.shape[1], 88 )).to(mel.device)
            for i in range(acoustic_out.shape[1]):
                current_data = torch.cat((acoustic_out[:,i:i+1,:], prev_out), dim=2)
                current_out, (h, c) = self.language_model(current_data, (h, c))
                current_out = self.language_post(current_out)
                current_out = current_out.view((mel.shape[0], 1, 88, 5))
                current_out = torch.softmax(current_out, dim=3)
                current_out = torch.argmax(current_out, dim=3)
                prev_out = self.class_embedding(current_out).view(mel.shape[0], 1, 88*2)
                total_result[:,i:i+1,:] = current_out
            
        return total_result

    def lm_model_step(self, acoustic_out, hidden, prev_out):
        '''
        acoustic_out: tensor, shape of (B x T(1) x C)
        prev_out: tensor, shape of (B x T(1) x pitch)
        '''
        # batch_size = acoustic_out.shape[0]
        # prev_embedded = self.class_embedding(prev_out).view(batch_size, 1, 88*2)  # (B x T x C)
        # combined_out = torch.cat([prev_embedded, acoustic_out], dim=2)
        # sequence_out, hidden_out = self.language_model(combined_out, hidden)
        # fc_out = self.language_post(sequence_out)
        # fc_out = fc_out.view(batch_size, -1, 88, 5)
        # return F.softmax(fc_out, dim=3), hidden_out
        prev_embedding = self.class_embedding(prev_out).view(acoustic_out.shape[0], 1, 88*2)
        current_data = torch.cat((acoustic_out, prev_embedding), dim=2)
        current_out, hidden = self.language_model(current_data, hidden)
        current_out = self.language_post(current_out)
        current_out = current_out.view((acoustic_out.shape[0], 1, 88, 5))
        current_out = torch.softmax(current_out, dim=3)
        return current_out, hidden


    def init_lstm_hidden(self, batch_size, device):
        h = torch.zeros(2, batch_size, self.language_hidden_size, device=device)
        c = torch.zeros(2, batch_size, self.language_hidden_size, device=device)
        return (h, c)