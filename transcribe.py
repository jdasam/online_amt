from collections import defaultdict
from pathlib import Path
import argparse
import tempfile
import shutil
import subprocess
import math

import torch as th
import torch.nn.functional as F
import numpy as np

from autoregressive import models
from autoregressive.mel import MelSpectrogram
from autoregressive.constants import *

from time import time

class OnlineTranscriber:
    def __init__(self, model, return_roll=True):
        self.model = model
        self.model.eval()
        for i in (0, 3, 8):
            self.model.acoustic_model.cnn[i].padding = (0,1)
        for i in (1, 4, 9):
            self.model.acoustic_model.cnn[i] 
        # self.model.melspectrogram = MelSpectrogram(
        #     N_MELS, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX)
        self.model.melspectrogram.stft.padding = False
        self.audio_buffer = th.zeros((1,5120)).to(th.float)
        self.mel_buffer = model.melspectrogram(self.audio_buffer)
        self.acoustic_layer_outputs = self.init_acoustic_layer(self.mel_buffer)
        self.hidden = model.init_lstm_hidden(1, torch.device('cpu'))
        # self.hidden = model.init_hidden()

        self.prev_output = th.zeros((1,1,88)).to(th.long)
        self.buffer_length = 0
        self.sr = 16000
        self.return_roll = return_roll
        
        self.inten_threshold = 0.05
        self.patience = 100
        self.num_under_thr = 0

    def update_buffer(self, audio):
        t_audio = th.tensor(audio).to(th.float)
        new_buffer = th.zeros_like(self.audio_buffer)
        new_buffer[0, :-len(t_audio)] = self.audio_buffer[0, len(t_audio):]
        new_buffer[0, -len(t_audio):] = t_audio
        self.audio_buffer = new_buffer

    def update_mel_buffer(self):
        self.mel_buffer[:,:,:6] = self.mel_buffer[:,:,1:7]
        self.mel_buffer[:,:,6:] = self.model.melspectrogram(self.audio_buffer[:, -2048:])

    
    def init_acoustic_layer(self, input_mel):
        x = input_mel.transpose(-1, -2).unsqueeze(1)
        acoustic_layer_outputs = []
        for i, layer in enumerate(self.model.acoustic_model.cnn):
            x = layer(x)
            if i in [2,7]:
                acoustic_layer_outputs.append(x)
        return acoustic_layer_outputs

    def update_acoustic_out(self, mel):
        x = mel[:,-3:,:].unsqueeze(1)
        layers = self.model.acoustic_model.cnn
        for i in range(3):
            x = layers[i](x)
        self.acoustic_layer_outputs[0][:,:,:-1,:] = self.acoustic_layer_outputs[0][:,:,1:,:]
        self.acoustic_layer_outputs[0][:,:,-1:,:] = x
        x = self.acoustic_layer_outputs[0][:,:,-3:,:]
        for i in range(3,8):
            x = layers[i](x)
        self.acoustic_layer_outputs[1][:,:,:-1,:] = self.acoustic_layer_outputs[1][:,:,1:,:]
        self.acoustic_layer_outputs[1][:,:,-1:,:] = x
        x = self.acoustic_layer_outputs[1]
        for i in range(8,13):
            x = layers[i](x)
        x = x.transpose(1, 2).flatten(-2)
        return self.model.acoustic_model.fc(x)
    
    def switch_on_or_off(self):
        pseudo_intensity = torch.max(self.audio_buffer) - torch.min(self.audio_buffer)
        if pseudo_intensity < self.inten_threshold:
            self.num_under_thr += 1
        else:
            self.num_under_thr = 0

    def inference(self, audio):
        # time_list = []
        with th.no_grad():
            # time_list.append(time())
            self.update_buffer(audio)
            # time_list.append(time())
            self.switch_on_or_off()
            # time_list.append(time())
            if self.num_under_thr > self.patience:
                if self.return_roll:
                    return [0]*88
                else:
                    return [], []
            self.update_mel_buffer()
            # time_list.append(time())
            acoustic_out = self.update_acoustic_out(self.mel_buffer.transpose(-1, -2))
            # time_list.append(time())
            # acoustic_out = self.model.acoustic_model(self.mel_buffer.transpose(-1, -2))
            language_out, self.hidden = self.model.lm_model_step(acoustic_out, self.hidden, self.prev_output)
            # language_out, self.hidden = self.model.lm_model_step(acoustic_out[:,3:4,:], self.hidden, self.prev_output)
            language_out[0,0,:,3:5] *= 2
            self.prev_output = language_out.argmax(dim=3)
            # time_list.append(time())
            # self.prev_output = language_out.argmax(dim=1)
            # print('total: {:.4f}, buffer: {:.4f}, intensity: {:.4f}, mel: {:.4f}, cnn: {:.4f}, rnn: {:.4f}'.format(
            #     time_list[5]-time_list[0], time_list[1]-time_list[0], time_list[2]-time_list[1], time_list[3]-time_list[2], time_list[4]-time_list[3],
            #     time_list[5]-time_list[4],  
            # ))
            out = self.prev_output[0,0,:].numpy()
        if self.return_roll:
            return (out == 2) + (out == 3)
            # return (out==2) +  (out==4)
        else: # return onset and offset only
            out[out==4]=3
            # out[out==4]=2
            onset_pitches = np.squeeze(np.argwhere(out == 3)).tolist()
            off_pitches = np.squeeze(np.argwhere(out == 1)).tolist()
            if isinstance(onset_pitches, int):
                onset_pitches = [onset_pitches]
            if isinstance(off_pitches, int):
                off_pitches = [off_pitches]
            # print('after', onset_pitches, off_pitches)
            return onset_pitches, off_pitches
        # return acoustic_out[:,3:4,:].numpy()


def load_model(filename):
    parameters = th.load(filename, map_location=th.device('cpu'))
    model = models.AR_Transcriber(229,
                            88,
                            parameters['model_complexity_conv'],
                            parameters['model_complexity_lstm'])

    model.load_state_dict(parameters['model_state_dict'])
    return model
