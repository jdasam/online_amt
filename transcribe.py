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

from time import time

class OnlineTranscriber:
    def __init__(self, model, return_roll=True):
        self.model = model
        self.model.eval()
        self.model.melspectrogram.stft.padding = False
        self.audio_buffer = th.zeros((1,5120)).to(th.float)
        self.mel_buffer = model.melspectrogram(self.audio_buffer)
        self.hidden = model.init_hidden()
        self.prev_output = th.zeros((1,1,88)).to(th.long)
        self.buffer_length = 0
        self.sr = 16000
        self.return_roll = return_roll

    def update_buffer(self, audio):
        # audio = librosa.resample(audio, 44100, 16000)
        t_audio = th.tensor(audio).to(th.float)
        new_buffer = th.zeros_like(self.audio_buffer)
        new_buffer[0, :-len(t_audio)] = self.audio_buffer[0, len(t_audio):]
        new_buffer[0, -len(t_audio):] = t_audio
        self.audio_buffer = new_buffer
        # pad_len = math.ceil(len(t_audio) / 512) * 512 - len(t_audio)
        # t_audio = F.pad(t_audio, (0, pad_len))
    
    def update_mel_buffer(self):
        new_mel = th.zeros_like(self.mel_buffer)
        new_mel[:,:,:6] = self.mel_buffer[:,:,1:7]
        new_mel[:,:,6:] = self.model.melspectrogram(self.audio_buffer[:, -2048:])
        self.mel_buffer = new_mel
        return self.mel_buffer
        # new_mel = np.zeros_like(self.mel_buffer)
        # added_audio_samples = self.audio_buffer[:, -2048:]
        # added_mel = self.model.melspectrogram(added_audio_samples)
        # new_mel[:,:,:-1] = self.mel_buffer[:,:,1:]
        # new_mel[:,:,-1:] = added_mel
        # self.mel_buffer = new_mel
        # return added_mel

        # self.mel_buffer = self.model.melspectrogram(self.audio_buffer)
        # return self.mel_buffer

    def inference(self, audio):
        with th.no_grad():
            self.update_buffer(audio)
            last_mel = self.update_mel_buffer()
            acoustic_out = self.model.conv_stack(last_mel.transpose(-1, -2))
            language_out, self.hidden = self.model.lm_model_step(acoustic_out[:,3:4,:], self.hidden, self.prev_output)
            language_out[0,1,0,:] /= 2
            self.prev_output = language_out.argmax(dim=1)
            # print(language_out[0,1,0,30:60], self.prev_output[0,0,30:60])
            out = self.prev_output[0,0,:].numpy()
        if self.return_roll:
            return (out == 2) + (out == 3)
            # return (out==2) +  (out==4)
        else: # return onset and offset only
            out[out==4]=2
            onset_pitches = np.squeeze(np.argwhere(out == 2)).tolist()
            # print('before', onset_pitches)
            off_pitches = np.squeeze(np.argwhere(out == 1)).tolist()
            if isinstance(onset_pitches, int):
                onset_pitches = [onset_pitches]
            if isinstance(off_pitches, int):
                off_pitches = [off_pitches]
            # print('after', onset_pitches, off_pitches)
            return onset_pitches, off_pitches
        # return acoustic_out[:,3:4,:].numpy()

def load_model(args):
    print(args.model_file)
    model_state_path = args.model_file
    checkpoint = th.load(model_state_path, map_location='cpu')
    model_complexity = checkpoint['model_complexity']
    model_name = checkpoint['model_name']
    model_class = getattr(models, model_name)
    recursive = not args.no_recursive

    if model_name == 'ARmodel':
        model = model_class(229, 88, model_complexity, n_class=args.n_class,
            recursive=recursive, context_len=args.context_len)
    if model_name == 'FlexibleModel':
        model = model_class(229, 88, model_complexity, n_class=args.n_class,
            ac_model_type=args.ac_model_type, lm_model_type=args.lm_model_type,
            recursive=recursive, context_len=args.context_len)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model