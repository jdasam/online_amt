from transcribe import load_model, OnlineTranscriber

import matplotlib
from matplotlib.animation import FuncAnimation
matplotlib.use('Qt5Agg')
import pyaudio
import wave
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import time
import struct
import argparse
import queue
from mic_stream import MicrophoneStream
from threading import Thread

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "output.wav"


# def prepare_stream_model():
#     model = load_model(args)
#     transcriber = OnlineTranscriber(model)
#     piano_roll = np.zeros((88, 32))
#     piano_roll[30, 0] = 1 # for test
    
#     plt.ion()
#     fig, ax = plt.subplots()

def get_buffer_and_transcribe(model, q):
    transcriber = OnlineTranscriber(model)
    with MicrophoneStream(RATE, CHUNK, CHANNELS) as stream:
        audio_generator = stream.generator()
        while True:
            data = stream._buff.get()
            decoded = np.frombuffer(data, dtype=np.int16) / 32768
            frame_output = transcriber.inference(decoded)
            q.put(frame_output)
            # new_roll = np.zeros_like(piano_roll)
            # new_roll[:, :-1] = piano_roll[:,1:]
            # new_roll[:, -1] = frame_output
            # piano_roll = new_roll
        # return piano_roll, transcriber

def draw_plot(q):
    piano_roll = np.zeros((88, 64 ))
    piano_roll[30, 0] = 1 # for test

    plt.ion()
    fig, ax = plt.subplots()

    plt.show(block=False)
    img = ax.imshow(piano_roll)
    ax_background = fig.canvas.copy_from_bbox(ax.bbox)
    ax.invert_yaxis()
    fig.canvas.draw()

    while True:
        updated_frames = []
        while q.qsize():
            updated_frames.append(q.get())
        num_updated = len(updated_frames)
        if num_updated == 0:
            continue
        new_roll = np.zeros_like(piano_roll)
        if num_updated == 1:
            new_roll[:, :-1] = piano_roll[:,1:]
            new_roll[:, -1] = updated_frames[0]
        else:
            new_roll[:, :-num_updated] = piano_roll[:,num_updated:]
            # new_roll[:, -num_updated] = frame_output
            new_roll[:, -num_updated:] = np.asarray(updated_frames).T
        piano_roll = new_roll
        fig.canvas.restore_region(ax_background)
        img.set_data(piano_roll)
        ax.draw_artist(img)
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()
        time.sleep(0.02)

def main(args):
    model = load_model(args)
    

    
    # 마이크 데이터 핸들을 가져옴 
    q = queue.Queue()
    print("* recording")
    t1 = Thread(target=get_buffer_and_transcribe, name=get_buffer_and_transcribe, args=(model, q))
    t1.start()
    draw_plot(q)
    # t2 = Thread(target=draw_plot, name=draw_plot, args=(fig, img, ax,ax_background, piano_roll))
    # t2.start()
    print("* done recording")


    # librosa.output.write_wav('lib_out.wav', np.concatenate(entire_frames), sr=44100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='/Users/1112919/Documents/ar_model_weights/model-210000.pt')
    parser.add_argument('--rep_type', default='base')
    parser.add_argument('--n_class', default=5, type=int)
    parser.add_argument('--ac_model_type', default='simple_conv', type=str)
    parser.add_argument('--lm_model_type', default='lstm', type=str)
    parser.add_argument('--context_len', default=1, type=int)
    parser.add_argument('--no_recursive', action='store_true')
    args = parser.parse_args()

    main(args)