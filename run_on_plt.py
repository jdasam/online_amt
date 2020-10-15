from transcribe import load_model, OnlineTranscriber
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pyaudio
import numpy as np
import time
import argparse
import queue
from mic_stream import MicrophoneStream
from threading import Thread

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = pyaudio.PyAudio().get_default_input_device_info()['maxInputChannels']
RATE = 16000

def get_buffer_and_transcribe(model, q):
    transcriber = OnlineTranscriber(model)
    with MicrophoneStream(RATE, CHUNK, CHANNELS) as stream:
        audio_generator = stream.generator()
        while True:
            data = stream._buff.get()
            decoded = np.frombuffer(data, dtype=np.int16) / 32768
            if CHANNELS > 1:
                decoded = decoded.reshape(CHANNELS, -1)
                decoded = np.mean(decoded, axis=0)
            frame_output = transcriber.inference(decoded)
            q.put(frame_output)

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

def main(model_file):
    model = load_model(model_file)
    
    q = queue.Queue()
    print("* recording")
    t1 = Thread(target=get_buffer_and_transcribe, name=get_buffer_and_transcribe, args=(model, q))
    t1.start()
    # print('model is running')
    draw_plot(q)
    # print("* done recording")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='model-180000.pt')
    args = parser.parse_args()

    main(args.model_file)