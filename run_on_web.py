from flask import Flask, render_template, jsonify
import pyaudio
from transcribe import load_model, OnlineTranscriber
from mic_stream import MicrophoneStream
import numpy as np
from threading import Thread
import queue
import json
import rtmidi

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
print('http://127.0.0.1:5000/')
app = Flask(__name__)
global Q
Q = queue.Queue()



@app.route('/')
def home():
    # args = Args()
    # model = load_model(args)
    model = load_model('/Users/jeongdasaem/Documents/model_weights/model-180000.pt')
    global Q
    t1 = Thread(target=get_buffer_and_transcribe, name=get_buffer_and_transcribe, args=(model, Q))
    t1.start()
    return render_template('home.html')

@app.route('/_amt', methods= ['GET', 'POST'])
def amt():
    global Q
    onsets = []
    offsets = []
    while Q.qsize() > 0:
        rst = Q.get()
        onsets += rst[0]
        offsets += rst[1]
    # if results['on'] != []:
    #     print(results['on'])
    return jsonify(on=onsets, off=offsets)
    # return jsonify(transcription_result=result)
# class Args:
#     def __init__(self):
#         self.model_file = '/Users/jeongdasaem/Documents/model_weights/model-128000.pt'
#         self.rep_type = 'base'
#         self.n_class = 5
#         self.ac_model_type = 'simple_conv'
#         self.lm_model_type = 'lstm'
#         self.context_len = 1
#         self.no_recursive = False        
def get_buffer_and_transcribe(model, q):
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()
    if available_ports:
        midiout.open_port(0)
    else:
        midiout.open_virtual_port("My virtual output")

    stream = MicrophoneStream(RATE, CHUNK, CHANNELS)
    transcriber = OnlineTranscriber(model, return_roll=False)
    with MicrophoneStream(RATE, CHUNK, CHANNELS) as stream:
        # 마이크 데이터 핸들을 가져옴 
        audio_generator = stream.generator()
        print("* recording")
        on_pitch = []
        while True:
            data = stream._buff.get()
            decoded = np.frombuffer(data, dtype=np.int16) / 32768
            if CHANNELS == 2:
                decoded = decoded.reshape(-1, CHANNELS)
                decoded = np.mean(decoded, axis=1)
                # decoded = decoded[:,1]
            frame_output = transcriber.inference(decoded)
            on_pitch += frame_output[0]
            for pitch in frame_output[0]:
                note_on = [0x90, pitch + 21, 64]
                midiout.send_message(note_on)
            for pitch in  frame_output[1]:
                note_off = [0x90, pitch + 21, 0]
                pitch_count = on_pitch.count(pitch)
                [midiout.send_message(note_off) for i in range(pitch_count)]
            on_pitch = [x for x in on_pitch if x not in frame_output[1]]
            q.put(frame_output)
            # print(sum(frame_output))
        stream.closed = True
    print("* done recording")

if __name__ == '__main__':
    app.run(debug=True)
