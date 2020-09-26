import pyaudio
import queue
import time
import numpy as np

CHUNK = 2048
RATE = 44100


### this code is from https://blog.naver.com/chandong83/221149828690
class MicrophoneStream(object):
    def __init__(self, rate, chunk, channels):
        self._rate = rate
        self._chunk = chunk
        self._channels = channels

        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._channels, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )        
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()

        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()
    
    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)



def main():
    # 마이크 열기 
    with MicrophoneStream(RATE, CHUNK, 1) as stream: 
        audio_generator = stream.generator()
        for i in range(1000):
            data = stream._buff.get()
            decoded = np.frombuffer(data, dtype=np.int16) / 32768
            print(stream._buff.qsize(), decoded[0:5])
            # for x in audio_generator:
            #     # 마이크 음성 데이터
            #     print(x)            

if __name__ == '__main__':
    main()
