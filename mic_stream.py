import pyaudio
import queue
import time
import numpy as np

CHUNK = 2048
RATE = 44100

class MicrophoneStream(object):
    """마이크 입력 클래스"""
    def __init__(self, rate, chunk, channels):
        self._rate = rate
        self._chunk = chunk
        self._channels = channels

        # 마이크 입력 버퍼 생성
        self._buff = queue.Queue()
        self.closed = True

    # 클래스 열면 발생함.
    def __enter__(self):
        # pyaudio 인터페이스 생성
        self._audio_interface = pyaudio.PyAudio()
        # 16비트, 모노로 마이크 열기
        # 여기서 _fill_buffer 함수가 바로 callback함수 인데
        # 실제 버퍼가 쌓이면 이곳이 호출된다.
        # 즉, _fill_buffer 마이크 입력을 _fill_buffer 콜백함수로 전달 받음
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._channels, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )        
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        # 클래스 종료시 발생
        # pyaudio 종료
        self._audio_stream.stop_stream()
        self._audio_stream.close()

        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()
    
    # 마이크 버퍼가 쌓이면(CHUNK = 1600) 이 함수 호출 됨. 
    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        # 마이크 입력 받으면 큐에 넣고 리턴
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    # 제너레이터 함수 
    def generator(self):
        #클래스 종료될 떄까지 무한 루프 돌림 
        while not self.closed:
            
            # 큐에 데이터를 기다림.
            # block 상태임.
            chunk = self._buff.get()

            # 데이터가 없다면 문제 있음
            if chunk is None:
                return

            # data에 마이크 입력 받기
            data = [chunk]

            # 추가로 받을 마이크 데이터가 있는지 체크 
            while True:
                try:
                    # 데이터가 더 있는지 체크
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    # 데이터 추가
                    data.append(chunk)
                except queue.Empty:
                    # 큐에 데이터가 더이상 없다면 break
                    break

            #마이크 데이터를 리턴해줌 
            yield b''.join(data)
# [END audio_stream]



def main():
    # 마이크 열기 
    with MicrophoneStream(RATE, CHUNK, 1) as stream:
        # 마이크 데이터 핸들을 가져옴 
        audio_generator = stream.generator()
        for i in range(1000):
            # 1000번만 마이크 데이터 가져오고 빠져나감.
            # print(stream._buff.qsize())
            data = stream._buff.get()
            decoded = np.frombuffer(data, dtype=np.int16) / 32768
            print(stream._buff.qsize(), decoded[0:5])
            # for x in audio_generator:
            #     # 마이크 음성 데이터
            #     print(x)            
            time.sleep(0.001)

if __name__ == '__main__':
    main()
