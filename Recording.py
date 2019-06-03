from pyaudio import PyAudio, paInt16
import wave
from time import localtime, time, strftime
from os import path

def Record(RECORD_SECONDS, FILENAME):
    ''' Record a RECORD_SECONDS wav audio, and return the .wav file path'''
    CHUNK = 1024
    FORMAT = paInt16
    CHANNELS = 2
    RATE = 44100

    # 用当前时间做文件名
    # WAVE_OUTPUT_FILENAME = strftime('%m_%d_%H_%M_%S', localtime(time())) + '.wav'

    WAVE_OUTPUT_FILENAME = FILENAME
    OUTPUT_PATH = 'recordings'
    p = PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print('*'*30+'Recording'+'*'*30)
    frames = []

    # 一共读 fs * 录音时间个数据点, 除以CHUNK得到帧数
    for i in range(0, int(RATE * RECORD_SECONDS / CHUNK)):
        data = stream.read(CHUNK)
        frames.append(data)

    print('*'*30+'Done recording'+'*'*30)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(path.join(OUTPUT_PATH, WAVE_OUTPUT_FILENAME+'.wav'), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return path.join(OUTPUT_PATH, WAVE_OUTPUT_FILENAME)