from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from Recording import Record
import time

if __name__ == '__main__':
    FILENAME = 'backgroundnoise'
    INPUTFILE = Record(0.05, FILENAME)
    f, x = wavfile.read(INPUTFILE+'.wav')
    x = np.delete(x, 1, axis=1)
    x = np.transpose(x)
    x = x[0]
    print(f)
    print(x)
    fft = np.fft.fft(x)
    fig, ax = plt.subplots(2,1)
    ax[0].plot(abs(fft))
    ax[1].plot(x)
    plt.show()
