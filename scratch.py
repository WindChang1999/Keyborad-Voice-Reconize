import numpy as np
from scipy.io import wavfile
import epdt

def Get_fft(inputfile):
    f, x = wavfile.read(inputfile)
    x = np.delete(x, 1, axis=1)
    x = np.transpose(x)
    x = x[0]
    x = x/max(abs(x))
    n = len(x)    # 帧总数
    x1,x2 = epdt.epdt(x)
    idata=abs(np.fft.fft(x[x1:x2]))
    data=idata[range(325)]
    x1, x2 = epdt.epdt(x)
    while x1>151:
        idata=abs(np.fft.fft(x[x1:x2]))
        data=np.vstack((data,idata[range(325)]))
        x = np.delete(x, range(x1+10000))
        #n = len(x)
        x1, x2 = epdt.epdt(x)
    return data
