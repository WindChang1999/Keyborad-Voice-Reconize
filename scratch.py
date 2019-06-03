import numpy as np
from scipy.io import wavfile
import epdt

def Get_fft(inputfile):
    f, x = wavfile.read(inputfile)
    x = np.delete(x, 1, axis=1)
    x = np.transpose(x)
    x = x[0]
    x = x/max(abs(x))
    x1,x2 = epdt.epdt(x)
    idata=abs(np.fft.fft(x[x1:x2]))
    t = x[x1:x2]
    data=idata[range(325)]
    x = np.delete(x, range(x1 + 7800))
    x1, x2 = epdt.epdt(x)
    t = np.vstack((t,x[x1:x2]))
    while x1>151:
        idata=abs(np.fft.fft(x[x1:x2]))
        t = np.vstack((t, x[x1:x2]))
        data=np.vstack((data,idata[range(325)]))
        x = np.delete(x, range(x1+7800))
        #n = len(x)
        x1, x2 = epdt.epdt(x)
    return data,t
