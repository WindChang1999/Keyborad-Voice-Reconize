import numpy as np
from scipy.signal import lfilter
import enframe

def epdt(x):
    FrameLen = 256 # 帧长
    Inc = 100 # 分帧未重叠部分
    amp1 = 0.2 #短时能量阈值
    zcr1 = 50  # 过零率阈值
    status = 0 # 记录语音段的状态
    fx1 = 0

    tmp1 = enframe.enframe(x[:-1], FrameLen, Inc,1)
    tmp2 = enframe.enframe(x[1:], FrameLen, Inc,1)
    signs = (tmp1 * tmp2) < 0
    diffs = (tmp1 - tmp2) > 0.02 # 度量相邻两个采样点之间距离，如果大于门限0.02(经验值)，则1，否则0
    zcr = np.sum(signs * diffs, axis=1) # 行求和得到各帧的过零率；
    # 计算短时能量
    amp = np.sum((enframe.enframe(lfilter([1 , - 0.9375], [1,0], x), FrameLen, Inc,1))** 2, axis=1) # 预加重

    # 开始端点检测
    for n in range(len(zcr)): # Length（zcr）得到的是整个信号的帧数。
        if status == 0 :
            if amp[n] > amp1 or zcr[n] > zcr1 : # 确信进入语音段
                fx1 = n  # 记录语音段的起始点
                status = 1
            else:               # 静音状态
                status = 0
        if status == 1 :
            break
    x1 = (fx1+1) * Inc + 51  # 记录起始点数
    x2 = x1 + 649
    return x1, x2
