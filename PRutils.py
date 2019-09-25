import numpy as np
import scipy.signal as signal

def decimate(x, q):
    return signal.decimate(x, q, n=127, ftype='fir', axis=0, zero_phase=False)

def deinterleave_IQ(interleaved_data):
    return interleaved_data[0:-1:2] + 1j*interleaved_data[1::2]

def frequency_shift(x, fc, Fs):
    nn = np.arange(x.shape[0], dtype=np.complex64)
    return x*np.exp(1j*2*np.pi*fc*nn/Fs)

def shift(x, n):
    if n == 0:
        return x
    elif n >= 0:
        e = np.empty_like(x)
        e[:n] = 0
        e[n:] = x[:-n]
        return e
    else:
        e = np.empty_like(x)
        e[n:] = 0
        e[:n] = x[-n:]
        return e


def shift_circ(x, n):
    if n == 0:
        return x
    else:
        e = np.empty_like(x)
        e[:n] = x[-n:]
        e[n:] = x[:-n]
        return e