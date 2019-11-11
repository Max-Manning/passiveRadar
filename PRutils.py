import numpy as np
import scipy.signal as signal

def decimate(x, q):
    '''decimate x by a factor of q'''
    return signal.decimate(x, q, n=255, ftype='fir', axis=0, zero_phase=False)

def deinterleave_IQ(interleaved_data):
    '''convert interleaved samples to complex float'''
    return interleaved_data[0:-1:2].astype(np.complex64) + 1j*interleaved_data[1::2].astype(np.complex64)

def frequency_shift(x, fc, Fs):
    '''frequency shift x by fc where Fs is the sample rate'''
    nn = np.arange(x.shape[0], dtype=np.complex64)
    return x*np.exp(1j*2*np.pi*fc*nn/Fs)

def shift(x, n):
    '''shift x by n samples, pad with zeros'''
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
    '''circularly shift x by n samples'''
    if n == 0:
        return x
    else:
        e = np.empty_like(x)
        e[:n] = x[-n:]
        e[n:] = x[:-n]
        return e