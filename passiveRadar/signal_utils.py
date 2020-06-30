''' signal_utils.py: a collection of signal processing utility functions
    for passive radar processing '''
    
import numpy as np
import scipy.signal as signal

def normalize(x):
    '''normalize ndarray to unit mean'''
    return x/np.mean(np.abs(x).flatten())

def decimate(x, q):
    '''decimate x by a factor of q with some settings that I like'''
    return signal.decimate(x, q, 20*q, ftype='fir', axis=0, zero_phase=False)

def deinterleave_IQ(interleavedIQ):
    '''convert interleaved IQ samples to complex64'''
    interleavedIQ = np.array(interleavedIQ)
    return (interleavedIQ[0:-1:2] + 1j*interleavedIQ[1::2]).astype(np.complex64)

def frequency_shift(x, fc, Fs):
    '''frequency shift x by fc where Fs is the sample rate of x'''
    nn = np.arange(x.shape[0], dtype=np.complex64)
    return x*np.exp(1j*2*np.pi*fc*nn/Fs)

def xcorr(s1, s2, nlead, nlag):
    ''' cross-correlated s1 and s2'''
    return signal.correlate(s1, np.pad(s2, (nlag, nlead), mode='constant'),
        mode='valid')

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

def offset_compensation(x1, x2, ns, ndec, nlag=2000):
    '''Find and correct a constant time offset between two signals using a 
    cross-correlation
    
    Parameters:
        s1, s2:     Arrays containing the input signals
        ns:         Number of samples to use for cross-correlation
        ndec:       Decimation factor prior to cross-correlation
        nlag:       Number of lag bins for cross-correlation
    Returns:
        x2s:        The signal x2 time-shifted so that it aligns with x1. Edges 
                    are padded with zeros.         
    '''
    s1 = x1[0:int(ns)]
    s2 = x2[0:int(ns)]

    # cross-correlate to find the offset
    os = find_channel_offset(s1, s2, ndec, nlag)

    if(os == 0):
        return x2
    else:
        return shift(x2, os)

def find_channel_offset(s1, s2, nd, nl):
    '''use cross-correlation to find channel offset in samples'''
    B1 = signal.decimate(s1, nd)
    B2 = np.pad(signal.decimate(s2, nd), (nl, nl), 'constant')
    xc = np.abs(signal.correlate(B1, B2, mode='valid'))
    return (np.argmax(xc) - nl)*nd

def channel_preprocessing(sig, dec, fc, Fs):
    '''deinterleave IQ samples, tune to channel frequency and decimate'''
    IQ = deinterleave_IQ(sig)
    IQ_tuned = frequency_shift(IQ, fc, Fs)
    IQd = decimate(IQ_tuned, dec)
    return IQd

