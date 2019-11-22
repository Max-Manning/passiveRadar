import numpy as np
import scipy.signal as signal
from signal_utils import shift, shift_circ

def fast_xambg(s1, s2, nlag, nfft):
    ''' Fast Cross-Ambiguity Fuction
    
    Parameters:
        s1, s2: input vectors for cross-ambiguity function
        nlag: number of lag bins to compute
        nfft: number of doppler bins to compute (should be power of 2)
    Returns:
        xc: (nfft, nlag+1, 1) matrix containing cross-ambiguity surface
        third dimension added for easy stacking in dask

    '''
    if s1.shape != s2.shape:
        raise ValueError('Input vectors must have the same length')
    ndecim = int(s1.shape[0]/nfft)
    xambg = np.zeros((nfft, nlag+1, 1), dtype=np.complex64)
    s2c = np.conj(s2)
    for k, lag in enumerate(np.arange(-nlag, 1)):
        sd = shift_circ(s2c, lag)*s1
        sd = signal.resample_poly(sd, 1, ndecim)
        xambg[:, k, 0] = np.fft.fftshift(np.fft.fft(sd, nfft))
    return xambg

def LS_Filter(s1, s2, nlag, reg=1):
    '''Least squares clutter removal for passive radar
    
    Parameters:
    s1: reference signal
    s2: surveillance signal
    nlag: filter length in samples
    reg: L2 regularization parameter for matrix inversion (default 1)
    
    Returns:
    y: surveillance signal with direct path interference and clutter removed

    '''
    if s1.shape != s2.shape:
        raise ValueError('Input vectors must have the same length')
    A = np.zeros((s1.shape[0], nlag+5), dtype=np.complex64)
    lags = np.arange(-5, nlag)
    for k in range(lags.shape[0]):
        A[:, k] = shift(s1, lags[k])
    ATA = A.conj().T @ A
    K = np.eye(ATA.shape[0], dtype=np.complex64)
    w = np.linalg.solve(ATA + K*reg, A.conj().T @ s2)
    # w = np.inv(ATA + K*reg) @ A.conj().T @ s2
    y = s2 - A @ w

    return y

def find_channel_offset(s1, s2, nd=32, nl=100):
    '''use cross-correlation to find offset between two channels in samples'''
    if s1.shape != s2.shape:
        raise ValueError('Input vectors must have the same length')
    B1 = signal.decimate(s1, nd)
    B2 = np.pad(signal.decimate(s2, nd), (nl, nl), 'constant')
    xc = np.abs(signal.correlate(B1, B2, mode='valid'))
    return (np.argmax(xc) - nl)*nd

def offset_compensation(x1, x2, ndec):
    ''' correct a constant time offset between two similar signals
    
    Parameters:
    x1, x2: input signals (must be the same length)
    ndec: decimation factor for finding offset with cross-correlation
    
    Returns:
    signal x2 shifted to align with x1'''

    s1 = x1[0:1000000]
    s2 = x2[0:1000000]
    # Q: is using a million samples to find the channel offset huge overkill?
    # A: yes, definitely.

    os = find_channel_offset(s1, s2, ndec, 6*ndec)
    if(os == 0):
        return x2
    else:
        return shift(x2, os)