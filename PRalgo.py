import numpy as np
import scipy.signal as signal
from PRutils import shift, shift_circ


def find_channel_offset(s1, s2, nd=32, nl=100):
    '''use cross-correlation to find channel offset in samples'''
    B1 = signal.decimate(s1, nd)
    B2 = np.pad(signal.decimate(s2, nd), (nl, nl), 'constant')
    xc = np.abs(signal.correlate(B1, B2, mode='valid'))
    return (np.argmax(xc) - nl)*nd


def fast_xambg(s1, s2, nlag, nfft):
    ''' algorithm from Howland (2005): FM radio based bistatic radar'''
    if s1.shape != s2.shape:
        raise ValueError('shape mismatch!!!1')
    ndecim = int(s1.shape[0]/nfft)
    xc = np.zeros((nfft, nlag+1), dtype=np.complex64)
    s2c = np.conj(s2)
    for k, lag in enumerate(np.arange(-nlag, 1)):
        sd = shift_circ(s2c, lag)*s1
        sd = signal.resample_poly(sd, 1, ndecim)
        xc[:, k] = np.fft.fftshift(np.fft.fft(sd, nfft))
    return xc


def LS_Filter(s1, s2, nlag, reg):
    '''Use least squares minimization to remove static echoes'''
    A = np.zeros((s1.shape[0], nlag+5), dtype=np.complex64)
    lags = np.arange(-5, nlag)
    for k in range(lags.shape[0]):
        A[:, k] = shift(s1, lags[k])
    ATA = A.conj().T @ A
    K = np.eye(ATA.shape[0], dtype=np.complex64)
    try:
        w = np.linalg.solve(ATA + K*reg, A.conj().T @ s2)
        # w = np.inv(ATA + K*reg) @ A.conj().T @ s2
    except MemoryError:
        print(s1.shape)
        print(s2.shape)
        print(K.shape)
        print(ATA.shape)
        raise Exception("something bad has happened!!!1")
    y = s2 - A @ w

    return y
