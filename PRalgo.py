import numpy as np
import scipy.signal as signal
from PRutils import shift, shift_circ


def find_channel_offset(s1, s2, nd=32, nl=100):
    B1 = signal.decimate(s1, nd)
    B2 = np.pad(signal.decimate(s2, nd), (nl, nl), 'constant')
    xc = np.abs(signal.correlate(B1, B2, mode='valid'))
    return (np.argmax(xc) - nl)*nd


def fast_xambg(s1, s2, nlag, nfft):
    ''' algorithm from Howland (2005): FM radio based bistatic radar'''
    ndecim = int(s1.shape[0]/nfft)
    xc = np.zeros((nfft, 2*nlag+1), dtype=np.float)
    s2c = np.conj(s2)
    for k, lag in enumerate(np.arange(-nlag, nlag+1)):
        sd = shift_circ(s2c, lag)*s1
        sd = signal.resample_poly(sd, 1, ndecim)
        xc[:, k] = np.abs(np.fft.fftshift(np.fft.fft(sd, nfft)))
    return xc


def LS_Filter(s1, s2, nlag, reg):

    A = np.zeros((s1.shape[0], nlag+10), dtype=np.complex64)
    lags = np.arange(-10, nlag)
    for k in range(lags.shape[0]):
        A[:, k] = shift(s1, lags[k])
    ATA = A.conj().T @ A
    K = np.eye(ATA.shape[0])
    w = np.linalg.solve(ATA + K*reg, A.conj().T @ s2)
    y = s2 - A @ w

    return y


def Apply_LS_Filter(s1, s2, nl, reg, nblocks=5):
    '''Apply a LS filter'''
    Bl = 524288
    y = np.zeros(s2.shape, dtype=np.complex64)

    for j in range(nblocks):
        yb = LS_Filter(s1[j*Bl:(j+1)*Bl], s2[j*Bl:(j+1)*Bl], nl, reg)
        y[j*Bl:(j+1)*Bl] = yb

    return y

