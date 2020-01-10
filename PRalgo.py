import numpy as np
import scipy.signal as signal
from scipy.linalg import solve_toeplitz
from signal_utils import xcorr

def fast_xambg(ref, srv, nlag, nfft):
    ''' Fast Cross-Ambiguity Fuction
    
    Parameters:
        s1, s2: input vectors for cross-ambiguity function
        nlag: number of lag bins to compute
        nfft: number of doppler bins to compute (should be power of 2)
    Returns:
        xc: (nfft, nlag+1, 1) matrix containing cross-ambiguity surface
        third dimension added for easy stacking in dask

    '''
    if ref.shape != srv.shape:
        raise ValueError('Input vectors must have the same length')
    ndecim = int(ref.shape[0]/nfft)
    xambg = np.zeros((nfft, nlag+1, 1), dtype=np.complex64)
    s2c = np.conj(srv)

    for k, lag in enumerate(np.arange(-nlag, 1)):
        sd = np.roll(s2c, lag)*ref
        sd = signal.resample_poly(sd, 1, ndecim)
        xambg[:, k, 0] = np.fft.fftshift(np.fft.fft(sd, nfft))
    return xambg


def LS_Filter(ref, srv, nlag, reg=1):
    '''Least squares clutter removal for passive radar. Computes filter taps
    using the direct matrix inversion method.  
    
    Parameters:
    s1: reference signal
    s2: surveillance signal
    nlag: filter length in samples
    reg: L2 regularization parameter for matrix inversion (default 1)
    
    Returns:
    y: surveillance signal with clutter removed

    '''
    if ref.shape != srv.shape:
        raise ValueError('Input vectors must have the same length')
    A = np.zeros((ref.shape[0], nlag+10), dtype=np.complex64)
    lags = np.arange(-10, nlag)
    
    # compute the data matrix
    for k in range(lags.shape[0]):
        A[:, k] = np.roll(ref, lags[k])
    
    # compute the autocorrelation matrix of ref
    ATA = A.conj().T @ A

    # create the Tikhonov regularization matrix
    K = np.eye(ATA.shape[0], dtype=np.complex64)

    # solve the least squares problem
    w = np.linalg.solve(ATA + K*reg, A.conj().T @ srv)

    # direct but slightly slower implementation:
    # w = np.inv(ATA + K*reg) @ A.conj().T @ srv

    return srv - A @ w



def LS_Filter_SVD(ref, srv, nlag):
    '''Least squares clutter removal for passive radar. Computes filter taps
    using the singular value decomposition. Slower than the direct matrix
    inversion, but guaranteed to be stable.
    
    Parameters:
    ref: reference signal
    srv: surveillance signal
    nlag: filter length in samples
    
    Returns:
    y: surveillance signal with clutter removed

    '''
    if ref.shape != srv.shape:
        raise ValueError('Input vectors must have the same length')
    A = np.zeros((ref.shape[0], nlag+10), dtype=np.complex64)
    lags = np.arange(-10, nlag)

    # create the data matrix
    for k in range(lags.shape[0]):
        A[:, k] = np.roll(ref, lags[k])
    
    #obtain the singular value decomposition
    U, S, V = np.linalg.svd(A, full_matrices=False)

    # compute the filter coefficients 
    w = V.conj().T @ np.diag(1/S) @ U.conj().T @ srv

    return srv - A @ w

def LS_Filter_Toeplitz(ref, srv, nlag):
    '''Least squares clutter removal for passive radar. Computes filter taps
    by assuming that the autocorrelation matrix of the reference channel signal 
    is Hermitian and Toeplitz. Faster than the direct matrix inversion method,
    but inaccurate if the assumptions are violated (i.e. if the input signal is
    not wide sense stationary)
    
    
    Parameters:
    ref: reference signal
    srv: surveillance signal
    nlag: filter length in samples
    
    Returns:
    y: surveillance signal with clutter removed

    '''
    # compute the first column of the autocorelation matrix of ref
    c = xcorr(ref, ref, nlag)

    # compute the cross-correlation matrix of ref and srv
    r = xcorr(srv, ref, nlag)

    # solve the Toeplitz least squares problem
    w = solve_toeplitz(c, r)

    return s2 - np.convolve(s1, w, mode = 'same')


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
    # A: yes, definitely. But I only need to do the offset compensation once so that's OK.

    os = find_channel_offset(s1, s2, ndec, 6*ndec)
    if(os == 0):
        return x2
    else:
        return np.roll(x2, os)