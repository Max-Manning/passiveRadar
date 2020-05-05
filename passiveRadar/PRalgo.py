import numpy as np
import scipy.signal as signal
from scipy.linalg import solve_toeplitz
from passiveRadar.signal_utils import xcorr, shift, frequency_shift

def fast_xambg(refChannel, srvChannel, rangeBins, freqBins):
    ''' Fast Cross-Ambiguity Fuction for processing passive radar data
    
    Parameters:
        refChannel: reference channel data
        srvChannel: surveillance channel data
        rangeBins:  number of range bins to compute
        freqBins:   number of doppler bins to compute (should be power of 2)
    Returns:
        xambg: the cross-ambiguity surface. Dimensions are (nf, nlag+1, 1)
        third dimension added for easy stacking in dask

    '''
    if refChannel.shape != srvChannel.shape:
        raise ValueError('Input vectors must have the same length')

    # calculate decimation factor
    ndecim = int(refChannel.shape[0]/freqBins)

    # pre-allocate space for the result
    xambg = np.zeros((freqBins, rangeBins+1, 1), dtype=np.complex64)

    # complex conjugate of the second input vector
    srvChannelConj = np.conj(srvChannel)

    # precompute FIR filter for decimation. (flat top filter of length
    # 10*decimation factor).  
    dtaps = signal.firwin(10*ndecim + 1, 1. / ndecim, window='flattop')
    dfilt = signal.dlti(dtaps, 1)


    # loop over range bins 
    for k, lag in enumerate(np.arange(-1*rangeBins, 1)):
        channelProduct = np.roll(srvChannelConj, lag)*refChannel
        #decimate the product of the reference channel and the delayed surveillance channel
        xambg[:,k,0] = signal.decimate(channelProduct, ndecim, ftype=dfilt)[0:ndecim]

    # take the FFT along the first axis (Doppler)
    xambg = np.fft.fftshift(np.fft.fft(xambg, axis=0), axes=0)
    return xambg

def fast_xambg_ones(refChannel, srvChannel, rangeBins, freqBins):
    ''' Fast Cross-Ambiguity Fuction for processing passive radar data.

    Uses a simple all-ones decimation filter whose length is equal to the
    decimation factor.
    
    Parameters:
        refChannel: reference channel data
        srvChannel: surveillance channel data
        rangeBins:  number of range bins to compute
        freqBins:   number of doppler bins to compute (should be power of 2)

    Returns:
        xambg: the cross-ambiguity surface. Dimensions are (nf, nlag+1, 1)
        third dimension added for easy stacking in dask

    '''
    if refChannel.shape != srvChannel.shape:
        raise ValueError('Input vectors must have the same length')

    # calculate decimation factor
    ndecim = int(refChannel.shape[0]/freqBins)

    # pre-allocate space for the result
    xambg = np.zeros((freqBins, rangeBins+1, 1), dtype=np.complex64)

    # complex conjugate of the second input vector
    srvChannelConj = np.conj(srvChannel)

    # precompute FIR filter for decimation (all ones filter)
    dtaps = np.ones((ndecim + 1,))
    dfilt = signal.dlti(dtaps, 1)


    # loop over range bins 
    for k, lag in enumerate(np.arange(-1*rangeBins, 1)):
        channelProduct = np.roll(srvChannelConj, lag)*refChannel
        #decimate the product of the reference channel and the delayed surveillance channel
        xambg[:,k,0] = signal.decimate(channelProduct, ndecim, ftype=dfilt)[0:ndecim]

    # take the FFT along the Doppler axis
    xambg = np.fft.fftshift(np.fft.fft(xambg, axis=0), axes=0)

    return xambg

def LS_Filter(refChannel, srvChannel, filterLength, regularization=1, return_filter=False):
    '''Least squares clutter removal for passive radar. Computes filter taps
    using the direct matrix inversion method.  
    
    Parameters:
    
    refChannel:     reference channel signal
    srvChannel:     surveillance channel signal
    filterLength:   length of the least squares filter (in samples)
    regularization: L2 regularization parameter for matrix inversion (default 1.0)
    return_filter:  (bool) option to return filter taps as well as cleaned signal
    
    Returns:

    srvChannelFiltered: surveillance channel signal with radar clutter removed
    filterTaps: (optional) least squares filter taps

    '''
    if refChannel.shape != srvChannel.shape:
        raise ValueError('Input vectors must have the same length')

    
    # Time lag associated with each tap in the least squares filter
    lags = np.arange(-10, filterLength)
    
    # Create a matrix of time-shited copies of the reference channel signal
    A = np.zeros((refChannel.shape[0], filterLength+10), dtype=np.complex64)
    for k in range(lags.shape[0]):
        A[:, k] = np.roll(refChannel, lags[k])
    
    # compute the autocorrelation matrix of ref
    ATA = A.conj().T @ A

    # create the Tikhonov regularization matrix
    K = np.eye(ATA.shape[0], dtype=np.complex64)

    # solve the least squares problem
    filterTaps = np.linalg.solve(ATA + K*regularization, A.conj().T @ srvChannel)

    # direct but slightly slower implementation:
    # filterTaps = np.inv(ATA + K*regularization) @ A.conj().T @ srvChannel

    # Apply the least squares filter to the surveillance channel
    srvChannelFiltered = srvChannel - A @ filterTaps

    if return_filter:
        return srvChannelFiltered, filterTaps
    else:
        return srvChannelFiltered

def LS_Filter_SVD(refChannel, srvChannel, filterLength, return_filter = False):
    '''Least squares clutter removal for passive radar. Computes filter taps
    using the singular value decomposition. Slower than the direct matrix
    inversion, but guaranteed to be stable.
    
    Parameters:

    refChannel:    reference channel signal
    srvChannel:    surveillance channel signal
    filterLength:  filter length in samples
    return_filter: (bool) option to return filter taps as well as cleaned signal
    
    Returns:

    srvChannelFiltered: surveillance channel signal with radar clutter removed
    filterTaps: (optional) least squares filter taps

    '''
    if refChannel.shape != srvChannel.shape:
        raise ValueError('Input vectors must have the same length')

    # Time lag associated with each tap in the least squares filter
    lags = np.arange(-10, filterLength)
    
    # Create a matrix of time-shited copies of the reference channel signal
    A = np.zeros((refChannel.shape[0], filterLength+10), dtype=np.complex64)
    for k in range(lags.shape[0]):
        A[:, k] = np.roll(refChannel, lags[k])
    
    # obtain the singular value decomposition
    U, S, V = np.linalg.svd(A, full_matrices=False)

    # zero out any small singular values 
    eps = 1e-10
    Sinv = 1/S
    Sinv[S < eps] = 0.0

    # compute the filter coefficients 
    filterTaps = V.conj().T @ np.diag(Sinv) @ U.conj().T @ srvChannel


    # Apply the least squares filter to the surveillance channel
    srvChannelFiltered = srvChannel - A @ filterTaps

    if return_filter:
        return srvChannelFiltered, filterTaps
    else:
        return srvChannelFiltered

def LS_Filter_Toeplitz(refChannel, srvChannel, filterLength, return_filter=False):
    '''Least squares clutter removal for passive radar. Computes filter taps
    by assuming that the autocorrelation matrix of the reference channel signal 
    is Hermitian and Toeplitz. Faster than the direct matrix inversion method,
    but inaccurate if the assumptions are violated (i.e. if the input signal is
    not wide sense stationary.)
    
    Parameters:

    refChannel:    reference channel signal
    srvChannel:    surveillance channel signal
    filterLength:  filter length in samples
    return_filter: (bool) option to return filter taps as well as cleaned signal
    
    Returns:

    srvChannelFiltered: surveillance channel signal with radar clutter removed
    filterTaps: (optional) least squares filter taps

    '''
    refChannelShift = np.roll(refChannel, -10)

    # compute the first column of the autocorelation matrix of ref
    autocorr_matrix = xcorr(refChannelShift, refChannelShift, 0, filterLength + 9)

    # compute the cross-correlation of ref and srv
    xcorr_matrix = xcorr(srvChannel, refChannelShift, 0, filterLength + 9)

    # solve the Toeplitz least squares problem
    filterTaps = solve_toeplitz(xcorr_matrix, autocorr_matrix)

    # compute clutter signal and remove from surveillance Channel
    clutter = np.convolve(refChannelShift, filterTaps, mode = 'full')[0:srvChannel.shape[0]]
    srvChannelFiltered = srvChannel - clutter

    if return_filter:
        return srvChannelFiltered, filterTaps
    else:
        return srvChannelFiltered


def offset_compensation(x1, x2, ndec, nlag=2000):
    # correct channel offset bwtween two signals
    s1 = x1[0:1000000]
    s2 = x2[0:1000000]
    os = find_channel_offset(s1, s2, ndec, nlag)
    if(os == 0):
        return x2
    else:
        return shift(x2, os)

def find_channel_offset(s1, s2, nd=32, nl=100):
    '''use cross-correlation to find channel offset in samples'''
    B1 = signal.decimate(s1, nd)
    B2 = np.pad(signal.decimate(s2, nd), (nl, nl), 'constant')
    xc = np.abs(signal.correlate(B1, B2, mode='valid'))
    return (np.argmax(xc) - nl)*nd
