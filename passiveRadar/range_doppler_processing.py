import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft   # use scipy's fftpack since np.fft.fft 
                                #automatically promotes input data to complex128
                                
from passiveRadar.signal_utils import frequency_shift, xcorr


def fast_xambg(refChannel, srvChannel, rangeBins, freqBins, shortFilt=True):
    ''' Fast Cross-Ambiguity Fuction (frequency domain method)
    
    Parameters:
        refChannel: reference channel data
        srvChannel: surveillance channel data
        rangeBins:  number of range bins to compute
        freqBins:   number of doppler bins to compute (should be power of 2)
        shortFilt:  (bool) chooses the type of decimation filter to use.
                    If True, uses an all-ones filter of length 1*(decimation factor)
                    If False, uses a flat-top window of length 10*(decimation factor)+1
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

    if shortFilt:
        # precompute short FIR filter for decimation (all ones filter with length
        # equal to the decimation factor)
        dtaps = np.ones((ndecim + 1,))
    else:
        # precompute long FIR filter for decimation. (flat top filter of length
        # 10*decimation factor).  
        dtaps = signal.firwin(10*ndecim + 1, 1. / ndecim, window='flattop')

    dfilt = signal.dlti(dtaps, 1)

    # loop over range bins 
    for k, lag in enumerate(np.arange(-1*rangeBins, 1)):
        channelProduct = np.roll(srvChannelConj, lag)*refChannel
        #decimate the product of the reference channel and the delayed surveillance channel
        xambg[:,k,0] = signal.decimate(channelProduct, ndecim, ftype=dfilt)[0:freqBins]

    # take the FFT along the first axis (Doppler)
    # xambg = np.fft.fftshift(np.fft.fft(xambg, axis=0), axes=0)
    xambg = np.fft.fftshift(fft(xambg, axis=0), axes=0)
    return xambg


def direct_xambg(refChannel, srvChannel, rangeBins, freqBins, sampleRate):
    ''' Direct Cross-Ambiguity Fuction (time domain method)
    
    Parameters:
        refChannel: reference channel data
        srvChannel: surveillance channel data
        rangeBins:  number of range bins to compute
        freqBins:   number of doppler bins to compute
        sampleRate: input sample rate in Hz
    Returns:
        xambg: the cross-ambiguity surface. Dimensions are (nf, nlag+1, 1)
        third dimension added for easy stacking in dask

    '''
    if refChannel.shape != srvChannel.shape:
        raise ValueError('Input vectors must have the same length')

    # calculate the coherent processing interval in seconds
    CPI = refChannel.shape[0]/sampleRate

    # pre-allocate space for the result
    xambg = np.zeros((freqBins, rangeBins+1, 1), dtype=np.complex64)

    # loop over frequency bins
    for i in range(freqBins):
        # get Doppler shift for the current bin
        df = (i - 0.5*freqBins)/CPI
        # create a frequency shifted copy of the reference signal
        ref_shifted = frequency_shift(refChannel, df, sampleRate)
        # correlate surveillance and shifted reference signals
        xambg[i,:,0] = xcorr(ref_shifted, srvChannel, rangeBins, 0)
    
    return xambg
