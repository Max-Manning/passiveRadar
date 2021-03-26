''' signal_utils.py: a collection of signal processing utility functions
    for passive radar processing '''

from numba import jit
import cupy as np
import scipy.signal as signal


@jit(nopython=True)
def normalize(x):
    '''normalize ndarray to unit mean'''
    return x/np.mean(np.abs(x).flatten())


@jit(nopython=True)
def decimate(x, q):
    '''decimate x by a factor of q'''
    return signal.decimate(x, q, 20*q, ftype='fir', axis=0)


def resample(x, up, dn):
    '''rational resampling by a factor of up/dn'''
    return signal.resample_poly(x, up, dn, padtype='line')


def deinterleave_IQ(interleavedIQ):
    '''convert interleaved IQ samples to complex64'''
    interleavedIQ = np.array(interleavedIQ)
    return (interleavedIQ[0:-1:2] + 1j*interleavedIQ[1::2]).astype(np.complex64)


@jit(nopython=True)
def frequency_shift(x, fc, Fs, phase_offset=0):
    '''frequency shift x by fc where Fs is the sample rate of x'''
    nn = np.arange(x.shape[0], dtype=np.complex64)
    return x*np.exp(1j*2*np.pi*fc*nn/Fs + 1j*phase_offset)


@jit(nopython=True)
def xcorr(s1, s2, nlead, nlag):
    ''' cross-correlate s1 and s2 with sample offsets from -1*nlag to nlead'''
    return signal.correlate(s1, np.pad(s2, (nlag, nlead), mode='constant'),
                            mode='valid')


def find_channel_offset(s1, s2, nd, nl):
    '''use cross-correlation to find channel offset in samples'''
    B1 = signal.decimate(s1, nd)
    B2 = np.pad(signal.decimate(s2, nd), (nl, nl), 'constant')
    xc = np.abs(signal.correlate(B1, B2, mode='valid'))
    return (np.argmax(xc) - nl)*nd


def preprocess_kerberossdr_input(arr):
    '''
    Found this solution from https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array/41191127
    '''
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = arr[idx]

    # Clip data to avoid having overflow when multiplying too big numbers
    return np.clip(out, -1e+10, 1e+10)
