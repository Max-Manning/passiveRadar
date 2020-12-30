import numpy as np
import scipy.signal as signal
from scipy.linalg import solve_toeplitz
from passiveRadar.signal_utils import xcorr, frequency_shift

def LS_Filter(refChannel, srvChannel, filterLen, reg=1.0, peek=10, 
    return_filter=False):
    '''Block least squares adaptive filter. Computes filter taps using the 
    direct matrix inversion method.  
    
    Parameters:
        refChannel:     Array containing the reference channel signal
        srvChannel:     Array containing the surveillance channel signal
        filterLen:      Length of the least squares filter (in samples)
        reg:            L2 regularization parameter for the matrix inversion 
                        (default 1.0)
        peek:           Number of noncausal filter taps. Set to zero for a 
                        causal filter. If nonzero, clutter estimates can depend 
                        on future values of the reference signal (this helps 
                        sometimes)
        return_filter:  Boolean indicating whether to return the filter taps

    Returns:
        srvChannelFiltered: Surveillance channel signal with clutter removed
        filterTaps:     (optional) least squares filter taps

    '''
    if refChannel.shape != srvChannel.shape:
        raise ValueError('Input vectors must have the same length')

    lags = np.arange(-1*peek, filterLen)
    
    # Create a matrix of time-shited copies of the reference channel signal
    A = np.zeros((refChannel.shape[0], filterLen+peek), dtype=np.complex64)
    for k in range(lags.shape[0]):
        A[:, k] = np.roll(refChannel, lags[k])
    
    # compute the autocorrelation matrix of ref
    ATA = A.conj().T @ A

    # create the Tikhonov regularization matrix
    K = np.eye(ATA.shape[0], dtype=np.complex64)

    # solve the least squares problem
    filterTaps = np.linalg.solve(ATA + K*reg, A.conj().T @ srvChannel)

    # direct but slightly slower implementation:
    # filterTaps = np.inv(ATA + K*reg) @ A.conj().T @ srvChannel

    # Apply the least squares filter to the surveillance channel
    srvChannelFiltered = srvChannel - A @ filterTaps

    if return_filter:
        return srvChannelFiltered, filterTaps
    else:
        return srvChannelFiltered

def LS_Filter_SVD(refChannel, srvChannel, filterLen, peek=10, 
    return_filter = False):
    '''Block least squares adaptive filter. Computes filter taps using the 
    singular value decomposition. Slower than the direct matrix inversion 
    method, but doesn't get upset if the data matrix is close to singular.
    
    Parameters:
        refChannel:     Array containing the reference channel signal
        srvChannel:     Array containing the surveillance channel signal
        filterLen:   Length of the least squares filter (in samples)
        peek:           Number of noncausal filter taps. Set to zero for a 
                        causal filter. If nonzero, clutter estimates can depend 
                        on future values of the reference signal (this helps 
                        sometimes)
        return_filter:  Boolean indicating whether to return the filter taps

    Returns:
        srvChannelFiltered: Surveillance channel signal with clutter removed
        filterTaps:     (optional) least squares filter taps

    '''
    if refChannel.shape != srvChannel.shape:
        raise ValueError('Input vectors must have the same length')

    # Time lag associated with each tap in the least squares filter
    lags = np.arange(-1*peek, filterLen)
    
    # Create a matrix of time-shited copies of the reference channel signal
    A = np.zeros((refChannel.shape[0], filterLen+peek), dtype=np.complex64)
    for k in range(lags.shape[0]):
        A[:, k] = np.roll(refChannel, lags[k])
    
    # obtain the singular value decomposition
    U, S, VH = np.linalg.svd(A, full_matrices=False)

    # zero out any small singular values 
    eps = 1e-10
    Sinv = 1/S
    Sinv[S < eps] = 0.0

    # compute the filter coefficients 
    filterTaps = VH.conj().T @ np.diag(Sinv) @ U.conj().T @ srvChannel

    # Apply the least squares filter to the surveillance channel
    srvChannelFiltered = srvChannel - A @ filterTaps

    if return_filter:
        return srvChannelFiltered, filterTaps
    else:
        return srvChannelFiltered

def LS_Filter_Toeplitz(refChannel, srvChannel, filterLen, peek=10, 
        return_filter=False):
    '''Block east squares adaptive filter. Computes filter coefficients using
    scipy's solve_toeplitz function. This assumes the autocorrelation matrix of 
    refChannel is Hermitian and Toeplitz (i.e. wide the reference signal is
    wide sense stationary). Faster than the direct matrix inversion method but
    inaccurate if the assumptions are violated. 
    
    Parameters:
        refChannel:     Array containing the reference channel signal
        srvChannel:     Array containing the surveillance channel signal
        filterLen:   Length of the least squares filter (in samples)
        peek:           Number of noncausal filter taps. Set to zero for a 
                        causal filter. If nonzero, clutter estimates can depend 
                        on future values of the reference signal (this helps 
                        sometimes)
        return_filter:  Boolean indicating whether to return the filter taps

    Returns:
        srvChannelFiltered: Surveillance channel signal with clutter removed
        filterTaps:     (optional) least squares filter taps

    '''

    if refChannel.shape != srvChannel.shape:
        raise ValueError(f'''Input vectors must have the same length - 
        got {refChannel.shape} and {srvChannel.shape}''')

    # shift reference channel because for some reason the filtering works 
    # better when you allow the clutter filter to be noncausal
    refChannelShift = np.roll(refChannel, -1*peek)

    # compute the first column of the autocorelation matrix of ref
    autocorrRef = xcorr(refChannelShift, refChannelShift, 0, 
        filterLen + peek - 1)

    # compute the cross-correlation of ref and srv
    xcorrSrvRef = xcorr(srvChannel, refChannelShift, 0, 
        filterLen + peek - 1)

    # solve the Toeplitz least squares problem
    filterTaps = solve_toeplitz(autocorrRef, xcorrSrvRef)

    # compute clutter signal and remove from surveillance Channel
    clutter = np.convolve(refChannelShift, filterTaps, mode = 'full')
    clutter = clutter[0:srvChannel.shape[0]]
    srvChannelFiltered = srvChannel - clutter

    if return_filter:
        return srvChannelFiltered, filterTaps
    else:
        return srvChannelFiltered

def LS_Filter_Multiple(refChannel, srvChannel, filterLen, sampleRate, 
    dopplerBins = [0]):
    '''Clutter removal with least squares filter. This function allows the least
    squares filter to be applied over several frequency bins. Useful when there
    is significant clutter at nonzero doppler values.

    Parameters:
        refChannel:     Array containing the reference channel signal
        srvChannel:     Array containing the surveillance channel signal
        filterLen:      Length of the least squares filter (in samples)
        dopplerBins:    List of doppler bins to filter (default only 0Hz)

    Returns:
        srvChannelFiltered: Surveillance channel signal with clutter removed
    '''

    srvChannelFiltered = srvChannel
    for doppler in dopplerBins:
        if doppler == 0:
            srvChannelFiltered = LS_Filter_Toeplitz(refChannel, 
                srvChannelFiltered, filterLen)
        else:
            refMod = frequency_shift(refChannel, doppler, sampleRate)
            srvChannelFiltered = LS_Filter_Toeplitz(refMod, srvChannelFiltered,
                filterLen)
    return srvChannelFiltered

def NLMS_filter(refChannel, srvChannel, filterLen, mu, peek=10, 
    initialTaps=None, returnFilter=False):

    '''Normalized least mean square (NLMS) adaptive filter

    Parameters:
        refChannel:     Array containing the reference channel signal
        srvChannel:     Array containing the surveillance channel signal
        filterLen:   Length of the least squares filter (in samples)
        mu:             Adaptive step size for updating filter weights.
                        Typically < 0.1.
        peek:           Number of noncausal filter taps. Set to zero for a 
                        causal filter. If nonzero, clutter estimates can depend 
                        on future values of the reference signal (this helps 
                        sometimes)
        return_filter:  Boolean indicating whether to return the filter taps

    Returns:
        srvChannelFiltered: Surveillance channel signal with clutter removed
        filterTaps:     (optional) filter taps at the last timestep
    '''

    def NLMS_update(u, d, w0, mu):
        ''' a single NLMS update step'''
        e = d - w0.conj().T @ u
        w1 = w0 + mu*u*e.conj()/(u.conj().T @ u)
        return w1, e


    if initialTaps is None:
        # initialize the filter taps to zeros
        w = np.zeros((filterLen+peek,), dtype = np.complex64)
    else:
        # use the specified initial filter taps
        w = initialTaps
        # choose the appropriate filter length
        filterLen = initialTaps.shape[0] - peek

    # initialize input vector
    refVector = np.flip(refChannel[0:filterLen+peek])

    # an array for the result
    srvChannelFiltered = np.zeros(srvChannel.shape, dtype = np.complex64)

    # loop over input samples
    for k in range(srvChannel.shape[0] - filterLen - peek):

        # update the input vector
        refVector = np.append(refChannel[filterLen+k+peek], refVector[:-1])

        # use the NLMS algorithm to get the new error estimate and update the 
        # filter coefficients
        w, e = NLMS_update(refVector, srvChannel[k+filterLen], w, mu)

        # save the output
        srvChannelFiltered[filterLen+k] = e

    if returnFilter:
        return srvChannelFiltered, w
    else:
        return srvChannelFiltered

def GAL_JPE(refChannel, srvChannel, latticeLen, delayLineLen, mu1, mu2, 
    peek=10, return_filter=False):
    ''' Gradient Adaptive Lattice Joint Process Estimator

    Consists of an adaptive lattice stage which uses the gradient adaptive 
    lattice algorithm followed by an adaptive transversal (tapped delay line) 
    stage which uses the normalized least mean squares algorithm.

    A lot more computationally expensive than the NLMS filter but converges
    faster, especially when the spectrum of the clutter signal isn't flat   
    
    Parameters:
        refChannel:     Array containing the reference channel signal
        srvChannel:     Array containing the surveillance channel signal
        latticeLen:     Length of the adaptive lattice stage (in samples)
        delayLineLen:   Length of the output filter (in samples)
        mu1:            Adaptive step size for updating the lattice filter 
                        reflection coefficients. Ballpark value ~1e-3.
        mu2:            Adaptive step size for updating the transversal output
                        filter coefficients. Ballpark value ~1e-2.
        peek:           Number of noncausal filter taps. Set to zero for a 
                        causal filter. If nonzero, clutter estimates can depend 
                        on future values of the reference signal (this helps 
                        sometimes)
        return_filter:  Boolean indicating whether to return the filter taps

    Returns:
        srvChannelFiltered: Surveillance channel signal with clutter removed
        k:              (optional) lattice filter reflection coefficients at the
                        last timestep
        h:              (optional) transversal filter coefficients at the last 
                        timestep
    '''

    # make sure everything's ok with the input values
    if refChannel.shape != srvChannel.shape:
        raise ValueError("Input vectors must have the same length")

    if latticeLen < delayLineLen:
        partialLattice = True
    elif latticeLen == delayLineLen:
        partialLattice = False
    else:
        raise ValueError("""Delay line order must be greater than
                            or equal to the lattice filter order""")

    # Initialize joint process estimator parameters
    N = refChannel.shape[0] # number of time steps
    # forward prediction errors
    f   = np.zeros((delayLineLen,), dtype=np.complex64) 
    # backward prediction errors
    b   = np.zeros((delayLineLen,), dtype=np.complex64)
    # old backward prediction coefficients 
    bo  = np.zeros((delayLineLen,), dtype=np.complex64) 
    # lattice reflection coefficients
    k   = np.zeros((delayLineLen,), dtype=np.complex64) 
    # normalization factor
    P   = np.zeros((delayLineLen,), dtype=np.complex64) + 1e-8 
    # transversal filter coefficients
    h   = np.zeros((delayLineLen,), dtype=np.complex64) 

    # Additional parameters (feel free to experiment but empirically i've found
    # that these ones are ok)
    beta = 0.9; gamma = 0.999; delta = 1e-8

    # an array for the output values
    srvChannelFiltered = np.zeros(srvChannel.shape, dtype = np.complex64)

    for n in range(N - peek - 1): # loop over sample times
        bo  = b.copy() # save backward prediction errors from the last timestep
        f[0] = refChannel[n + peek]
        b[0] = refChannel[n + peek]

        # lattice filter prediction
        for m in range(1, latticeLen):
            f[m] = f[m-1]  - np.conj(k[m])*bo[m-1] # forward prediction error
            b[m] = bo[m-1] - k[m]*f[m-1]           # backward prediction error
        
        # lattice filter coefficient update
        for m in range(1, latticeLen): 
            # instantaneous energy estimate
            Em = np.abs(f[m-1])**2 + np.abs(bo[m-1])**2
            # filtered energy estimate
            P[m-1] = beta*P[m-1] + (1.0 - beta**2)*Em
            # instantaneous gradient estimate
            grad = np.conj(f[m-1])*b[m] + bo[m-1]*np.conj(f[m])
            # reflection coefficient update
            k[m] = k[m] + mu1*grad/(P[m-1] + 1e-10)
        
        if partialLattice:
            # if length of the lattice filter is less than the length of the 
            # transversal filter then the rest is just a delay line
            # (k[m] = 0 for latticeLen < m < delayLineLen)
            b[latticeLen:] = bo[latticeLen-1:-1]

            for m in range(latticeLen, delayLineLen):
                # get the remaining backward prediction errors
                b[m] = bo[m-1]
        

        # get the adaptive filter output
        e = srvChannel[n] - h.conj().T @ b
        # update transversal filter coefficients according to the NLMS algorithm
        h = h + mu2*np.conj(e)*b/(b.conj().T @ b + 1e-10)
        # save the result
        srvChannelFiltered[n] = e
    
        # variable step size for lattice filter (starts big and gets small)
        mu1 = min(gamma*mu1 + delta*e**2, 5e-3)
    
    if return_filter:
        return srvChannelFiltered, k, h
    else:
        return srvChannelFiltered
        
