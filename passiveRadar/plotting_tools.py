import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from .signal_utils import normalize

def persistence(X, k, hold, decay):
    '''Add persistence (digital phosphor) effect along the time axis of 
    a sequence of range-doppler maps
    
    Parameters: 

    X: Input frame stack (NxMxL matrix)
    k: index of frame to acquire
    hold: number of samples to persist
    decay: frame decay rate (should be less than 1)
    
    Returns:

    frame: (NxM matrix) frame k of the original stack with persistence effect'''
    
    frame = np.zeros((X.shape[0], X.shape[1]))

    n_persistence_frames = min(k+1, hold)
    for i in range(n_persistence_frames):
        if k-i >= 0:
            frame = frame + X[:,:,k-i]*decay**i
    return frame

