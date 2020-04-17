import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt
import h5py

from signal_utils import normalize

def persistence(X, k, hold, decay):
    '''Add persistence (digital phosphor) effect to sequence of frames
    
    Parameters: 
    X: Input frame stack (NxMxL matrix)
    k: index of frame to acquire
    hold: number of samples to persist
    decay: frame decay rate (should be less than 1)
    
    Returns:
    frame: (NxM matrix) frame k of the original stack with persistence added'''
    frame = np.zeros((X.shape[0], X.shape[1]))

    nf = min(k+1, hold)
    for i in range(nf):
        if k-i >= 0:
            frame = frame + X[:,:,k-i]*decay**i
    return frame

def CFAR_2D(X, fw, gw, thresh):
    '''constant false alarm rate target detection
    
    Parameters:
    fw: CFAR kernel width 
    gw: number of guard cells
    thresh: detection threshold
    
    Returns:
    X with CFAR filter applied'''

    Tfilt = np.ones((fw,fw))/(fw**2 - gw**2)
    e1 = (fw - gw)//2
    e2 = fw - e1 + 1
    Tfilt[e1:e2, e1:e2] = 0

    CR = normalize(X) / (signal.convolve2d(X, Tfilt, mode='same', boundary='wrap') + 1e-10)
    return CR > thresh

def CFAR_sequential(X, rx, ry, gx, gy, thresh = None, order='xy'):
    
    filtx = np.ones((rx,))/(rx - gx)
    ex1 = (rx-gx)//2
    ex2 = rx - ex1 + 1
    filtx[ex1:ex2] = 0

    filty = np.ones((ry,))/(ry - gy)
    ey1 = (ry-gy)//2
    ey2 = ry - ey1 + 1
    filtx[ey1:ey2] = 0

    if order == 'xy':

        C1 = np.zeros(X.shape)
        for i in range(X.shape[1]):
            C1[:,i] = X[:,i] / np.convolve(X[:,i], filtx, mode='same')

        C2 = np.zeros(X.shape)
        for i in range(X.shape[0]):
            C2[i,:] = C1[i,:] / np.convolve(C1[i,:], filty, mode='same')

    else:

        C1 = np.zeros(X.shape)
        for i in range(X.shape[0]):
            C1[i,:] = X[i,:] / np.convolve(X[i,:], filty, mode='same')

        C2 = np.zeros(X.shape)
        for i in range(X.shape[1]):
            C2[:,i] = C1[:,i] / np.convolve(C1[:,i], filtx, mode='same')
    
    if thresh is None:
        return C2
    else:
        return C2 > thresh


if __name__ == "__main__":

    f = h5py.File('..\\XAMBG_1011_256Hz_test2.hdf5', 'r')
    xambg = np.abs(f['/xambg'])
    f.close()

    print("Loaded data!!")

    Nframes = xambg.shape[2]

    # CFAR filtering
    CF = np.zeros(xambg.shape)
    for i in range(Nframes):

        # CFAR filtering using a 2D kernel
        # CF[:,:,i] = CFAR_2D(xambg[:,:,i], 20, 5, 3.0)

        # CFAR filtering along range and doppler separately (range first)
        CF[:,:,i] = CFAR_sequential(xambg[:,:,i], 30, 30, 4, 6, None, 'yx')

    # plot each frame
    for kk in range(Nframes):

        data = persistence(CF, kk, 30, 0.91)
        data = np.fliplr(data.T) # get the orientation right
        
        svname = '..\\IMG2\\img_' + "{0:0=3d}".format(kk) + '.png'
        figure = plt.figure()

        # get max and min values for color map
        vmn = np.percentile(data.flatten(), 1)
        vmx = 1.8*np.percentile(data.flatten(),99)

        # plot
        plt.imshow(data,cmap = 'gnuplot2', vmin=vmn, vmax=vmx, extent = [-256,256,0,250])
        plt.ylabel('Bistatic Range (km)')
        plt.xlabel('Doppler Shift (Hz)')
        
        plt.tight_layout()
        plt.savefig(svname, dpi=100)
        plt.close()