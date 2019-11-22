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
    for i in range(hold):
        if k-i >= 0:
            frame = frame + X[:,:,k-i]*decay**i
    return frame

def CFAR(X, fw, gw, thresh):
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

    CR = X / (signal.convolve2d(X, Tfilt, mode='same', boundary='wrap') + 1e-10)
    return CR > thresh

if __name__ == "__main__":

    f = h5py.File('xambg\\XAMBG_1011_1019M_250km_256Hz_filt120_012.hdf5', 'r')
    xambg = normalize(abs(f['/xambg'][:,:,:]))
    f.close()

    # xambg = xambg[:,:,100:200]

    Nframes = xambg.shape[2]

    # CFAR filtering
    CF = np.zeros_like(xambg)
    for i in range(Nframes):
        CF[:,:,i] = CFAR(xambg[:,:,i], 20, 5, 3)
    
    # plot each frame
    for kk in range(Nframes):
        
        svname = '.\\IMG2\\img_' + "{0:0=3d}".format(kk) + '.png'
        
        figure = plt.figure()

        data = persistence(CF, kk, 20, 0.88)
        data = np.fliplr(data.T) # get the orientation rights

        # crop to region of interest
        data = data[:, 64:448]

        vmn = np.percentile(data.flatten(), 1)
        vmx = 1.5*np.percentile(data.flatten(),99)
        plt.imshow(data,cmap = 'jet', vmin=vmn, vmax=vmx, extent = [-192,192,0,250])
        plt.ylabel('Bistatic Range (km)')
        plt.xlabel('Doppler Shift (Hz)')
    #    plt.plot(np.abs(mframes[kk,:]))
        # plt.show()
        
        plt.tight_layout()
        plt.savefig(svname, dpi=300)
        plt.close()