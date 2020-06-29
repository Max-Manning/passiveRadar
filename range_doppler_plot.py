import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

from passiveRadar.signal_utils import normalize
from passiveRadar.target_detection import CFAR_2D
from passiveRadar.plotting_tools import persistence
from passiveRadar.config_params import getConfigParams

if __name__ == "__main__":

    ## plot a sequence of passive radar range-doppler maps
    config_fname = "PRconfig.yaml"
    config = getConfigParams(config_fname)
    xambgfile = config['outputFile']
    
    f = h5py.File(xambgfile, 'r')
    xambg = np.abs(f['/xambg'])
    Nframes = xambg.shape[2]
    f.close()

    # CFAR filter each frame using a 2D kernel
    CF = np.zeros(xambg.shape)
    for i in range(Nframes):
        CF[:,:,i] = CFAR_2D(xambg[:,:,i], 18, 4)

    # make sure the save directory exists
    savedir = os.path.join(os.getcwd(),  "IMG")
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    # plot each frame
    for kk in range(Nframes):

        data = persistence(CF, kk, 20, 0.90)
        data = np.fliplr(data.T) # get the orientation right

        svname = os.path.join(savedir, 'img_' + "{0:0=3d}".format(kk) + '.png')
        
        # svname = '.\\IMG4\\img_' + "{0:0=3d}".format(kk) + '.png'
        figure = plt.figure(figsize = (8, 4.5))

        # get max and min values for color map
        vmn = np.percentile(data.flatten(), 1)
        vmx = 1.8*np.percentile(data.flatten(),99)

        plt.imshow(data,
            cmap = 'gnuplot2', 
            vmin=vmn,
            vmax=vmx, 
            extent = [-1*config['doppler_extent'],config['doppler_extent'],0,config['range_extent']], 
            aspect='auto')

        plt.ylabel('Bistatic Range (km)')
        plt.xlabel('Doppler Shift (Hz)')
        
        plt.tight_layout()
        plt.savefig(svname, dpi=200)
        plt.close()