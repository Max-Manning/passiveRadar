''' Plotting script for passive radar data. Plots a sequence of range-doppler 
    maps as separate images.'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

from passiveRadar.signal_utils import normalize
from passiveRadar.target_detection import CFAR_2D
from passiveRadar.plotting_tools import persistence
from passiveRadar.config_params import getConfigParams

if __name__ == "__main__":

    config_fname = "PRconfig.yaml"
    config = getConfigParams(config_fname)
    xambgfile = config['outputFile']
    f = h5py.File(xambgfile, 'r')
    xambg = np.abs(f['/xambg'])
    Nframes = xambg.shape[2]
    f.close()

    # Apply a CFAR filter each frame
    # (btw this step is pretty slow so if you're running this a bunch of times you
    # may want to make an intermediate save point after this)
    CF = np.zeros(xambg.shape)
    for i in range(Nframes):
        CF[:,:,i] = CFAR_2D(xambg[:,:,i], 18, 4)

    # make sure the save directory exists (create it if not)
    savedir = os.path.join(os.getcwd(),  "IMG")
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    # loop over frames
    for kk in range(Nframes):

        # add a digital phosphor effect to make targets easier to see
        data = persistence(CF, kk, 20, 0.90)
        data = np.fliplr(data.T) # get the orientation right

        # get the save name for this frame
        svname = os.path.join(savedir, 'img_' + "{0:0=3d}".format(kk) + '.png')
        
        # make a figure
        figure = plt.figure(figsize = (8, 4.5))

        # get max and min values for the color map (this is ad hoc, change as
        # u see fit)
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