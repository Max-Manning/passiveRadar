import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import yaml
import h5py
import errno
import os

from passiveRadar.signal_utils import normalize
from passiveRadar.target_detection import CFAR_2D
from passiveRadar.plotting_tools import persistence

if __name__ == "__main__":

    ## plot a sequence of passive radar range-doppler maps


    # get config parameters from file
    config_file = open('PRconfig.yaml', 'r')
    config_params = yaml.safe_load(config_file)
    config_file.close()


    xambgfile           = config_params['outputFile']
    blockLength         = config_params['blockLength']
    channelBandwidth    = config_params['channelBandwidth']
    rangeCells          = config_params['rangeCells']
    dopplerCells        = config_params['dopplerCells']

    # length of the coherent processing interval in seconds
    cpi_seconds = blockLength/channelBandwidth
    # range extent in km
    range_extent = rangeCells*3e8/(channelBandwidth*1000)
    # doppler extent in Hz
    doppler_extent = dopplerCells/(2 * cpi_seconds)

    # get the processed passive radar data (range-doppler maps)
    if not os.path.isfile(xambgfile):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), xambgfile)
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

        plt.imshow(data,cmap = 'gnuplot2', vmin=vmn, vmax=vmx, 
            extent = [-1*doppler_extent,doppler_extent,0,range_extent], 
            aspect='auto')

        plt.ylabel('Bistatic Range (km)')
        plt.xlabel('Doppler Shift (Hz)')
        
        plt.tight_layout()
        plt.savefig(svname, dpi=200)
        plt.close()