import numpy as np
import matplotlib.pyplot as plt
import h5py
import zarr
import os
import argparse
from tqdm import tqdm
from celluloid import Camera

from passiveRadar.signal_utils import normalize
from passiveRadar.target_detection import CFAR_2D
from passiveRadar.plotting_tools import persistence
from passiveRadar.config import getConfiguration

def parse_args():

    parser = argparse.ArgumentParser(
        description="PASSIVE RADAR VIDEO RENDERER")

    parser.add_argument(
        '--config',
        required=True,
        type=str,
        help="Path to the configuration file")
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['video', 'frames'],
        default='video',
        help="Path to the configuration file")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    config = getConfiguration(args.config)
    xambgfile = config['range_doppler_map_fname']
    CFARfile = config['output_fname'] + "_CFAR.zarr"

    if config['range_doppler_map_ftype'] == 'hdf5':
        f = h5py.File(xambgfile, 'r')
        xambg = np.abs(f['/xambg'])
        f.close()   
    else:
        xambg = np.abs(zarr.load(xambgfile))
    Nframes = xambg.shape[2]

    print("Loaded range-doppler map frames.")
    print("Applying CFAR filter...")

    # Apply a constant false alarm rate (CFAR) filter to each frame
    CF = np.zeros(xambg.shape)
    for i in tqdm(range(Nframes)):
        CF[:,:,i] = CFAR_2D(xambg[:,:,i], 18, 4)

    if args.mode == 'frames':
        savedir = os.path.join(os.getcwd(),  "IMG")
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
    else:
        fig = plt.figure(figsize = (8, 4.5))
        camera = Camera(fig)

    print("Rendering frames...")
    # loop over frames
    for kk in tqdm(range(Nframes)):

        # add a digital phosphor effect to make targets easier to see
        data = persistence(CF, kk, 20, 0.90)
        data = np.fliplr(data.T) # get the orientation right

        if args.mode == 'frames':
            # get the save name for this frame
            svname = os.path.join(savedir, 'img_' + "{0:0=3d}".format(kk) + '.png')
            
            # make a figure
            figure = plt.figure(figsize = (8, 4.5))

        # get max and min values for the color map (this is ad hoc, change as
        # u see fit)
        vmn = np.percentile(data.flatten(), 35)
        vmx = 1.5*np.percentile(data.flatten(),99)

        plt.imshow(data,
            cmap = 'gnuplot2', 
            vmin=vmn,
            vmax=vmx, 
            extent = [-1*config['max_doppler_actual'],config['max_doppler_actual'],0,config['max_range_actual']], 
            aspect='auto')

        plt.ylabel('Bistatic Range (km)')
        plt.xlabel('Doppler Shift (Hz)')
        plt.tight_layout()

        if args.mode == 'frames':
            plt.savefig(svname, dpi=200)
            plt.close()
        else:
            camera.snap()
    
    if args.mode == 'video':
        print("Animating...")
        animation = camera.animate(interval=40) # 25 fps
        animation.save("RADAR_VIDEO.mp4", writer='ffmpeg')
