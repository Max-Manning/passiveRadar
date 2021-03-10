''' Simple Kalman filter based target tracker for a single passive radar
    target. Mostly intended as a simplified demonstration script. For better
    performance, use multitarget_kalman_tracker.py'''

import numpy as np
import matplotlib.pyplot as plt
import zarr
import os
import argparse
from tqdm import tqdm
from celluloid import Camera

from passiveRadar.config import getConfiguration
from passiveRadar.target_detection import simple_target_tracker
from passiveRadar.target_detection import CFAR_2D
from passiveRadar.plotting_tools import persistence


def parse_args():

    parser = argparse.ArgumentParser(
        description="PASSIVE RADAR TARGET TRACKER")

    parser.add_argument(
        '--config',
        required=True,
        type=str,
        help="Path to the configuration file")

    parser.add_argument(
        '--output',
        type=str,
        help="Output type")

    parser.add_argument(
        '--mode',
        choices=['video', 'frames', 'plot'],
        default='plot',
        help="Output a video, video frames or a plot."
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    config = getConfiguration(args.config)
    xambgfile = config['range_doppler_map_fname']
    xambg = np.abs(zarr.load(xambgfile))

    print("Loaded range-doppler maps.")
    Nframes = xambg.shape[2]
    print("Applying CFAR filter...")
    # CFAR filter each frame using a 2D kernel
    CF = np.zeros(xambg.shape)
    for i in tqdm(range(Nframes)):
        CF[:, :, i] = CFAR_2D(xambg[:, :, i], 18, 4)

    print("Applying Kalman Filter...")
    history = simple_target_tracker(
        CF, config['max_range_actual'], config['max_doppler_actual'])

    estimate = history['estimate']
    measurement = history['measurement']
    lockMode = history['lock_mode']

    unlocked = lockMode[:, 0].astype(bool)
    estimate_locked = estimate.copy()
    estimate_locked[unlocked, 0] = np.nan
    estimate_locked[unlocked, 1] = np.nan
    estimate_unlocked = estimate.copy()
    estimate_unlocked[~unlocked, 0] = np.nan
    estimate_unlocked[~unlocked, 1] = np.nan

    if args.mode == 'plot':
        plt.figure(figsize=(12, 8))
        plt.plot(estimate_locked[:, 1],
                 estimate_locked[:, 0], 'b', linewidth=3)
        plt.plot(
            estimate_unlocked[:, 1], estimate_unlocked[:, 0], c='r', linewidth=1, alpha=0.3)
        plt.xlabel('Doppler Shift (Hz)')
        plt.ylabel('Bistatic Range (km)')
        plt.show()

    else:

        if args.mode == 'frames':
            savedir = os.path.join(os.getcwd(),  "IMG")
            if not os.path.isdir(savedir):
                os.makedirs(savedir)
        else:
            figure = plt.figure(figsize=(8, 4.5))
            camera = Camera(figure)

        print("Rendering frames...")
        # loop over frames
        for kk in tqdm(range(Nframes)):

            # add a digital phosphor effect to make targets easier to see
            data = persistence(CF, kk, 20, 0.90)
            data = np.fliplr(data.T)  # get the orientation right

            if args.mode == 'frames':
                # get the save name for this frame
                svname = os.path.join(
                    savedir, 'img_' + "{0:0=3d}".format(kk) + '.png')
                # make a figure
                figure = plt.figure(figsize=(8, 4.5))

            # get max and min values for the color map (this is ad hoc, change as
            # u see fit)
            vmn = np.percentile(data.flatten(), 35)
            vmx = 1.5*np.percentile(data.flatten(), 99)

            plt.imshow(data,
                       cmap='gnuplot2',
                       vmin=vmn,
                       vmax=vmx,
                       extent=[-1*config['max_doppler_actual'],
                               config['max_doppler_actual'], 0, config['max_range_actual']],
                       aspect='auto')

            if kk > 3:
                nr = np.arange(kk)
                decay = np.flip(0.98**nr)
                col = np.ones((kk, 4))
                cc1 = col @ np.diag([0.2, 1.0, 0.7, 1.0])
                cc2 = col @ np.diag([1.0, 0.2, 0.3, 1.0])
                cc1[:, 3] = decay
                cc2[:, 3] = decay
                plt.scatter(
                    estimate_locked[:kk, 1], estimate_locked[:kk, 0], 8,  marker='.', color=cc1)
                plt.scatter(
                    estimate_unlocked[:kk, 1], estimate_unlocked[:kk, 0], 8,  marker='.', color=cc2)

            plt.xlim([-1*config['max_doppler_actual'],
                      config['max_doppler_actual']])
            plt.ylim([0, config['max_range_actual']])

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
            animation = camera.animate(interval=40)  # 25 fps
            animation.save("SIMPLE_TRACKER_VIDEO.mp4", writer='ffmpeg')
