''' Kalman filter based target tracker that can handle multiple targets'''

import numpy as np
import matplotlib.pyplot as plt
import zarr
import os
import argparse
from tqdm import tqdm
from celluloid import Camera

from passiveRadar.config import getConfiguration
from passiveRadar.target_detection import multitarget_tracker
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
        '--mode',
        choices=['video', 'frames', 'plot'],
        default='plot',
        help="Output a video, video frames or a plot."
    )

    return parser.parse_args()


def multitarget_track_and_plot(config, xambg):
    print("Loaded range-doppler maps.")
    Nframes = xambg.shape[2]
    print("Applying CFAR filter...")
    # CFAR filter each frame using a 2D kernel
    CF = np.zeros(xambg.shape)
    for i in tqdm(range(Nframes)):
        CF[:, :, i] = CFAR_2D(xambg[:, :, i], 18, 4)

    print("Applying Kalman Filter...")
    # run the target tracker
    N_TRACKS = 10
    tracker_history = multitarget_tracker(CF,
                                          [config['max_doppler_actual'],
                                              config['max_range_actual']],
                                          N_TRACKS)

    # find the indices of the tracks where there are confirmed targets
    tracker_status = tracker_history['status']
    tracker_status_confirmed_idx = np.nonzero(tracker_status != 2)

    # get the range and doppler values for each target track
    tracker_range = np.squeeze(tracker_history['estimate'][:, :, 0]).copy()
    tracker_doppler = np.squeeze(tracker_history['estimate'][:, :, 1]).copy()

    # if the target is uncorfirmed change the range/doppler values to nan
    tracker_range[tracker_status_confirmed_idx] = np.nan
    tracker_doppler[tracker_status_confirmed_idx] = np.nan

    if args.mode == 'plot':
        # # plot the tracks
        plt.figure(figsize=(16, 10))
        plt.scatter(tracker_doppler, tracker_range, marker='.')
        plt.xlabel("Doppler Shift (Hz)")
        plt.ylabel("Bistatic Range (km)")
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
                cc1 = np.ones((kk, 4))
                cc1 = cc1 @ np.diag([0.2, 1.0, 0.7, 1.0])
                cc1[:, 3] = decay

                cc1 = np.expand_dims(cc1, 1)
                cc1 = np.tile(cc1, (1, 10, 1))

                plt.scatter(tracker_doppler[:kk, :].flatten(), tracker_range[:kk, :].flatten(
                ), 8,  marker='.', color=cc1.reshape(10*kk, 4))

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
            animation.save("TRACKER_VIDEO.mp4", writer='ffmpeg')


if __name__ == "__main__":
    args = parse_args()
    config = getConfiguration(args.config)

    xambgfile = config['range_doppler_map_fname']
    xambg = np.abs(zarr.load(xambgfile))

    multitarget_track_and_plot(config, xambg)
