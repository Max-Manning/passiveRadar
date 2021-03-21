import numpy as np
import zarr
import argparse

from passiveRadar.config import getConfiguration
from tracker.multitarget import multitarget_track_and_plot


def parse_args():
    parser = argparse.ArgumentParser(
        description="PASSIVE RADAR - MAIN PROCESSING SCRIPT")

    parser.add_argument(
        '--config',
        required=True,
        type=str,
        help="Path to the configuration file")

    parser.add_argument(
        '--mode',
        choices=['video', 'frames', 'plot'],
        default='plot',
        help="Output a video, video frames or a plot.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = getConfiguration(args.config)

    results = zarr.load('./TEST_DATA')
    multitarget_track_and_plot(config, np.abs(results), args.mode)
    # It might be interesting to remove the np.abs
    # multitarget_track_and_plot(config, results, args.mode)

    # result = run_identification('image_path')  # ['noise', 'noise', 'drone_1']
    # print('Identifier result', result)
