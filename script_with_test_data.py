import numpy as np
import zarr
import os
import argparse

from passiveRadar.config import getConfiguration
from tracker.multitarget import multitarget_track_and_plot
from machineLearning.main import identify_drones


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

    image_path = os.path.join(os.getcwd(),  "results/test.jpg")
    print('New image will be printed in', image_path)

    results = zarr.load('./TEST_DATA')
    multitarget_track_and_plot(config, np.abs(results), args.mode, image_path)

    # It might be interesting to remove the np.abs
    # multitarget_track_and_plot(config, results, args.mode)

    result = identify_drones(image_path)
    print('Identification result', result)
