import cupy as np
import os
import argparse

from passiveRadar.config import getConfiguration
from tracker.multitarget import multitarget_track_and_plot
from machineLearning.main import identify_drones
from passiveRadar.data_processor import process_data


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

    image_path = os.path.join(os.getcwd(),  "results/one_script_processed.jpg")

    results = process_data(config)
    multitarget_track_and_plot(config, np.abs(results), 'plot', image_path)
    result = run_identification('image_path')  # ['noise', 'noise', 'drone_1']

    identification_result = identify_drones(image_path)
    print('Identification result', identification_result)

    # client = Client()
    # with performance_report(filename="dask-report.html"):
    #     xambg.compute()
    # with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
    #     xambg.compute()
    #     visualize([prof, rprof, cprof])

    # xambg.visualize(filename='transpose.svg')
