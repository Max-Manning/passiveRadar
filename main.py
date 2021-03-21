import zarr
from dask.diagnostics import ProgressBar
import argparse
from passiveRadar.config import getConfiguration
from passiveRadar.data_processor import process_data


def parse_args():

    parser = argparse.ArgumentParser(
        description="PASSIVE RADAR - MAIN PROCESSING SCRIPT")

    parser.add_argument(
        '--config',
        required=True,
        type=str,
        help="Path to the configuration file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = getConfiguration(args.config)

    xambg = process_data(config)

    # save the result to a zarr file
    outfile = zarr.open(config['range_doppler_map_fname'],
                        mode='w',
                        shape=xambg.shape,
                        chunks=(config['num_doppler_cells'],
                                config['num_range_cells']+1, 1),
                        dtype=xambg.dtype)
    with ProgressBar():
        xambg.to_zarr(outfile)
