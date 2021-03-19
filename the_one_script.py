import numpy as np
import zarr
import dask.array as da
import scipy.signal as signal
from dask.diagnostics import ProgressBar
import multiprocessing.popen_spawn_posix
from dask.distributed import Client, performance_report
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler, visualize
import argparse

from passiveRadar.config import getConfiguration
from passiveRadar.signal_utils import find_channel_offset, \
    deinterleave_IQ, frequency_shift, resample, preprocess_kerberossdr_input
from passiveRadar.clutter_removal import LS_Filter_Multiple, NLMS_filter
from passiveRadar.range_doppler_processing import fast_xambg

import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from celluloid import Camera

from passiveRadar.target_detection import simple_target_tracker
from passiveRadar.target_detection import CFAR_2D
from passiveRadar.plotting_tools import persistence


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


def simple_kalman_tracker(config, xambg):
    print("Loaded range-doppler maps.")
    Nframes = xambg.shape[2]
    print("Applying CFAR filter...")
    print('NFrames', Nframes)
    print('xambg.shape', xambg.shape)
    # CFAR filter each frame using a 2D kernel
    CF = np.zeros(xambg.shape)
    for i in tqdm(range(Nframes)):
        res = CFAR_2D(xambg[:, :, i], 18, 4)
        CF[:, :, i] = res

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


def main(config):
    print("-------------------------------------------------------------------")
    print("   RUNNING PASSIVE RADAR PROCESSING   ")
    print("-------------------------------------------------------------------")
    print(f"Input file Reference: {config['input_file_ref']}")
    print(f"Input file Surveilance: {config['input_file_srv']}")
    print(f"Using radio channel centered at {config['channel_freq']/1e6:.1f} MHz"
          f" with bandwidth {config['channel_bandwidth']/1e3:.1f} kHz")
    print(f"Input sample rate {config['input_sample_rate']/1e6:.1f} MHz"
          f", IF sample rate {config['IF_sample_rate']/1e3:.1f} kHz")
    print(f"Maximum bistatic range {config['max_range_actual']:.2f} km "
          f"with range resolution {config['range_cell_width']:.2f} km")
    print(f"Maximum Doppler shift {config['max_doppler_actual']:.2f} Hz"
          f" with Doppler resolution {config['doppler_cell_width']:.4f} Hz")

    refInputFile = preprocess_kerberossdr_input(np.fromfile(
        open(config['input_file_ref']), dtype=np.float32))
    svrInputFile = preprocess_kerberossdr_input(np.fromfile(
        open(config['input_file_srv']), dtype=np.float32))

    # get the first few hundred thousand samples of data and use it to
    #  estimate the offset between the channels
    refc1 = refInputFile[0:20*config['cpi_samples']]
    srvc1 = svrInputFile[0:20*config['cpi_samples']]

    # TODO: Something is wrong with the offset. I'm setting manually to 0
    offset = find_channel_offset(refc1, srvc1, 1, 5000000)

    print('Offset', offset)

    refc1 = deinterleave_IQ(refc1)
    srvc1 = deinterleave_IQ(srvc1)

    if offset > 0:
        ref_data = da.from_array(refInputFile[offset:],
                                 chunks=(config['input_chunk_length'],))
        srv_data = da.from_array(svrInputFile[:-offset],
                                 chunks=(config['input_chunk_length'],))

    elif offset < 0:
        ref_data = da.from_array(refInputFile[:-offset],
                                 chunks=(config['input_chunk_length'],))
        srv_data = da.from_array(svrInputFile[offset:],
                                 chunks=(config['input_chunk_length'],))
    else:
        ref_data = da.from_array(refInputFile,
                                 chunks=(config['input_chunk_length'],))
        srv_data = da.from_array(svrInputFile,
                                 chunks=(config['input_chunk_length'],))

    # de-interleave IQ samples
    ref_data = da.map_blocks(deinterleave_IQ, ref_data,
                             meta=np.zeros(
                                 (config['input_chunk_length']//2,), dtype=np.complex64),
                             dtype=np.complex64, chunks=(config['input_chunk_length']//2,))
    srv_data = da.map_blocks(deinterleave_IQ, srv_data,
                             meta=np.zeros(
                                 (config['input_chunk_length']//2,), dtype=np.complex64),
                             dtype=np.complex64, chunks=(config['input_chunk_length']//2,))

    print(f"Successfully loaded data.")
    print(f"Corrected a sample offset of {offset} samples between channels")

    # trim the data to an integer number of block lengths
    N_chunks_ref = ref_data.shape[0] // (config['input_chunk_length']//2)
    N_chunks_srv = srv_data.shape[0] // (config['input_chunk_length']//2)
    N_chunks = min(N_chunks_ref, N_chunks_srv, config['num_frames']) - 1
    ref_data = ref_data[0:N_chunks*config['input_chunk_length']//2]
    srv_data = srv_data[0:N_chunks*config['input_chunk_length']//2]

    # make sure that block-wise frequency shifting doesn't introduce phase
    # discontinuities at the block edges - this is avoided in the frequency_shift
    # function by adding an appropriate starting phase to each block
    n_chunks = len(ref_data.chunks[0])
    mod_period = config['input_sample_rate'] // config['offset_freq']
    offset_samples_per_block = (config['input_chunk_length']//2) % mod_period
    block_numbers = da.arange(n_chunks, chunks=(1,))
    block_phase_offsets = 2*np.pi*block_numbers*offset_samples_per_block * \
        (config['offset_freq'] / config['input_sample_rate'])

    # tune to the center frequency of the channel
    ref_data = da.map_blocks(frequency_shift,
                             ref_data,
                             config['offset_freq'],
                             config['input_sample_rate'],
                             block_phase_offsets,
                             # meta = np.zeros((config['input_chunk_length']//2,), dtype=np.complex64),
                             dtype=np.complex64,
                             chunks=(config['input_chunk_length']//2,))

    srv_data = da.map_blocks(frequency_shift,
                             srv_data,
                             config['offset_freq'],
                             config['input_sample_rate'],
                             block_phase_offsets,
                             # meta = np.zeros((config['input_chunk_length']//2,), dtype=np.complex64),
                             dtype=np.complex64,
                             chunks=(config['input_chunk_length']//2,))

    # resample to the desired bandwidth
    ref_data = da.map_blocks(resample,
                             ref_data,
                             config['resamp_up'],
                             config['resamp_dn'],
                             # meta = np.zeros((config['output_chunk_length'],), dtype=np.complex64),
                             dtype=np.complex64,
                             chunks=(config['output_chunk_length'],))

    srv_data = da.map_blocks(resample,
                             srv_data,
                             config['resamp_up'],
                             config['resamp_dn'],
                             # meta = np.zeros((config['output_chunk_length'],), dtype=np.complex64),
                             dtype=np.complex64,
                             chunks=(config['output_chunk_length'],))

    # apply the block least squares filter
    srv_cleaned = da.map_blocks(LS_Filter_Multiple,
                                ref_data,
                                srv_data,
                                config['num_range_cells'],
                                config['IF_sample_rate'],
                                # remove clutter at 0Hz, +/-1Hz, +/-2Hz
                                [0, 1, -1, 2, -2],
                                dtype=np.complex64,
                                chunks=(config['output_chunk_length'],))

    # This is making everything NAN
    if config['overlap_cpi']:
        # pad chunks with overlapping sections
        ref_data = da.overlap.overlap(
            ref_data, depth=config['window_overlap'], boundary=0)
        srv_cleaned = da.overlap.overlap(
            srv_cleaned, depth=config['window_overlap'], boundary=0)

    window = signal.get_window(('kaiser', 5.0), config['cpi_samples'])

    # use the cross-ambiguity function to compute range-doppler maps
    xambg = da.map_blocks(fast_xambg,
                          ref_data,
                          srv_cleaned,
                          config['num_range_cells'],
                          config['num_doppler_cells'],
                          config['cpi_samples'],
                          window,
                          dtype=np.complex64,
                          chunks=(config['num_doppler_cells'], config['num_range_cells']+1, 1))

    print(f"Saving range-doppler maps to to {config['range_doppler_map_fname']},"
          f" metadata to {config['meta_fname']}")
    print(f"Output shape: {xambg.shape}, dtype: {xambg.dtype}")

    frame_timestamps = np.arange(xambg.shape[2])*config['frame_interval']
    range_bins = np.arange(xambg.shape[1])*config['range_cell_width']
    doppler_bins = np.arange(-1*xambg.shape[0], xambg.shape[0]) \
        * config['doppler_cell_width']

    np.savez(config['meta_fname'], frame_timestamps=frame_timestamps,
             range_bins=range_bins, doppler_bins=doppler_bins)

    return xambg
    # xambg.visualize(filename='transpose.svg')

    # # save the result to a zarr file
    # outfile = zarr.open(config['range_doppler_map_fname'],
    #                     mode='w',
    #                     shape=xambg.shape,
    #                     chunks=(config['num_doppler_cells'],
    #                             config['num_range_cells']+1, 1),
    #                     dtype=xambg.dtype)
    # with ProgressBar():
    #     xambg.to_zarr(outfile)


if __name__ == "__main__":
    args = parse_args()
    config = getConfiguration(args.config)

    results = main(config)
    simple_kalman_tracker(config, np.abs(results))
    # result = run_identification('image_path')  # ['noise', 'noise', 'drone_1']

    # client = Client()
    # with performance_report(filename="dask-report.html"):
    #     xambg.compute()
    # with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
    #     xambg.compute()
    #     visualize([prof, rprof, cprof])

    # xambg.visualize(filename='transpose.svg')
