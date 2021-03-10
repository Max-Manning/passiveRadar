from scipy.signal.signaltools import resample_poly
import yaml
import fractions
import numpy as np


def getConfiguration(config_fname):
    '''sets up parameters for passive radar processing'''

    config_file = open(config_fname, 'r')
    config = yaml.safe_load(config_file)
    config_file.close()

    # get the cpi length in samples - should be a power of 2 for computational
    # efficiency.
    config['cpi_samples'] = nextpow2(config['channel_bandwidth']
                                     * config['cpi_seconds_nominal'])

    # get the desired intermediate frequency (IF) sample rate
    config['desired_IF_sample_rate'] = config['cpi_samples'] \
        / config['cpi_seconds_nominal']

    # get the fractional resampling factor for converting the input data to the
    # desired IF sample rate
    desired_resample_ratio = fractions.Fraction(config['input_sample_rate']
                                                / config['desired_IF_sample_rate'])

    # get the closest fraction to desired_resample_ratio whose denominator is
    # smaller than 20. This gets us close to the desired IF sample rate, while
    # making sure that we're not upsampling by too much in the resampling stage.
    resample_ratio = desired_resample_ratio.limit_denominator(20)
    config['resamp_up'] = resample_ratio.denominator
    config['resamp_dn'] = resample_ratio.numerator
    config['IF_sample_rate'] = config['input_sample_rate'] * config['resamp_up'] \
        / config['resamp_dn']

    # as a result of the approximate rational resampling, the actual cpi
    # duration differs from the nominal value by a small amount.
    config['cpi_seconds_actual'] = config['cpi_samples'] * float(resample_ratio) \
        / config['input_sample_rate']
    config['doppler_cell_width'] = 1 / config['cpi_seconds_actual']

    # the width of each range cell in km
    config['range_cell_width'] = 2.998e5 / config['IF_sample_rate']

    # number of range cells needed to obtaine the desired range
    config['num_range_cells'] = round(config['max_range_nominal']
                                      / config['range_cell_width'])

    # true bistatic range
    config['max_range_actual'] = config['num_range_cells'] \
        * config['range_cell_width']

    # number of doppler cells - is a power of 2 for computational efficiency
    config['num_doppler_cells'] = nearestpow2(2 * config['max_doppler_nominal']
                                              * config['cpi_seconds_actual'])

    # actual maximum doppler shift
    config['max_doppler_actual'] = config['num_doppler_cells'] \
        / (2 * config['cpi_seconds_actual'])

    # frequency offset of the channel from the input center frequency
    config['offset_freq'] = config['input_center_freq'] - \
        config['channel_freq']

    # get the chunk sizes to be used for processing. This depends on whether
    # the CPI sections overlap
    if config['overlap_cpi']:
        config['input_chunk_length'] = int(np.floor(config['cpi_samples']
                                                    * config['resamp_dn'] / config['resamp_up']))
        if config['input_chunk_length'] % 2 != 0:
            config['input_chunk_length'] -= 1
        config['output_chunk_length'] = config['cpi_samples'] // 2
        config['window_overlap'] = config['cpi_samples'] // 4
        config['frame_interval'] = config['cpi_seconds_actual'] / 2
    else:
        config['input_chunk_length'] = int(np.floor(config['cpi_samples']
                                                    * config['resamp_dn'] / config['resamp_up']) * 2)
        config['output_chunk_length'] = config['cpi']
        config['frame_interval'] = config['cpi_seconds_actual']

    config['range_doppler_map_fname'] = config['output_fname'] + '.' \
        + config['range_doppler_map_ftype']

    config['meta_fname'] = config['output_fname'] + '.npz'

    return config


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def nearestpow2(i):
    nextp2 = nextpow2(i)
    prevp2 = nextp2 // 2
    if (nextp2 - i) < (i - prevp2):
        return nextp2
    else:
        return prevp2
