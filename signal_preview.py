import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

from passiveRadar.config import getConfiguration
from passiveRadar.signal_utils import find_channel_offset, \
    deinterleave_IQ, resample, frequency_shift, xcorr

def parse_args():

    parser = argparse.ArgumentParser(
        description="SIGNAL PREVIEW")
    
    parser.add_argument(
        '--config',
        required=True,
        type=str,
        help="Path to the configuration file")

    return parser.parse_args()


def signal_preview(config):

    inputFile = h5py.File(config['input_file'], 'r')

    if config['interleaved_input_channels']:

        data = inputFile[config['interleaved_data_path']][0:config['input_chunk_length']]
        data_IQ = deinterleave_IQ(data)
        
        ref = data_IQ[0::2]
        srv = data_IQ[1::2]

        offset = find_channel_offset(ref,srv,4,50000)

    else:

        # get the first few seconds of data and use it to estimate the
        # offset between the channels
        ref = inputFile[config['input_ref_path']][0:config['input_chunk_length']]
        srv = inputFile[config['input_srv_path']][0:config['input_chunk_length']]

        ref = deinterleave_IQ(ref)
        srv = deinterleave_IQ(srv)
        
        offset = find_channel_offset(ref,srv,4,50000)

    plt.figure()
    plt.psd(ref, NFFT=8192, Fs=config['input_sample_rate'], 
        Fc=config['input_center_freq'])
    plt.psd(srv, NFFT=8192, Fs=config['input_sample_rate'], 
        Fc=config['input_center_freq'])
    plt.legend(['Reference channel', 'Surveillance channel'])
    plt.title("Spectrum of Input Data")
    plt.show()

    reff = frequency_shift(ref, 
        config['offset_freq'], config['input_sample_rate'])
    srvv = frequency_shift(srv, 
        config['offset_freq'], config['input_sample_rate'])

    reff = resample(reff, config['resamp_up'], config['resamp_dn'])
    srvv = resample(srvv, config['resamp_up'], config['resamp_dn'])

    plt.figure()
    plt.psd(reff, NFFT=2048, Fs=config['channel_bandwidth'], 
        Fc=config['channel_freq'])
    plt.psd(srvv, NFFT=2048, Fs=config['channel_bandwidth'], 
        Fc=config['channel_freq'])
    plt.legend(['Reference channel', 'Surveillance channel'])
    plt.title("Channel Spectrum")
    plt.show()

    fig, ax = plt.subplots()
    xx = np.arange(-2000, 2001)
    xc = np.abs(xcorr(ref, srv, 2000, 2000))
    plt.plot(xx, xc)
    plt.title("Cross-correlation between channels")
    plt.text(0.1, 0.9, f"Offset of {offset} samples between channels", transform=ax.transAxes)
    plt.show()

if __name__ == "__main__":
    
    args = parse_args()
    config = getConfiguration(args.config)

    signal_preview(config)