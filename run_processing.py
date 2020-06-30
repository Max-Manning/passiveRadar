''' Main passive radar processing script '''

import numpy as np
import yaml
import h5py
import dask.array as da
from dask.diagnostics import ProgressBar

from passiveRadar.config_params import getConfigParams
from passiveRadar.signal_utils import offset_compensation, channel_preprocessing
from passiveRadar.clutter_removal import LS_Filter_Multiple
from passiveRadar.range_doppler_processing import fast_xambg


if __name__ == "__main__":

    # get the configuration parameters
    config_fname = "PRconfig.yaml"
    config = getConfigParams(config_fname)
    
    # load passive radar data
    inputFile = h5py.File(config['inputFile'], 'r')
    ref_data = da.from_array(inputFile[config['inputReferencePath']],     
        chunks = (config['chunkLen'],))
    srv_data = da.from_array(inputFile[config['inputSurveillancePath']],  
        chunks = (config['chunkLen'],))

    # make sure there's enough data for the number of frames we want to compute
    maxNFrames = min(ref_data.shape[0]//config['chunkLen'], 
        srv_data.shape[0]//config['chunkLen'])
    nFrames = min(maxNFrames, config['nFrames'])

    # trim input data
    ref_data = ref_data[0:config['nFrames']*config['chunkLen']]
    srv_data = srv_data[0:config['nFrames']*config['chunkLen']]

    # do the channel preparation stuff (de-interleaving complex samples, tuning 
    # to the center frequency of the channel and decimating)
    ref_data = da.map_blocks(channel_preprocessing,
        ref_data,
        config['channel_decim'],
        config['offsetFrequency'],
        config['inputSampleFreq'],
        dtype=np.complex64, 
        chunks=(config['chunkLen']//20,))

    srv_data = da.map_blocks(channel_preprocessing,
        srv_data,
        config['channel_decim'],
        config['offsetFrequency'],
        config['inputSampleFreq'],
        dtype=np.complex64, 
        chunks=(config['chunkLen']//20,))

    # correct the time offset between the channels
    # coarse alignment
    srv_data = da.map_blocks(offset_compensation, 
        ref_data,
        srv_data,
        1e6,
        32,
        dtype=np.complex64,
        chunks=(config['chunkLen']//20,))
    
    # fine alignment
    srv_data = da.map_blocks(offset_compensation,
        ref_data,
        srv_data,
        1e6,
        1,
        dtype=np.complex64,
        chunks=(config['chunkLen']//20,))

    # apply the clutter removal filter
    srv_cleaned = da.map_blocks(LS_Filter_Multiple, 
        ref_data, 
        srv_data, 
        config['LSFilterLength'],
        config['inputSampleFreq']//config['channel_decim'], 
        [0,1,-1,2,-2],
        dtype=np.complex64, 
        chunks = (config['chunkLen']//20,))

    # compute the cross-ambiguity function
    XAMBG = da.map_blocks(fast_xambg, 
        ref_data, 
        srv_cleaned, 
        config['rangeCells'], 
        config['dopplerCells'], 
        dtype=np.complex64, 
        chunks=(config['dopplerCells'], config['rangeCells']+1, 1))

    # create the output file
    f = h5py.File(config['outputFile'])
    d = f.require_dataset('/xambg', shape=XAMBG.shape, dtype=XAMBG.dtype)

    # compute the result
    with ProgressBar():
        da.store(XAMBG, d)
    f.close()