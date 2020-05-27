import numpy as np
import yaml
import h5py
import dask.array as da
from dask.diagnostics import ProgressBar

from passiveRadar.signal_utils import offset_compensation, channel_preprocessing
from passiveRadar.clutter_removal import LS_Filter_Multiple
from passiveRadar.range_doppler_processing import fast_xambg

if __name__ == "__main__":

    ## Example passive radar processing script

    # get config parameters from file
    config_file = open('PRconfig.yaml', 'r')
    config_params = yaml.safe_load(config_file)
    config_file.close()

    blockLength      = config_params['blockLength']
    sampleFrequency  = config_params['inputSampleRate']
    channelBandwidth = config_params['channelBandwidth']
    inputCenterFreq  = config_params['inputCenterFreq']
    channelFreq      = config_params['channelFreq']
    LSFilterLength   = config_params['LSFilterLength']
    rangeCells       = config_params['rangeCells']
    dopplerCells     = config_params['dopplerCells']
    nFrames          = config_params['nFrames']

    inputReferencePath = config_params['inputReferencePath']
    inputSurveillancePath = config_params['inputSurveillancePath']

    # channel offset from the center frequency
    offsetFrequency  = inputCenterFreq - channelFreq

    # decimation factor
    channel_decim = sampleFrequency//channelBandwidth

    # length of the coherent processing interval in seconds
    cpi_seconds = blockLength/channelBandwidth

    # range extent in km
    range_extent = rangeCells*3e8/(channelBandwidth*1000)

    # number of input samples for chunk processing
    # (*2 for de-interleaving complex samples)
    chunkLen = blockLength*channel_decim*2
    
    # load passive radar data
    inputFile = h5py.File(config_params['inputFile'], 'r')
    ref_data = da.from_array(inputFile[inputReferencePath],     
        chunks = (chunkLen,))
    srv_data = da.from_array(inputFile[inputSurveillancePath],  
        chunks = (chunkLen,))

    # make sure there's enough data for the number of frames we want to compute
    maxNFrames = min(ref_data.shape[0]//chunkLen, srv_data.shape[0]//chunkLen)
    nFrames = min(maxNFrames, nFrames)

    # trim input data
    ref_data = ref_data[0:nFrames*chunkLen]
    srv_data = srv_data[0:nFrames*chunkLen]

    # do the channel preparation stuff (de-interleaving complex samples, tuning 
    # to the center frequency of the channel and decimating)
    ref_data = da.map_blocks(channel_preprocessing, ref_data, channel_decim,
         offsetFrequency, sampleFrequency, dtype=np.complex64, 
            chunks=(chunkLen//20,))
    srv_data = da.map_blocks(channel_preprocessing, srv_data, channel_decim,
         offsetFrequency, sampleFrequency, dtype=np.complex64, 
            chunks=(chunkLen//20,))

    # correct the time offset between the channels
    # coarse alignment
    srv_data = da.map_blocks(offset_compensation, ref_data, srv_data, 1e6,
        32, dtype=np.complex64, chunks = (chunkLen//20,))
    # fine alignment
    srv_data = da.map_blocks(offset_compensation, ref_data, srv_data, 1e6,
        1, dtype=np.complex64, chunks = (chunkLen//20,))

    # apply the clutter removal filter
    arg1 = (ref_data, srv_data, LSFilterLength, sampleFrequency//channel_decim, 
        [0, 1, -1, 2, -2])
    srv_cleaned = da.map_blocks(LS_Filter_Multiple, *arg1, dtype=np.complex64,
         chunks = (chunkLen//20,))

    # compute the cross-ambiguity function
    xarg = (ref_data, srv_cleaned, rangeCells, dopplerCells)
    XAMBG = da.map_blocks(fast_xambg, *xarg, dtype=np.complex64, 
        chunks=(dopplerCells, rangeCells+1, 1))

    # create the output file
    f = h5py.File(config_params['outputFile'])
    d = f.require_dataset('/xambg', shape=XAMBG.shape, dtype=XAMBG.dtype)

    # compute the result
    with ProgressBar():
        da.store(XAMBG, d)

    f.close()