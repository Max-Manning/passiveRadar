import yaml

def getConfigParams(config_fname):

    # get config parameters from file
    config_file = open(config_fname, 'r')
    config = yaml.safe_load(config_file)
    config_file.close()

    # channel offset from the center frequency
    config['offsetFrequency']  = config['inputCenterFreq'] - config['channelFreq']

    # decimation factor
    config['channel_decim'] = config['inputSampleFreq']//config['channelBandwidth']

    # number of input samples for chunk processing
    # (*2 for de-interleaving complex samples)
    config['chunkLen'] = config['blockLength']*config['channel_decim']*2
    
    # length of the coherent processing interval in seconds
    config['cpi_seconds'] = config['blockLength']/config['channelBandwidth']

    # range extent in km
    config['range_extent'] = config['rangeCells']*3e8/(config['channelBandwidth']*1000)

    config['doppler_extent'] = config['dopplerCells']/(2 * config['cpi_seconds'])


    return config
