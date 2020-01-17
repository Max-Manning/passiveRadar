import numpy as np
import yaml
import h5py
import dask.array as da
from dask.diagnostics import ProgressBar
from passiveRadar.signal_utils import decimate, deinterleave_IQ, frequency_shift
from passiveRadar.PRalgo import fast_xambg, LS_Filter, offset_compensation

def pad_to_chunks(darray, chunklen):
    padlen = chunklen - np.mod(darray.shape[0], chunklen)
    pad = da.zeros((padlen,), dtype=np.complex64)
    padded = da.concatenate([darray, pad], axis=0)
    return padded

def clutter_removal(s1, s2, nlag, Fs, dbins = [0]):
    '''clutter removal with least squares filter

    Parameters:
    s1: reference signal
    s2: surveillance signal
    nlag: length of least squares filter
    Fs: input signal sample frequency
    dbins: list of doppler bins to do filtering on (default 0)
    
    returns: y (surveillance signal with static echoes removed)'''
    y = s2
    for ds in dbins:
        if ds == 0:
            y = LS_Filter(s1, y, nlag, 1)
        else:
            s1s = frequency_shift(s1, ds, Fs)
            y = LS_Filter(s1s, y, nlag, 1)
    return y

def channel_preprocessing(sig, dec, fc, Fs):
    '''deinterleave IQ samples, tune to channel frequency and decimate'''
    IQ = deinterleave_IQ(sig)
    IQ_tuned = frequency_shift(IQ, fc, Fs)
    IQd = decimate(IQ_tuned, dec)
    return IQd

if __name__ == "__main__":

    # GET CONFIG PARAMETERS
    yf = open('PRconfig.yaml', 'r')
    PRconfig = yaml.safe_load(yf)
    yf.close()

    # chunk_length = 5124880  # 512488*2*5*5 (# CPI samples)*(2 floats/complex)*(decimation factor)*(5 CPIs/chunk)
    chunk_length = PRconfig['blockLength']*2*10
    
    # LOADING DATA
    h5file = h5py.File(PRconfig['inputFile'], 'r')
    ref_dat = da.from_array(h5file[PRconfig['inputReferencePath']][0:50*chunk_length],  chunks = (chunk_length,))
    srv_dat = da.from_array(h5file[PRconfig['inputSurveillancePath']][0:50*chunk_length], chunks = (chunk_length,))
    
    # PADDING TO INTEGER NUMBER OF CHUNK LENGTHS

    # ref_dat = pad_to_chunks(ref_data, chunk_length)
    # ref_dat = ref_dat.rechunk((chunk_length,))
    # srv_dat = pad_to_chunks(srv_data, chunk_length)
    # srv_dat = srv_dat.rechunk((chunk_length,))


    # DEINTERLEAVING DATA, TUNING TO CHANNEL AND DECIMATING

    tuneFreq = PRconfig['inputCenterFreq'] - PRconfig['channelFreq']
    Fs0 = PRconfig['inputSampleRate']

    r2 = da.map_blocks(channel_preprocessing, ref_dat, 10, tuneFreq, Fs0,dtype=np.complex64, chunks=(chunk_length//20,))
    s2 = da.map_blocks(channel_preprocessing, srv_dat, 10, tuneFreq, Fs0,dtype=np.complex64, chunks=(chunk_length//20,))

    # CHANNEL OFFSET COMPENSATION

    s2o = da.map_blocks(offset_compensation, r2, s2, 32, dtype=np.complex64, chunks = (chunk_length//20,))
    s2o2 = da.map_blocks(offset_compensation, r2, s2o, 3, dtype=np.complex64, chunks = (chunk_length//20,))

    # APPLY LS FILTER

    arg1 = (r2, s2o2, 100, Fs0//10, [0,1,-1,2,-2])
    SRV_CLEANED = da.map_blocks(clutter_removal, *arg1, dtype=np.complex64, chunks = (chunk_length//20,))

    # # COMPUTE CROSS-AMBIGUITY FUNCTION

    xarg = (r2, SRV_CLEANED, 300, 512)
    XAMBG = da.map_blocks(fast_xambg, *xarg, dtype=np.complex64, chunks=(512,301,1))

    f = h5py.File(PRconfig['outputFile'])
    d = f.require_dataset('/xambg', shape=XAMBG.shape, dtype=XAMBG.dtype)

    with ProgressBar():
        da.store(XAMBG, d)
    f.close()