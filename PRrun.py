import numpy as np
import yaml
import h5py
import dask.array as da
from dask.diagnostics import ProgressBar
from signal_utils import decimate, deinterleave_IQ, frequency_shift
from PRalgo import fast_xambg, find_channel_offset, LS_Filter, offset_compensation

def pad_to_chunks(darray, chunklen):
    padlen = chunklen - np.mod(darray.shape[0], chunklen)
    pad = da.zeros((padlen,), dtype=np.complex64)
    padded = da.concatenate([darray, pad], axis=0)
    return padded

if __name__ == "__main__":

    # GET CONFIG PARAMETERS

    yf = open('PRconfig.yaml', 'r')
    PRconfig = yaml.safe_load(yf)
    yf.close()

    # chunk_length = 5124880  # 512488*2*5*5 (# CPI samples)*(2 floats/complex)*(decimation factor)*(5 CPIs/chunk)
    chunk_length = PRconfig['blockLength']*2*10
    
    # LOADING DATA

    h5file = h5py.File(PRconfig['inputFile'], 'r')
    ref_data = da.from_array(h5file[PRconfig['inputReferencePath']][0:9*chunk_length],  chunks = (chunk_length,))
    srv_data = da.from_array(h5file[PRconfig['inputSurveillancePath']][0:9*chunk_length], chunks = (chunk_length,))
    
    # PADDING TO INTEGER NUMBER OF CHUNK LENGTHS

    ref_dat = pad_to_chunks(ref_data, chunk_length)
    ref_dat = ref_dat.rechunk((chunk_length,))
    srv_dat = pad_to_chunks(srv_data, chunk_length)
    srv_dat = srv_dat.rechunk((chunk_length,))

    # DEINTERLEAVING IQ DATA

    ref_IQ_r = da.map_blocks(deinterleave_IQ, ref_dat, dtype=np.complex64, chunks=(chunk_length//2,))
    srv_IQ_r = da.map_blocks(deinterleave_IQ, srv_dat, dtype=np.complex64, chunks=(chunk_length//2,))

    # TUNING TO CHANNEL AND DECIMATING

    tuneFreq = PRconfig['inputCenterFreq'] - PRconfig['channelFreq']
    Fs0 = PRconfig['inputSampleRate']

    r2 = da.map_blocks(frequency_shift, ref_IQ_r, tuneFreq, Fs0, dtype=np.complex64, chunks=(chunk_length//2,))
    r3 = da.map_blocks(decimate, r2, 10, dtype=np.complex64, chunks=(chunk_length//20,))
    
    s2 = da.map_blocks(frequency_shift, srv_IQ_r,  tuneFreq, Fs0, dtype=np.complex64, chunks=(chunk_length//2,))
    s3 = da.map_blocks(decimate, s2, 10, dtype=np.complex64, chunks=(chunk_length//20,))

    # CHANNEL OFFSET COMPENSATION

    s3o = da.map_blocks(offset_compensation, r3, s3, 32, dtype=np.complex64, chunks = (chunk_length//20,))
    s3o2 = da.map_blocks(offset_compensation, r3, s3o, 3, dtype=np.complex64, chunks = (chunk_length//20,))

    # APPLY LS FILTER

    arg1 = (r3, s3o2, 120, 1)
    SRV_CLEANED_1 = da.map_blocks(LS_Filter, *arg1, dtype=np.complex64, chunks = (chunk_length//20,))

    rs05p = da.map_blocks(frequency_shift, r3,  1, Fs0//10, dtype=np.complex64, chunks=(chunk_length//20,))
    rs05n = da.map_blocks(frequency_shift, r3, -1, Fs0//10, dtype=np.complex64, chunks=(chunk_length//20,))
    rs1p = da.map_blocks(frequency_shift, r3,  2, Fs0//10, dtype=np.complex64, chunks=(chunk_length//20,))
    rs1n = da.map_blocks(frequency_shift, r3, -2, Fs0//10, dtype=np.complex64, chunks=(chunk_length//20,))

    arg1 = (rs05p, SRV_CLEANED_1, 120, 1)
    SRV_CLEANED_2 = da.map_blocks(LS_Filter, *arg1, dtype=np.complex64, chunks = (chunk_length//20,))

    arg1 = (rs05n, SRV_CLEANED_2, 120, 1)
    SRV_CLEANED_3 = da.map_blocks(LS_Filter, *arg1, dtype=np.complex64, chunks = (chunk_length//20,))

    arg1 = (rs1p, SRV_CLEANED_3, 120, 1)
    SRV_CLEANED_4 = da.map_blocks(LS_Filter, *arg1, dtype=np.complex64, chunks = (chunk_length//20,))

    arg1 = (rs1n, SRV_CLEANED_4, 120, 1)
    SRV_CLEANED_5 = da.map_blocks(LS_Filter, *arg1, dtype=np.complex64, chunks = (chunk_length//20,))

    # # COMPUTE CROSS-AMBIGUITY FUNCTION

    xarg = (r3, SRV_CLEANED_5, 200, 512)
    XAMBG = da.map_blocks(fast_xambg, *xarg, dtype=np.complex64, chunks=(512,201,1))

    f = h5py.File(PRconfig['outputFile'])
    d = f.require_dataset('/xambg', shape=XAMBG.shape, dtype=XAMBG.dtype)

    with ProgressBar():
        da.store(XAMBG, d)
    f.close()
