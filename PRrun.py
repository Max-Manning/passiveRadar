import numpy as np
import h5py
import dask.array as da
from dask.diagnostics import ProgressBar
from PRutils import decimate, deinterleave_IQ, frequency_shift, shift
from PRalgo import fast_xambg, find_channel_offset, LS_Filter

def pad_to_chunks(darray, chunklen):
    padlen = chunklen - np.mod(darray.shape[0], chunklen)
    pad = da.zeros((padlen,), dtype=np.complex64)
    padded = da.concatenate([darray, pad], axis=0)
    return padded

def offset_compensation(x1, x2, ndec):
    s1 = x1[0:1000000]
    s2 = x2[0:1000000]
    os = find_channel_offset(s1, s2, ndec, 6*ndec)
    if(os == 0):
        return x2
    else:
        return shift(x2, os)

def getXambg(s1, s2, nlag, nfft, nblocks):
    cl = s1.shape[0]//nblocks
    XX = np.zeros((nfft, nlag+1, nblocks), dtype=np.complex64)
    for k in range(nblocks):
        XX[:,:,k] = fast_xambg(s1[k*cl:(k+1)*cl], s2[k*cl:(k+1)*cl], nlag, nfft)
    return XX

def Apply_LS_Filter(s1, s2, nl, reg, nblocks=5):
    '''Apply a LS filter to a dask chunk'''
    Bl = 524288 // 2
    y = np.zeros(s2.shape, dtype=np.complex64)

    for j in range(nblocks):
        yb = LS_Filter(s1[j*Bl:(j+1)*Bl], s2[j*Bl:(j+1)*Bl], nl, reg)
        y[j*Bl:(j+1)*Bl] = yb

    return y

if __name__ == "__main__":

    chunk_length = 25624400  # 512488*2*5*5 (# CPI samples)*(2 floats/complex)*(decimation factor)*(5 CPIs/chunk)

    # LOADING DATA

    h5file = h5py.File('PassiveRadar_20191102_1011.hdf5', 'r')
    ref_data = da.from_array(h5file['/data/ref'],  chunks = (chunk_length,))
    srv_data = da.from_array(h5file['/data/srv'], chunks = (chunk_length,))
    
    # PADDING TO INTEGER NUMBER OF CHUNK LENGTHS

    ref_dat = pad_to_chunks(ref_data, chunk_length)
    ref_dat = ref_dat.rechunk((chunk_length,))
    srv_dat = pad_to_chunks(srv_data, chunk_length)
    srv_dat = srv_dat.rechunk((chunk_length,))

    # DEINTERLEAVING IQ DATA

    ref_IQ_r = da.map_blocks(deinterleave_IQ, ref_dat, dtype=np.complex64, chunks=(chunk_length//2,))
    srv_IQ_r = da.map_blocks(deinterleave_IQ, srv_dat, dtype=np.complex64, chunks=(chunk_length//2,))

    # TUNING TO CHANNEL AND DECIMATING

    r2 = da.map_blocks(frequency_shift, ref_IQ_r, -7.e5, 2.4e6, dtype=np.complex64, chunks=(chunk_length//2,))
    r3 = da.map_blocks(decimate, r2, 10, dtype=np.complex64, chunks=(chunk_length//20,))
    
    s2 = da.map_blocks(frequency_shift, srv_IQ_r, -7.e5, 2.4e6, dtype=np.complex64, chunks=(chunk_length//2,))
    s3 = da.map_blocks(decimate, s2, 10, dtype=np.complex64, chunks=(chunk_length//20,))

    # CHANNEL OFFSET COMPENSATION

    s3o = da.map_blocks(offset_compensation, r3, s3, 32, dtype=np.complex64, chunks = (chunk_length//20,))
    s3o2 = da.map_blocks(offset_compensation, r3, s3o, 3, dtype=np.complex64, chunks = (chunk_length//20,))

    # APPLY LS FILTER

    arg1 = (r3, s3o2, 120, 1)
    SRV_CLEANED_1 = da.map_blocks(Apply_LS_Filter, *arg1, dtype=np.complex64, chunks = (chunk_length//20,))

    rs05p = da.map_blocks(frequency_shift, r3,  1, 2.4e5, dtype=np.complex64, chunks=(chunk_length//20,))
    rs05n = da.map_blocks(frequency_shift, r3, -1, 2.4e5, dtype=np.complex64, chunks=(chunk_length//20,))
    rs1p = da.map_blocks(frequency_shift, r3,  2, 2.4e5, dtype=np.complex64, chunks=(chunk_length//20,))
    rs1n = da.map_blocks(frequency_shift, r3, -2, 2.4e5, dtype=np.complex64, chunks=(chunk_length//20,))

    arg1 = (rs05p, SRV_CLEANED_1, 120, 1)
    SRV_CLEANED_2 = da.map_blocks(Apply_LS_Filter, *arg1, dtype=np.complex64, chunks = (chunk_length//20,))

    arg1 = (rs05n, SRV_CLEANED_2, 120, 1)
    SRV_CLEANED_3 = da.map_blocks(Apply_LS_Filter, *arg1, dtype=np.complex64, chunks = (chunk_length//20,))

    arg1 = (rs1p, SRV_CLEANED_3, 120, 1)
    SRV_CLEANED_4 = da.map_blocks(Apply_LS_Filter, *arg1, dtype=np.complex64, chunks = (chunk_length//20,))

    arg1 = (rs1n, SRV_CLEANED_4, 120, 1)
    SRV_CLEANED_5 = da.map_blocks(Apply_LS_Filter, *arg1, dtype=np.complex64, chunks = (chunk_length//20,))



    # COMPUTE CROSS-AMBIGUITY FUNCTION

    xarg = (r3, SRV_CLEANED_5, 200, 512, 5)
    XAMBG = da.map_blocks(getXambg, *xarg, dtype=np.complex64, chunks=(512,201,5))

    f = h5py.File('XAMBG_1011_1027M_250km_256Hz_filt120_012.hdf5')
    d = f.require_dataset('/xambg', shape=XAMBG.shape, dtype=XAMBG.dtype)

    with ProgressBar():
        da.store(XAMBG, d)
        # A = XAMBG.compute()

    f.close()
    
    # np.save("PR_20191025_1027_200c.npy", A)

    # f.close()
