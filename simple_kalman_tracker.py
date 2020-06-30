''' Simple Kalman filter based target tracker for a single passive radar
    target. Mostly intended as a simplified demonstration script. For better
    performance, use multitarget_kalman_tracker.py'''

import numpy as np
import matplotlib.pyplot as plt
import h5py

from passiveRadar.config_params    import getConfigParams
from passiveRadar.target_detection import kalman_filter_dtype
from passiveRadar.target_detection import adaptive_kalman_update
from passiveRadar.target_detection import kalman_extrapolate
from passiveRadar.target_detection import CFAR_2D

# create a data type that represents the internal state of the 
# target tracker
target_track_dtype_simple = np.dtype([
    ('lock_mode',       np.float, (4,)),
    ('measurement',     np.float, (2,)),
    ('measurement_idx', np.int,   (2,)),
    ('estimate',        np.float, (2,)),
    ('range_extent',    np.float),
    ('doppler_extent',  np.float),
    ('kalman_state',    kalman_filter_dtype)])

def simple_track_update(currentState, inputFrame):
    ''' Update step for passive radar target tracker.
    
    Parameters:
        currentState: target_track_dtype_simple_simple containting the current state
        of the target tracker
        inputFrame: range-doppler map from which to acquire a measurement
    Returns:
        newState: target_track_dtype_simple_simple containing the updated target tracker
        state'''

    # Get the current tracker state
    lockMode        = currentState['lock_mode'][0]
    measurement     = currentState['measurement'][0]
    measIdx         = currentState['measurement_idx'][0]
    estimate        = currentState['estimate'][0]
    rangeExtent     = currentState['range_extent'][0]
    dopplerExtent   = currentState['doppler_extent'][0]
    kalmanState     = currentState['kalman_state'][0]

    # Now based on the current tracker state we choose where on the
    # input frame to look for our new measurement. If the tracker is 
    # in an unlocked state we look for the target anywhere. If it is in 
    # one of the target-locked states we restrict attention to a 
    # rectangle around the previous measurement. The size of the
    # rectangle depends on which of the target-locked states we're in.

    lx = measIdx[1]
    ly = measIdx[0]

    # first lock state (initial lock on)
    if (lockMode[1] == 1):
        measurementGate = np.zeros(inputFrame.shape)
        measurementGate[ly-24:ly+24,lx-24:lx+24] = 1.0
        inputFrame = inputFrame * measurementGate

    # second lock state (fully locked)
    elif (lockMode[2] == 1):
        measurementGate = np.zeros(inputFrame.shape)
        measurementGate[ly-16:ly+16,lx-16:lx+16] = 1.0
        inputFrame = inputFrame * measurementGate
        
    # third lock state (losing lock)
    elif (lockMode[3] == 1):
        measurementGate = np.zeros(inputFrame.shape)
        measurementGate[ly-24:ly+24,lx-24:lx+24] = 1.0
        inputFrame = inputFrame * measurementGate

    # else unlocked (don't apply a measurement gate)

    # obtain the new measurement by finding the indices of the 
    # maximum amplitude point of the range-doppler map
    newMeasIdx = np.unravel_index(np.argmax(inputFrame), inputFrame.shape)

    # convert the indices to range-doppler values
    range_meas = rangeExtent*(1 - newMeasIdx[0]/inputFrame.shape[0])
    doppler_meas = dopplerExtent*(2*newMeasIdx[1]/inputFrame.shape[1] - 1)

    # construct the new measurement vector
    newMeasurement = np.array([range_meas, doppler_meas])

    # check if we seem to be locked on to a target. If the new measurement
    # is close to the estimate from the last time step then we assume that
    # a target has been found and we are pleased about it
    surprise_level = newMeasurement - estimate
    badnessMetric = (surprise_level[0]**2 + surprise_level[1]**2)**0.5
    targetFound    = badnessMetric < 12 

    if targetFound:
        # matrix encodes state update rules for target found
        track_update_matrix = np.array([[0,1,0,0], [0,0,1,0], [0,0,1,0],[0,0,1,0]]).T
    else:
        # matrix encodes state update rules for target not found
        track_update_matrix = np.array([[1,0,0,0], [1,0,0,0], [0,0,0,1],[1,0,0,0]]).T
    
    # update the tracker state
    newLockMode = track_update_matrix @ lockMode

    # use the Kalman filter to update the state estimate
    newEstimate, newKalmanState = adaptive_kalman_update(newMeasurement, measurement, kalmanState)

    newState = np.array([(newLockMode, newMeasurement, newMeasIdx, 
        newEstimate, rangeExtent, dopplerExtent, newKalmanState)], dtype = target_track_dtype_simple)

    return newState


def run_target_tracker(data):

    N_measurements = data.shape[2]

    # create initial values for the Kalman Filter
    # initial values for P and Q are taken from P.E. Howland et al, "FM radio 
    # based bistatic radar"

    x = np.array([30, 2, -20, -1])
    P = np.diag([5.0, 0.0225, 0.04, 0.1])
    F1 = np.array([[1,0,-0.003, 0], [0, 0,-0.003,-0.03], [0,0,1,1], [0,0,0,1]])
    F2 = np.array([[1,1,0,0], [0,1,0,0], [0,0,1,1], [0,0,0,1]])
    Q = np.diag([2.0, 0.02, 0.2, 0.05])
    H = np.array([[1,0,0,0], [0,0,1,0]])
    R = np.diag([5, 5])
    S = np.diag([1, 1])

    # initialize the target tracker state
    lockMode      = np.array([1,0,0,0])    # start in unlocked state
    estimate      = H @ x
    measurement   = np.array([35.0, -30.0]) # random IC
    measIdx       = np.array([50,50], dtype = np.int) # random IC
    rangeExtent   = 375 # km
    dopplerExtent = 256/1.092 # Hz
    kalmanState   = np.array([(x, P, F1, F2, Q, H, R, S)], dtype=kalman_filter_dtype)
    
    trackerState = np.array([(lockMode, estimate, measurement,
        measIdx, rangeExtent, dopplerExtent, kalmanState)], dtype = target_track_dtype_simple)

    # preallocate an array to store the tracking results
    history = np.empty((N_measurements,), dtype = target_track_dtype_simple)

    # do the tracking!!111
    for i in range(N_measurements):

        # get a frame of the range-doppler map
        dataFrame = data[:,:,i]

        # normalize it
        dataFrame = dataFrame/np.mean(np.abs(dataFrame).flatten())

        # get the orientation right :))
        dataFrame = np.fliplr(dataFrame.T)

        # zero out the problematic sections (small range / doppler)
        dataFrame[:8, :] = 0
        dataFrame[-8:, :] = 0
        dataFrame[:,250:260] = 0

        # do the update step
        trackerState = simple_track_update(trackerState, dataFrame)
        
        # save the results
        history[i] = trackerState
    
    return history

if __name__ == "__main__":

    config_fname = "PRconfig.yaml"
    config = getConfigParams(config_fname)

    # load the processed passive radar data (range-doppler maps)
    f = h5py.File(config['outputFile'], 'r')
    xambg = np.abs(f['/xambg'])
    Nframes = xambg.shape[2]

    # CFAR filter each frame using a 2D kernel
    CF = np.zeros(xambg.shape)
    for i in range(Nframes):
        CF[:,:,i] = CFAR_2D(xambg[:,:,i], 18, 4)

    history = run_target_tracker(CF)

    estimate = history['estimate']
    measurement = history['measurement']
    lockMode = history['lock_mode']

    unlocked = lockMode[:,0].astype(bool)    
    estimate_locked   = estimate.copy()
    estimate_locked[unlocked, 0] = np.nan
    estimate_locked[unlocked, 1] = np.nan
    estimate_unlocked = estimate.copy()
    estimate_unlocked[~unlocked, 0] = np.nan
    estimate_unlocked[~unlocked, 1] = np.nan

    # range_pred   = np.load('flightradar_range_1011.npy')
    # doppler_pred = np.load('flightradar_doppler_1011.npy')

    plt.figure(figsize=(16,10))
    plt.plot(estimate_locked[:,1], estimate_locked[:,0], 'b', linewidth=3)
    plt.plot(estimate_unlocked[:,1], estimate_unlocked[:,0], c='r', linewidth=1, alpha=0.3)
    plt.xlabel('Doppler Shift (Hz)')
    plt.ylabel('Bistatic Range (km)')
    plt.show()



    


