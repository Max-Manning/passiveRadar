''' Kalman filter based target tracker that can handle multiple targets. Mostly
    based on material from "Multitarget-Multisensor Tracking: Principles and 
    Techniques" by Yaakov Bar-Shalom and Xiao-Rong Li'''
    
import numpy as np
import matplotlib.pyplot as plt
import h5py

from passiveRadar.config_params    import getConfigParams
from passiveRadar.target_detection import kalman_filter_dtype
from passiveRadar.target_detection import adaptive_kalman_update
from passiveRadar.target_detection import kalman_extrapolate
from passiveRadar.target_detection import CFAR_2D

# A datatype that represents a single target track 
target_track_dtype = np.dtype([
    ('status', np.int),     # the status of the track is 0, 1 or 2.
                            # 0: free (no target assigned to this track)
                            # 1: tracking preliminary target 
                            # 2: tracking confirmed target
    ('lifetime', np.int),   # keeps track of how long the track has been alive
    ('measurement', np.float, (2,)), # measurement for the current timestep
    ('estimate', np.float, (2,)),    # state estimate for the current timestep
    ('measurement_history', np.float, (20,)), # over the past 20 timesteps, keep
    # track of where there were confirmed measurements assigned to this track.
    # This is used to determine when a track has lost its target
    ('kalman_state', kalman_filter_dtype)]) # the state of the Kalman filter

def get_measurements(dataFrame, p, frame_extent):
    ''' extract a list of candidate measurements from a range-doppler map frame

    Parameters:

        dataFrame:      2D array containting the range-doppler map to acquire
                        measurements from.
        p:              Detection threshold. Pixels whose amplitudes are in the 
                        upper pth percentile of the values in dataFrame are 
                        added to the list of candidate measurements.
        frame_extent:   Defines the edges lengths for  the input frame allowing
                        pixel indices to be converted to measurement values.
    Returns:

        candidateRange:     Vector containing all the range values for the M 
                            candidate measurements
        candidateDoppler:   Vector containing all the Doppler values for the M 
                            candidate measurements
        candidateStrength:  Vector containing all the pixel amplitudes values 
                            for the M candidate measurements
    '''

    # define the extent of the measurement region
    rangeExtent   = frame_extent[1]   # km
    dopplerExtent = frame_extent[0]   # Hz
    rpts = np.linspace(rangeExtent, 0, dataFrame.shape[1])
    dpts = np.linspace(-1*dopplerExtent, dopplerExtent, dataFrame.shape[0])
    rangeBinCenters   = np.expand_dims(rpts, 1)
    dopplerBinCenters = np.expand_dims(dpts, 0)

    rangeBinCenters   = np.tile(rangeBinCenters,    (1, dataFrame.shape[0]))
    dopplerBinCenters = np.tile(dopplerBinCenters,  (dataFrame.shape[1], 1))

    # normalize input frame
    dataFrame = dataFrame/np.mean(np.abs(dataFrame).flatten())
    # get the orientation right :))
    dataFrame = np.fliplr(dataFrame.T)
    # zero out the problematic sections where there is persistent
    # clutter (small range / doppler)
    dataFrame[:8, :] = 0
    dataFrame[-8:, :] = 0
    dataFrame[:,251:259] = 0

    # calculate the detection threshold. There are 300x512 = 153600 pixels per 
    # range doppler frame, so taking 99.8th percentile selects the strongest 300
    # or so to be used as potential measurements.
    threshold = np.percentile(dataFrame, 99.8)

    # find points on the input frame where amplitude exceeds threshold
    candidateMeasIdx    = np.nonzero(dataFrame >= threshold)

    # extract candidate measurement positions and measurement strength
    candidateRange      = rangeBinCenters[candidateMeasIdx]
    candidateDoppler    = dopplerBinCenters[candidateMeasIdx]
    candidateStrength   = dataFrame[candidateMeasIdx]

    # sort the candidate measurements in decreasing order of strength
    sort_idx = np.flip(np.argsort(candidateStrength))
    candidateRange      = candidateRange[sort_idx]
    candidateDoppler    = candidateDoppler[sort_idx]
    candidateStrength   = candidateStrength[sort_idx]

    candidateMeas = np.stack((candidateRange, candidateDoppler, candidateStrength))

    return candidateMeas

def associate_measurements(trackState, candidateMeas):
    '''Associate a set of candidate measurements with a target track
    
    Parameters:
        trackState:     target_track_dtype containing the current track state
        candidateMeas:  list of candidate measurements
    Returns:
        newMeasurement: the measurement selected to update this target track. 
                        Returns None if there are no suitable measurements. 
        candidateMeas:  The updated list of candidate measurements. If a 
                        measurement has been selected for this target then 
                        measurements close to the selected measurement are 
                        removed.
    '''
    
    track_status = trackState['status']

    # get the last measurement and state estimate for this track
    lastRangeMeas    = trackState['measurement'][0]
    lastDopplerMeas  = trackState['measurement'][1]
    lastDopplerEst  = trackState['estimate'][1]
    lastRangeEst    = trackState['estimate'][0]

    candidateRange      = candidateMeas[0, :]
    candidateDoppler    = candidateMeas[1, :]
    candidateStrength   = candidateMeas[2, :]

    ########### FIRST VALIDATION STEP #########################################

    if track_status == 0:
        # if the track state is free we're not picky, we consider any measurement
        earlyGate = np.ones(candidateRange.shape).astype(bool)

    elif track_status == 1:
        # if the track state is preliminary we look within 5km and 12Hz of the 
        # last confirmed measurement
        rangeGate   = (np.abs(candidateRange   - lastRangeMeas)   < 5)
        dopplerGate = (np.abs(candidateDoppler - lastDopplerMeas) < 12)
        earlyGate =  rangeGate.astype(bool) & dopplerGate.astype(bool)

    else:
        # if the track state is confirmed we look within 4km and 10Hz of the 
        # last state estimate
        rangeGate = (np.abs(candidateRange - lastRangeEst) < 4)
        dopplerGate = (np.abs(candidateDoppler - lastDopplerEst) < 10)
        earlyGate =  rangeGate.astype(bool) & dopplerGate.astype(bool)
    
    rangeMeas       = candidateRange[earlyGate]
    dopplerMeas     = candidateDoppler[earlyGate]
    strengthMeas    = candidateStrength[earlyGate]

    # SECOND VALIDATION STEP (ONLY FOR CONFIRMED TRACKS)########################

    if track_status == 2: # confirmed tracks
        # apply a stricter validation gate based on the innovation
        # covariance matrix of the kalman filter for this track
        NCloseCandidates = np.sum(earlyGate)
        validationGate = np.zeros((NCloseCandidates,)).astype(bool)
        S = trackState['kalman_state']['S']
        for kk in range(NCloseCandidates):
            rangeDiff = lastRangeMeas - rangeMeas[kk]
            dopplerDiff = lastDopplerMeas - dopplerMeas[kk]
            zerr = np.array([rangeDiff, dopplerDiff])
            validationGate[kk] = zerr.T @ np.linalg.inv(S) @ zerr < 6
            
        # Get the measurements that fit the final validation criteria for 
        # this track
        rangeMeas       = rangeMeas[validationGate]
        dopplerMeas     = dopplerMeas[validationGate]
        strengthMeas    = strengthMeas[validationGate]

    ############################################################################
    
    # how many measurements did we get?
    measurementsFound = rangeMeas.size

    if measurementsFound == 0:
        # if there are no suitable measurements for this track we return None
        # and leave the list of candidate measurements unchanged
        return None, candidateMeas

    elif measurementsFound > 1:
        # if there are multiple measurements that match with this target we need
        # to pick which one to use
        if track_status == 0:
            # take the strongest measurement if the target is unconfirmed
            rangeMeas       = candidateRange[0]
            dopplerMeas     = candidateDoppler[0]
            strengthMeas    = candidateStrength[0] 
            newMeasurement = np.squeeze(np.array([rangeMeas, dopplerMeas]))   
        
            rangeGate   = (np.abs(candidateRange - rangeMeas) < 10).astype(bool)
            dopplerGate = (np.abs(candidateDoppler - dopplerMeas) < 12).astype(bool)
            earlyGate   =  rangeGate & dopplerGate
        elif track_status==1:
            # if there are multiple candidate measurements pick the nearest one
            # (Nearest Neighbour Standard Filter). I should definitely screw
            # around with various other methods here eg PDAF
            ixm = np.argmin(np.sqrt(rangeMeas**2 + dopplerMeas**2))
            rangeMeas       = rangeMeas[ixm]
            dopplerMeas     = dopplerMeas[ixm]
            strengthMeas    = strengthMeas[ixm]
        if track_status == 2:
            # # if there are multiple candidate measurements pick the strongest one
            # # (Strongest Neighbour Standard Filter). 
            rangeMeas       = rangeMeas[0]
            dopplerMeas     = dopplerMeas[0]
            strengthMeas    = strengthMeas[0]
            newMeasurement = np.squeeze(np.array([rangeMeas, dopplerMeas]))

    candidateRange      = candidateRange[~earlyGate]
    candidateDoppler    = candidateDoppler[~earlyGate]
    candidateStrength   = candidateStrength[~earlyGate]

    newMeasurement = np.squeeze(np.array([rangeMeas, dopplerMeas])) 
    candidateMeas = np.stack((candidateRange, candidateDoppler, candidateStrength))

    return newMeasurement, candidateMeas

def initialize_track(measurement):
    '''Create a new target track with default parameter values

    Parameters:
        measurement:            first measurement for this target track
            if measurement is None, the track is initialized in the 'free' state
            (status = 0) at an arbitrary set of coordinates
            if measurement is a set of coordinates, the track is initialized at 
            these coordinates in the 'preliminary' state (status = 1)
    Returns:    
        initialTrackerState:    target_track_dtype containing the initialized
                                target track
    '''
    if measurement is None:
        r = 0 # initialize position to 0 (this is arbitrary)
        f = 0
        status=0
    else:
        r = measurement[0]
        f = measurement[1]
        status = 1

    # create initial Kalman filter parameters
    x = np.array([r, 0, f, -1])
    P = np.diag([5.0, 0.0225, 0.04, 0.1])
    F1 = np.array([[1,0,-0.003, 0], [0, 0,-0.003,-0.003], [0,0,1,1], [0,0,0,1]])
    F2 = np.array([[1,1,0,0], [0,1,0,0], [0,0,1,1], [0,0,0,1]])
    Q = np.diag([4.0, 0.03, 0.2, 0.08])
    H = np.array([[1,0,0,0], [0,0,1,0]])
    R = np.diag([5, 2])
    S = np.diag([1, 1])

    # initialize the target tracker state
    lifetime            = 1
    estimate            = H @ x
    measurement         = np.array([r, f])
    targetHistory       = np.zeros((20,))
    targetHistory[0]    = 1
    targetHistory[5:10] = 1
    kalmanState         = (x, P, F1, F2, Q, H, R, S)
    
    initialTrackState = np.array([(status, lifetime, estimate, measurement, 
        targetHistory, kalmanState)], dtype=target_track_dtype)

    return initialTrackState

def update_track(currentState, newMeasurement):
    ''' Update a single target track with a new measurement.
    
    Parameters:
        currentState:   target_track_dtype containting the current state of the 
                        target track
        newMeasurement: measurement vector for this timestep
    Returns:
        newState:       target_track_dtype containing the updated track state
    '''

    status = currentState['status']
    lifetime = np.squeeze(currentState['lifetime'])
    measurement = currentState['measurement']
    kalmanState = currentState['kalman_state']
    targetFoundHistory = currentState['measurement_history'].flatten()

    if newMeasurement is None:
        # no suitable measurements have been assigned to this track.
        # extrapolate the next state from the current state estimate.
        newEstimate, newKalmanState = kalman_extrapolate(kalmanState)
        newTargetFoundHistory = np.concatenate(([0], targetFoundHistory[:-1]))
        newMeasurement = measurement # just keep the last confirmed measurement

    else:

        # use the Kalman filter to update the state estimate
        newEstimate, newKalmanState = adaptive_kalman_update(newMeasurement, 
                                        measurement, kalmanState)
        newTargetFoundHistory = np.concatenate(([1], targetFoundHistory[:-1]))

    
    # if the track status is currently 1 (preliminary), we can decide to either
    # promote it to a confirmed target track or kill it depending on how many
    # measurements consistent with the estimated target state have been obtained 
    # over the past few timesteps
    if status == 1:
        # condition to kill track
        if (lifetime > 4 ) and (np.sum(targetFoundHistory[0:10]) < 5):
            status = 0
        # condition to promote to confirmed target track
        if (lifetime > 4 ) and (np.sum(targetFoundHistory[0:10]) > 8):
            status = 2

    # If the track status is confirmed we can kill it if not enough measurements
    # have been found.
    elif status == 2:
        # condition to kill track
        if (lifetime > 4) and (np.sum(targetFoundHistory) < 4):
            status = 0

    # construct the new track state
    newState = np.array([(status, lifetime+1, newMeasurement, newEstimate, 
        newTargetFoundHistory, newKalmanState)], dtype=target_track_dtype)

    return newState

def multitarget_tracker(data, frame_extent, N_TRACKS):
    ''' Radar target tracker for multiple targets. 
    
    Parameters:
        data:           3D array containting a stack of range-doppler map frames

        frame_extent:   Defines the edges lengths for  the input frame allowing
                        pixel indices to be converted to measurement values.

        N_TRACKS:       Number of tracks. Corresponds to the maximum number of
                        targets that can be tracked simultaneously
    Returns:
        history:        (Nframes, N_TRACKS) array of target_track_dtype.
                        Contains the complete state of each of the target tracks
                        at each time step
    '''

    # number of data frames
    Nframes = data.shape[2]

    # initialize a vector of tatget tracks
    trackStates  = np.empty((N_TRACKS,), dtype=target_track_dtype)
    for i in range(trackStates.shape[0]):
        # start each track in the unlocked state
        trackStates[i] =  initialize_track(None)

    # make a storage array for the results at each timestep
    tracker_history = np.empty((Nframes, N_TRACKS), 
        dtype = target_track_dtype)

    # loop over input frames
    for i in range(Nframes):

        # get the range-doppler frame for this timestep
        dataFrame = data[:,:,i]

        # get the new list of candidate measurements for this frame
        candidateMeas = get_measurements(dataFrame, 99.8, frame_extent)

        # find out which tracks are in the confirmed, preliminary and free states
        confirmedTracks = np.argwhere(trackStates['status'] == 2).flatten()
        prelimTracks    = np.argwhere(trackStates['status'] == 1).flatten()
        freeTracks      = np.argwhere(trackStates['status'] == 0).flatten()

        for track_idx in confirmedTracks:
            # loop over the confirmed tracks first (confirmed tracks get first
            # access to the candidate measurements.)
            trackState = trackStates[track_idx]
            newMeasurement, candidateMeas = associate_measurements(trackState, candidateMeas)
            newTrackState = update_track(trackState, newMeasurement)
            trackStates[track_idx] = newTrackState

        for track_idx in prelimTracks:
            # Next update the preliminary tracks
            trackState = trackStates[track_idx]
            newMeasurement, candidateMeas = associate_measurements(trackState, candidateMeas)
            newTrackState = update_track(trackState, newMeasurement)
            trackStates[track_idx] = newTrackState

        for track_idx in freeTracks:
            # Finally, assign free tracks to remaining measurements
            trackState = trackStates[track_idx]
            if candidateMeas.size == 0:
                print("no more measurements available, track remaining free")
                break
            newMeasurement, candidateMeas = associate_measurements(trackState, candidateMeas)
            newTrackState = initialize_track(newMeasurement)
            trackStates[track_idx] = newTrackState

        # save all the track states for this timestep
        tracker_history[i,:] = trackStates

    return tracker_history

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

    # run the target tracker
    N_TRACKS = 20 
    tracker_history = multitarget_tracker(CF, 
        [config['doppler_extent'], config['range_extent']], 
        N_TRACKS)
    
    # find the indices of the tracks where there are confirmed targets
    tracker_status = tracker_history['status']
    tracker_status_confirmed_idx = np.nonzero(tracker_status == 2)

    #get the range and doppler values for each target track
    tracker_range = np.squeeze(tracker_history['estimate'][:,:,0]).copy()
    tracker_doppler = np.squeeze(tracker_history['estimate'][:,:,1]).copy()

    # if the target is uncorfirmed change the range/doppler values to nan
    tracker_range[~tracker_status_confirmed_idx] = np.nan
    tracker_doppler[~tracker_status_confirmed_idx] = np.nan

    # plot the tracks
    plt.figure(figsize = (16, 10))
    plt.scatter(tracker_doppler, tracker_range, marker='.')
    plt.xlabel("Doppler Shift (Hz)")
    plt.ylabel("Bistatic Range (km)")
    plt.show()
