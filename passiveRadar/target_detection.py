''' target_detection.py: target detection tools for passive radar '''

import numpy as np
from passiveRadar.signal_utils import normalize
import scipy.signal as signal


# create a data type to represent the Kalman filter's internals
kalman_filter_dtype = np.dtype([
    ('x' , np.float, (4, )),   # state estimate vector
    ('P' , np.float, (4,4)),   # state estimate covariance matrix
    ('F1', np.float, (4,4)),   # state transition model #1
    ('F2', np.float, (4,4)),   # state transition model #2
    ('Q' , np.float, (4,4)),   # process noise covariance matrix
    ('H' , np.float, (2,4)),   # measurement matrix
    ('R' , np.float, (2,2)),   # measurement noise covariance matrix
    ('S' , np.float, (2,2))])  # innovation covariance matrix

def kalman_update(measurement, currentState):
    ''' The standard Kalman filter update algorithm
    
    Parameters: 
        measurement:     measurement vector for current time step
        currentState:    kalman_filter_dtype containing the current filter state
    Returns:
        estimate:        the new estimate of the system state
        newState:        the new filter state 
    '''

    x  = currentState['x']  # state estimate vector
    P  = currentState['P']  # state estimate covariance matrix
    F1 = currentState['F1'] # state transition model #1
    F2 = currentState['F2'] # state transition model #2
    Q  = currentState['Q']  # process noise covariance matrix
    H  = currentState['H']  # measurement matrix
    R  = currentState['R']  # measurement noise covariance matrix
    S  = currentState['S']  # innovation covariance matrix

    # update state according to state transition model #1
    x = F1 @ x
    # update state covariance according to state transition model #2
    P = F2 @ P @ F2.T + Q
    # compute the innovation covariance
    S = H @ P @ H.T +  R
    # compute the optimal Kalman gain
    K = P @ H.T @ np.linalg.inv(S)
    # get the measurement
    Z = measurement
    # compute difference between prediction and measurement
    y = Z - H @ x
    # update the filter state
    x = x + K @ y
    # update the state covariance
    P = (np.eye(4) - K @ H) @ P

    # save the a posteriori state estimate
    estimate  = H @ x
    # construct the new filter state
    newState = (x, P, F1, F2, Q, H, R, S)

    return estimate, newState

def adaptive_kalman_update(measurement, lastMeasurement, currentState):
    ''' The standard Kalman filter update algorithm with adaptive estimation
    of the measurement covariance matrix.
    
    Parameters: 
        measurement:     measurement vector for current time step
        lastMeasurement: measurement vector for previous time step
        currentState:    kalman_filter_dtype containing the current filter state
    Returns:
        estimate:        the new estimate of the system state
        newState:        the new filter state 
    '''

    x  = currentState['x']  # state estimate vector
    P  = currentState['P']  # state estimate covariance matrix
    F1 = currentState['F1'] # state transition model #1
    F2 = currentState['F2'] # state transition model #1
    Q  = currentState['Q']  # process noise covariance matrix
    H  = currentState['H']  # measurement matrix
    R  = currentState['R']  # measurement noise covariance matrix
    S  = currentState['S']  # innovation covariance matrix

    # Adaptive estimation of  the measurement covariance matrix. Here I am 
    # using the squared distance between the current measurement and the 
    # previous measurement. This is very ad hoc but,,,, it seems to work well
    delta_meas = np.squeeze(measurement - lastMeasurement)
    R_scaling_factor = (delta_meas[0]**2 + delta_meas[1]**2)
    # R_estimate = delta_meas.T @ np.linalg.inv(S) @ delta_meas

    # update state according to state transition model #1
    x = F1 @ x
    # update state covariance according to state transition model #2
    P = F2 @ P @ F2.T + Q
    # compute the innovation covariance
    S = H @ P @ H.T +  R*R_scaling_factor
    # compute the optimal Kalman gain
    K = P @ H.T @ np.linalg.inv(S)
    # get the measurement
    Z = measurement
    # compute difference between prediction and measurement
    y = Z - H @ x
    # update the filter state
    x = x + K @ y
    # update the state covariance
    P = (np.eye(4) - K @ H) @ P

    # save the a posteriori state estimate
    estimate  = H @ x
    # construct the new filter state
    newState = (x, P, F1, F2, Q, H, R, S)

    return estimate, newState

def kalman_extrapolate(currentState):
    ''' Update Kalman filter state according to internal model (use when no 
    measurements are available)
    
    Parameters: 
        currentState:  kalman_filter_dtype containing the filter state at time t
    Returns:
        estimate:      the estimate of the system state for time t+1
        newState:      the filter state for time t+1 
    '''

    x  = currentState['x']  # state estimate vector
    P  = currentState['P']  # state estimate covariance matrix
    F1 = currentState['F1'] # state transition model #1
    F2 = currentState['F2'] # state transition model #1
    Q  = currentState['Q']  # process noise covariance matrix
    H  = currentState['H']  # measurement matrix
    R  = currentState['R']  # measurement noise covariance matrix
    S  = currentState['S']  # innovation covariance matrix

    # update state according to state transition model #1
    x = F1 @ x
    # update state covariance according to state transition model #2
    P = F2 @ P @ F2.T + Q
    # compute the innovation covariance
    S = H @ P @ H.T +  R

    # save the a posteriori state estimate
    estimate  = H @ x
    # construct the new filter state
    newState = (x, P, F1, F2, Q, H, R, S)

    return estimate, newState

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
    dopplerCenter = dataFrame.shape[1]//2
    dataFrame[:, dopplerCenter-4:dopplerCenter+4] = 0

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

    ####################### FIRST VALIDATION STEP ##############################

    if track_status == 0:
        # if the track state is free we're not picky, we consider any measurement
        earlyGate = np.ones(candidateRange.shape).astype(bool)

    elif track_status == 1:
        # if the track state is preliminary we look within 5km and 12Hz of the 
        # last confirmed measurement
        rangeGate   = (np.abs(candidateRange   - lastRangeMeas)   < 5)
        dopplerGate = (np.abs(candidateDoppler - lastDopplerMeas) < 24)
        earlyGate =  rangeGate.astype(bool) & dopplerGate.astype(bool)

    else:
        # if the track state is confirmed we look within 4km and 10Hz of the 
        # last state estimate
        rangeGate = (np.abs(candidateRange - lastRangeEst) < 4)
        dopplerGate = (np.abs(candidateDoppler - lastDopplerEst) < 20)
        earlyGate =  rangeGate.astype(bool) & dopplerGate.astype(bool)
    
    rangeMeas       = candidateRange[earlyGate]
    dopplerMeas     = candidateDoppler[earlyGate]
    strengthMeas    = candidateStrength[earlyGate]

    ############ SECOND VALIDATION STEP (ONLY FOR CONFIRMED TRACKS) ############

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
    # the algorithm is pretty robust to changes in these values, the numbers
    # given here seem to work well
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
        if (lifetime > 4 ) and (np.sum(targetFoundHistory[0:10]) < 6):
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
    tracker_history = np.empty((Nframes, N_TRACKS), dtype = target_track_dtype)

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
        measurementGate[ly-24:ly+24,lx-48:lx+48] = 1.0
        inputFrame = inputFrame * measurementGate

    # second lock state (fully locked)
    elif (lockMode[2] == 1):
        measurementGate = np.zeros(inputFrame.shape)
        measurementGate[ly-16:ly+16,lx-32:lx+32] = 1.0
        inputFrame = inputFrame * measurementGate
        
    # third lock state (losing lock)
    elif (lockMode[3] == 1):
        measurementGate = np.zeros(inputFrame.shape)
        measurementGate[ly-24:ly+24,lx-48:lx+48] = 1.0
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
    badnessMetric = (surprise_level[0]**2 + (0.5*surprise_level[1])**2)**0.5
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


def simple_target_tracker(data, rangeExtent, dopplerExtent):

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
    # rangeExtent   = 375 # km
    # dopplerExtent = 256/1.092 # Hz
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

def CFAR_2D(X, fw, gw, thresh = None):
    '''constant false alarm rate target detection
    
    Parameters:
        fw: CFAR kernel width 
        gw: number of guard cells
        thresh: detection threshold
    
    Returns:
        X with CFAR filter applied'''

    Tfilt = np.ones((fw,fw))/(fw**2 - gw**2)
    e1 = (fw - gw)//2
    e2 = fw - e1 + 1
    Tfilt[e1:e2, e1:e2] = 0

    CR = normalize(X) / (signal.convolve2d(X, Tfilt, mode='same', boundary='wrap') + 1e-10)
    if thresh is None:
        return CR
    else:
        return CR > thresh