import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import h5py
import errno

from passiveRadar.plotting_tools   import persistence
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

def initialize_track(measurement):
    '''Create a new target track with default parameter values

    Parameters:
        measurement:            first measurement for this target track
            if measurement is None, the track is initialized in the 'free' state
            (status = 0) at an arbitrary set of coordinates
            if measurement is a set of coordinates, the track is initialized at 
            these coordinates the 'preliminary' state (status = 1)
    Returns:    
        initialTrackerState:    target_track_dtype containing the initialized
                                target track
    '''
    if measurement is None:
        r = 0
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

def track_update(currentState, newMeasurement):


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

def associate_measurements(trackState, candidateMeas):

    '''Associate a set of candidate measurements with a target track
    
    Parameters:
        trackState:
        candidateMeas:
    Returns:
        newMeasurement:
        candidateMeas:
    
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
            validationGate[kk] = zerr.T @ np.linalg.inv(S) @ zerr < 12
            
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

    # loop through frames of the input data
    for i in range(Nframes):

        # get the new candidate measurements for this frame
        dataFrame = data[:,:,i]
        candidateMeas = get_measurements(dataFrame, 99.8, frame_extent)

        # find out which of the tracks are in the confirmed, preliminary and 
        # free states
        confirmedTracks = np.argwhere(trackStates['status'] == 2).flatten()
        prelimTracks    = np.argwhere(trackStates['status'] == 1).flatten()
        freeTracks      = np.argwhere(trackStates['status'] == 0).flatten()

        for track_idx in confirmedTracks:
            # loop over the confirmed tracks first (confirmed tracks get first
            # access to the candidate measurements.)
            trackState = trackStates[track_idx]
            newMeasurement, candidateMeas = associate_measurements(trackState, candidateMeas)
            newTrackState = track_update(trackState, newMeasurement)
            trackStates[track_idx] = newTrackState

        for track_idx in prelimTracks:
            # Next we update the tracks that are in the preliminary state
            trackState = trackStates[track_idx]
            newMeasurement, candidateMeas = associate_measurements(trackState, candidateMeas)
            newTrackState = track_update(trackState, newMeasurement)
            trackStates[track_idx] = newTrackState

        for track_idx in freeTracks:
            # Finally we assign free tracks to remaining measurements
            trackState = trackStates[track_idx]
            if candidateMeas.size == 0:
                print("no more measurements available, track remaining free")
                break
            newMeasurement, candidateMeas = associate_measurements(trackState, candidateMeas)
            newTrackState = initialize_track(newMeasurement)
            trackStates[track_idx] = newTrackState
    
        tracker_history[i,:] = trackStates

    return tracker_history

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

if __name__ == "__main__":

    # get the current passive radar config file
    config_file = open('PRconfig.yaml', 'r')
    config_params = yaml.safe_load(config_file)
    config_file.close()
    xambgfile           = config_params['outputFile']
    blockLength         = config_params['blockLength']
    channelBandwidth    = config_params['channelBandwidth']
    rangeCells          = config_params['rangeCells']
    dopplerCells        = config_params['dopplerCells']

    # length of the coherent processing interval in seconds
    cpi_seconds = blockLength/channelBandwidth
    # range extent in km
    range_extent = rangeCells*3e8/(channelBandwidth*1000)
    # doppler extent in Hz
    doppler_extent = dopplerCells/(2 * cpi_seconds)

    # load the processed passive radar data (range-doppler maps)
    if not os.path.isfile(xambgfile):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), xambgfile)
    f = h5py.File(xambgfile, 'r')
    xambg = np.abs(f['/xambg'])
    Nframes = xambg.shape[2]
    f.close()

    # CFAR filter each frame using a 2D kernel
    CF = np.zeros(xambg.shape)
    for i in range(Nframes):
        CF[:,:,i] = CFAR_2D(xambg[:,:,i], 18, 4)

    N_TRACKS = 20

    tracker_history = multitarget_tracker(CF, [256/1.092, 375], N_TRACKS)

    tracker_status = tracker_history['status']
    tracker_status_confirmed_idx = np.nonzero(tracker_status != 2)

    promoted_tracks_idx = (tracker_status == 1) & (np.roll(tracker_status, -10, axis=0) == 2)
    promoted_track_range = np.squeeze(tracker_history['estimate'][:,:,0]).copy()
    promoted_track_doppler = np.squeeze(tracker_history['estimate'][:,:,1]).copy()
    promoted_track_range[~promoted_tracks_idx] = np.nan
    promoted_track_doppler[~promoted_tracks_idx] = np.nan

    tracker_range = np.squeeze(tracker_history['estimate'][:,:,0]).copy()
    tracker_doppler = np.squeeze(tracker_history['estimate'][:,:,1]).copy()
    tracker_range[tracker_status_confirmed_idx] = np.nan
    tracker_doppler[tracker_status_confirmed_idx] = np.nan
    
    # from svgpath2mpl import parse_path
    
    # target_rectangle = parse_path(""""M1438 12630 c-331 -30 -643 -176 -879 -411 -205 -205 -345 -477 -396
    #     -769 -17 -101 -18 -309 -18 -5060 0 -4751 1 -4959 18 -5060 105 -602 556
    #     -1053 1167 -1167 90 -17 351 -18 5070 -18 4719 0 4980 1 5070 18 302 56 564
    #     191 771 398 205 205 345 477 396 769 17 101 18 309 18 5060 0 4751 -1 4959
    #     -18 5060 -79 455 -361 834 -774 1039 -107 53 -249 100 -383 128 -80 16 -386
    #     17 -5020 19 -2714 1 -4974 -2 -5022 -6z m9930 -648 c89 -27 198 -80 261 -126
    #     111 -81 214 -215 270 -350 66 -158 61 244 61 -5086 0 -5306 4 -4923 -59 -5079
    #     -97 -237 -279 -403 -531 -483 l-75 -23 -4925 0 c-4728 0 -4928 1 -4990 18
    #     -272 76 -487 285 -572 557 l-23 75 0 4930 c0 4733 1 4933 18 4995 85 302 342
    #     539 640 589 29 5 2184 8 4952 7 l4900 -1 73 -23z""")

    # target_rectangle.vertices -= target_rectangle.vertices.mean(axis=0)

    # for kk in range(5185, Nframes):
    #     print(kk)
    #     frame = persistence(data, kk, 30, 0.91)
    #     frame = np.fliplr(frame.T) # get the orientation right
        
    #     svname = '.\\IMG2\\img_' + "{0:0=3d}".format(kk) + '.png'

    #     figure = plt.figure(figsize = (8,4.5))

    #     # get max and min values for color map
    #     vmn = np.percentile(frame.flatten(), 93)
    #     vmx = 1.8*np.percentile(frame.flatten(),99)

    #     # plot
    #     plt.imshow(frame,cmap = 'gnuplot2', vmin=vmn, vmax=vmx, extent = [-256/1.092,256/1.092,0,375], aspect='auto')
    #     # plt.ylabel('Range (km)')
    #     # plt.xlabel('Doppler (Hz)')
    #     if kk>0:
    #         st = 1000*int(np.floor(max(0, kk-1000)/1000))
    #         nr = np.arange(kk-st)
    #         cc = np.zeros((kk-st, 4))
    #         cc[:,1] = 0.9
    #         cc[:,2] = 0.2 
    #         cc[:,3] = np.flip(0.98**nr) # alpha channel


    #         cc2 = np.zeros((kk-st, 4))
    #         cc2[:,0] = 1.0
    #         cc2[:,2] = 0.2 
    #         cc2[:,3] = np.flip(0.98**nr) # alpha channel


    #         plt.scatter(promoted_track_doppler[st:kk,0], promoted_track_range[st:kk,0], 20,  marker='.',  color = cc2)
    #         plt.scatter(promoted_track_doppler[st:kk,1], promoted_track_range[st:kk,1], 20,  marker='.',  color = cc2)
    #         plt.scatter(promoted_track_doppler[st:kk,2], promoted_track_range[st:kk,2], 20,  marker='.',  color = cc2)
    #         plt.scatter(promoted_track_doppler[st:kk,3], promoted_track_range[st:kk,3], 20,  marker='.',  color = cc2)
    #         plt.scatter(promoted_track_doppler[st:kk,4], promoted_track_range[st:kk,4], 20,  marker='.',  color = cc2)
    #         plt.scatter(promoted_track_doppler[st:kk,5], promoted_track_range[st:kk,5], 20,  marker='.',  color = cc2)
    #         plt.scatter(promoted_track_doppler[st:kk,6], promoted_track_range[st:kk,6], 20,  marker='.',  color = cc2)
    #         plt.scatter(promoted_track_doppler[st:kk,7], promoted_track_range[st:kk,7], 20,  marker='.',  color = cc2)

    #         plt.scatter(tracker_doppler[st:kk,0], tracker_range[st:kk,0], 20,  marker='.',  color = cc)
    #         plt.scatter(tracker_doppler[st:kk,1], tracker_range[st:kk,1], 20,  marker='.',  color = cc)
    #         plt.scatter(tracker_doppler[st:kk,2], tracker_range[st:kk,2], 20,  marker='.',  color = cc)
    #         plt.scatter(tracker_doppler[st:kk,3], tracker_range[st:kk,3], 20,  marker='.',  color = cc)
    #         plt.scatter(tracker_doppler[st:kk,4], tracker_range[st:kk,4], 20,  marker='.',  color = cc)
    #         plt.scatter(tracker_doppler[st:kk,5], tracker_range[st:kk,5], 20,  marker='.',  color = cc)
    #         plt.scatter(tracker_doppler[st:kk,6], tracker_range[st:kk,6], 20,  marker='.',  color = cc)
    #         plt.scatter(tracker_doppler[st:kk,7], tracker_range[st:kk,7], 20,  marker='.',  color = cc)

    #         plt.scatter(tracker_doppler[kk,0], tracker_range[kk,0], 300,  marker=target_rectangle,  color = np.array([0,1,0,0.5]))
    #         plt.scatter(tracker_doppler[kk,1], tracker_range[kk,1], 300,  marker=target_rectangle,  color = np.array([0,1,0,0.5]))
    #         plt.scatter(tracker_doppler[kk,2], tracker_range[kk,2], 300,  marker=target_rectangle,  color = np.array([0,1,0,0.5]))
    #         plt.scatter(tracker_doppler[kk,3], tracker_range[kk,3], 300,  marker=target_rectangle,  color = np.array([0,1,0,0.5]))
    #         plt.scatter(tracker_doppler[kk,4], tracker_range[kk,4], 300,  marker=target_rectangle,  color = np.array([0,1,0,0.5]))
    #         plt.scatter(tracker_doppler[kk,5], tracker_range[kk,5], 300,  marker=target_rectangle,  color = np.array([0,1,0,0.5]))
    #         plt.scatter(tracker_doppler[kk,6], tracker_range[kk,6], 300,  marker=target_rectangle,  color = np.array([0,1,0,0.5]))
    #         plt.scatter(tracker_doppler[kk,7], tracker_range[kk,7], 300,  marker=target_rectangle,  color = np.array([0,1,0,0.5]))


        # plt.xlim([-200, 200])
        # plt.xticks([])
        # plt.yticks([])
        # plt.ylim([0, 150])
        # plt.tight_layout()
        # plt.savefig(svname, dpi=150)
        # plt.close()

    plt.figure(figsize = (16, 10))
    plt.scatter(tracker_doppler[:,0], tracker_range[:,0], marker='.')
    plt.scatter(tracker_doppler[:,1], tracker_range[:,1], marker='.')
    plt.scatter(tracker_doppler[:,2], tracker_range[:,2], marker='.')
    plt.scatter(tracker_doppler[:,3], tracker_range[:,3], marker='.')
    plt.scatter(tracker_doppler[:,4], tracker_range[:,4], marker='.')
    plt.scatter(tracker_doppler[:,5], tracker_range[:,5], marker='.')
    plt.scatter(tracker_doppler[:,6], tracker_range[:,6], marker='.')
    plt.scatter(tracker_doppler[:,7], tracker_range[:,7], marker='.')
    plt.scatter(tracker_doppler[:,8], tracker_range[:,8], marker='.')
    plt.scatter(tracker_doppler[:,9], tracker_range[:,9], marker='.')
    plt.scatter(tracker_doppler[:,10], tracker_range[:,10], marker='.')
    plt.scatter(tracker_doppler[:,11], tracker_range[:,11], marker='.')
    plt.scatter(tracker_doppler[:,12], tracker_range[:,12], marker='.')
    plt.scatter(tracker_doppler[:,13], tracker_range[:,13], marker='.')
    plt.scatter(tracker_doppler[:,14], tracker_range[:,14], marker='.')
    plt.scatter(tracker_doppler[:,15], tracker_range[:,15], marker='.')
    plt.scatter(tracker_doppler[:,16], tracker_range[:,16], marker='.')
    plt.scatter(tracker_doppler[:,17], tracker_range[:,17], marker='.')
    plt.scatter(tracker_doppler[:,18], tracker_range[:,18], marker='.')
    plt.scatter(tracker_doppler[:,19], tracker_range[:,19], marker='.')
    plt.xlim([-200, 200])
    plt.ylim([0, 150])
    plt.xlabel("Doppler Shift (Hz)")
    plt.ylabel("Bistatic Range (km)")
    # plt.savefig("kalman_multitracking_1.png", dpi=300)
    plt.show()

    plt.figure(figsize = (16, 10))
    plt.subplot(2,1,1)
    plt.plot(tracker_range)
    plt.plot(promoted_track_range)
    plt.ylim([0, 150])
    plt.xlabel("Time (s)")
    plt.ylabel("Bistatic Range (km)")
    plt.subplot(2,1,2)
    plt.plot(promoted_track_doppler)
    plt.plot(tracker_doppler)
    plt.ylim([-200, 200])
    plt.xlabel("Time (s)")
    plt.ylabel("Doppler Shift (Hz)")
    # plt.savefig("kalman_multitracking_2.png", dpi=300)
    plt.show()
