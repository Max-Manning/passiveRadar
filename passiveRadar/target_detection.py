import numpy as np
from passiveRadar.signal_utils import normalize
import scipy.signal as signal


# create a data type to represent the Kalman filter's internals
kalman_filter_dtype = np.dtype([
    ('x' , np.float, (4, )),   # state estimate vector
    ('P' , np.float, (4,4)),   # state estimate covariance matrix
    ('F1', np.float, (4,4)),   # state transition model #1
    ('F2', np.float, (4,4)),   # state transition model #1
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