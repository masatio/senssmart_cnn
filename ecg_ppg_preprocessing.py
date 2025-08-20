import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences, savgol_filter
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import median_filter
import random
import pandas as pd
from scipy.fft import rfft
from scipy.signal import correlate, hilbert

### Filtering of selected ECG channel : lowpass with cutoff = 150 Hz and notch filters for 50, 100 and 150 Hz

def moving_average(x, w):
    """ Moving average filtering.

    Parameters 
    ----------
    x: np.ndarray 
    w: integer - size of the moving window

    Returns
    -------
    np.ndarray
        Filtered signal
    """
    return np.convolve(x, np.ones(w), 'valid') / w

def low_pass_filter(x: np.ndarray, cutoff_freq: float, sampling_freq: float, order: int = 11) -> np.ndarray:
    """Butterworth low pass filtering.

    Parameters
    ----------
    x : np.ndarray
        2D array where each column is a signal to be filtered
    cutoff_freq : float
        Bandwidth of the low-pass filter
    sampling_freq : float
        The sampling frequency
    order : int, optional
        Order of the filter, by default 11

    Returns
    -------
    np.ndarray
        Filtered signals, with same shape as input x
    """
    wn = 2 * cutoff_freq / sampling_freq
    sos = butter(order, wn, output='sos')
    y = sosfiltfilt(sos, x, axis=0)
    return y

def high_pass_filter(x: np.ndarray, cutoff_freq: float, sampling_freq: float) -> np.ndarray:
    """High-pass filter.

    Parameters
    ----------
    x : np.ndarray
        2D array where each column is a signal to be filtered
    cutoff_freq : float
        Cut-off frequency
    sampling_freq : float
        Sampling frequency

    Returns
    -------
    np.ndarray
        Filtered signals, with same shape as input x
    """
    theta = 2 * np.pi * cutoff_freq / sampling_freq
    P = (1 - np.sin(theta)) / np.cos(theta)
    G = (1 + P) / 2
    b = [G, -G]
    a = [1, -P]
    y = filtfilt(b, a, x, axis=0)
    return y

def notch_filter(x: np.ndarray, stop_freq: float, sampling_freq: float) -> np.ndarray:
    """Return result of applying a notch filter at stop_freq to x.

    Parameters
    ----------
    x : np.ndarray
        Time series to be filtered. For 2d arrays, filtering is done along the
        row dimension (each column is a separate signal).
    stop_freq : float
        The frequency to be suppressed.
    sampling_freq : float
        The sampling frequency. stop_freq/sampling_freq must be bellow 0.5.

    Returns
    -------
    np.ndarray
        The zero-phase notch filtered signal, the same size as x.
    """
    b, a = iirnotch(stop_freq, Q=30, fs=sampling_freq)
    y = filtfilt(b, a, x, axis=0)
    return y

def filter_ecg(signal: np.ndarray, fs=500):
    """Low pass at 150 Hz and notch at 50, 100 and 150 Hz.
    """
    # Notch & lowpass filter(s)
    #for i in range(1,9):
    signal = low_pass_filter(signal, cutoff_freq=150, sampling_freq=fs)

    for stop_freq in [50, 100, 150]:
        signal = notch_filter(signal, stop_freq=stop_freq, sampling_freq=fs)
    signal = (signal - np.mean(signal)) / np.std(signal)

        # Modify ECG if it has outliers
        #signal[:,i] = hampel_filter(signal[:,i], percentile=0.95)

    return signal

### Finding R-peaks from ECG signals

def find_ecg_r_peak_indices(ecg, fs=500):
    """Given the 2D array (1st column = time, 2nd column = ECG), return the
    indices corresponding to R-peaks.
    """
    # Some ECG's have negative R peak values (switched electrodes?)
    x = np.abs(ecg[:, 4])
    t = ecg[:, 0]
    # Normalize
    x /= np.max(x)
    # R-peaks are assumed to have normalized height above 0.5
    # and to be at least 330 ms apart (heart rate of around 180 bpm)
    distance_ms = 330
    #sampling_time = t[1] - t[0]
    sampling_time = 1/fs*1000
    distance = int(distance_ms / sampling_time)
    peak_idx, _ = find_peaks(x, height=0.5, distance=distance)
    r_peak_times = t[peak_idx]

    return peak_idx, r_peak_times

def ppg_indices_of_r_peak_times(ppg_timestamps, r_peak_times):
    """Among the ppg_timestamps, find the points closest to the R-peak
    timestamps.
    """
    # Find the indices into the PPG signals that correspond to R peaks in ECG.
    return np.searchsorted(ppg_timestamps, r_peak_times, side='left')

### PPG processing

def filter_ppg(signal: np.ndarray, fs=500):
    # Band-pass filter between 1 Hz and 6 Hz
    signal = low_pass_filter(signal, cutoff_freq=6, sampling_freq=fs)
    signal = high_pass_filter(signal, cutoff_freq=1, sampling_freq=fs)
    # Signal normalization
    signal = (signal - np.mean(signal)) / np.std(signal)

    return signal

def split_ppg_to_beats(t, ppg_signal, r_peak_times):
    """
    Returns a list of PPG beats and time intervals, without the interpolation
    """
    #ppg_signal = filter_ppg(ppg_signal, fs)
    #ecg_signal = filter_ecg(ecg_signal, fs)
    #_, r_peak_times = find_ecg_r_peak_indices(ecg_signal, fs)
    r_peak_indices = ppg_indices_of_r_peak_times(t, r_peak_times)

    #median_len = round(np.median(np.diff(r_peak_indices))) # in samples
    #t_axis = np.linspace(0, median_len)/fs # in seconds

    ppg_beats = []
    beat_times = []
    for ind in range(0,len(r_peak_indices)-1):
        beat = ppg_signal[r_peak_indices[ind]:r_peak_indices[ind+1]]
        time = t[r_peak_indices[ind]:r_peak_indices[ind+1]] - t[r_peak_indices[ind]]
        ppg_beats.append(beat)
        beat_times.append(time)

    return beat_times, ppg_beats

def find_onset_beat_by_beat(t, ppg_signal, r_peak_times):
    """
    Returns a list of PPG onsets (their timestamps) for each beat of the signal
    """
    times, beats = split_ppg_to_beats(t, ppg_signal, r_peak_times)
    onsets = []

    for i in range(0, len(beats)):
        onset = find_beat_onset(beats[i], times[i])
        if(~np.isnan(onset)):
            onsets.append(times[i][onset])
        else:
            onsets.append(np.nan)
    return onsets

def find_notch_beat_by_beat(t, ppg_signal, r_peak_times):
    """
    Returns a list of PPG dicrotic notches (their timestamps) for each beat of the signal
    """
    times, beats = split_ppg_to_beats(t, ppg_signal, r_peak_times)
    dns = []

    for i in range(0, len(beats)):
        first, second = get_derivatives(beats[i], times[i][1]-times[i][0])
        sp = find_systolic_peaks(beats[i], times[i])

        if(~np.isnan(sp)):
            onset, _ = _get_dic_notch_and_dias_peak(beats[i], sp, second)
        else:
            onset = np.nan
        if(~np.isnan(onset)):
            dns.append(times[i][onset])
        else:
            dns.append(np.nan)
    return dns


def median_beat_interp(x, t, interval_limits, n_time_points=None, standardize=False):
    """Find median beat by interpolating individual beats onto a common time axis.

    Parameters
    ----------
    x : 1d array
        Sequence of signal samples, containing multiple beats (periods).
    t : 1d array
        Sequence of timestamps in milliseconds, same size as x.
    interval_limits : 1d array
        Sequence of timestamps in milliseconds that correspond to starts of individual beats.
    n_time_points : int, optional
        If specified, than the common time axis is defined by n_time_points
         equidistant points from 0 to 1.
         If None, than the common time axis is computed from a slice from t that
         corresponds to the beat whose duration is closest to the median duration.
         By default None.
    standardize : bool, optional
        If True, values of x are standardized to have zero mean and unit std
         over the course of each beat (standardization is per-beat). By default
         False.

    Returns
    -------
    median_waveform : 1d array
        Sequence corresponding to the median beat.
    median_time : 1d array
        Timestamps corresponding to the median_waveform.
        If n_time_points is None, this is a slice from t corresponding to the
        nearest-to-median-duration beat, shifted to start from 0.
        If n_time_points is an int, this is the normalized time axis with
        n_time_points equidistant points from 0 to 1.
    x_interp : 2d array
        Matrix where each column is one beat, interpolated onto the common
        time axis.
    """
    if n_time_points is None:
        # Determine common time axis
        # First find the interval whose duration is closest to median duration
        interval_durations = np.ediff1d(interval_limits)
        median_duration = np.median(interval_durations)
        idx_nearest_median = (np.abs(interval_durations - median_duration)).argmin()

        # Set common time axis as the axis of the chosen closest-to-median interval
        t_start = interval_limits[idx_nearest_median]
        t_end = interval_limits[idx_nearest_median + 1]
        t_axis = t[np.logical_and(t >= t_start, t < t_end)]
        # Map common axis to [0, 1]
        t_axis_ = (t_axis - t_axis[0]) / (t_axis[-1] - t_axis[0])
    else:
        t_axis_ = np.linspace(0, 1, num=n_time_points)

    # Interpolated (stretched or squeezed) intervals are stored in rows
    x_interp = np.empty((len(t_axis_), len(interval_limits) - 1))

    # Iterate over beats
    for i in range(len(interval_limits) - 1):
        # Get t and x for current interval
        interval_mask = np.logical_and(t >= interval_limits[i], t < interval_limits[i+1])
        tp = t[interval_mask]
        xp = x[interval_mask]

        if standardize:
            xp = (xp - xp.mean()) / xp.std()

        # Map time to [0, 1]
        tp_ = (tp - tp[0]) / (tp[-1] - tp[0])

        # Interpolate current interval onto common time axis
        x_interp[:, i] = np.interp(t_axis_, tp_, xp)

    # Get median over ensemble
    median_waveform = np.median(x_interp, axis=1)

    # Get corresponding common time
    if n_time_points is None:
        median_time = t_axis - t_axis[0]
    else:
        median_time = t_axis_

    #median_waveform = np.convolve(median_waveform, np.ones(5) / 5, mode='valid')
    median_waveform = savgol_filter(median_waveform, window_length=15, polyorder=3)

    return median_waveform, median_time, x_interp

def median_waveform_quality(median_waveform, x_interp):
    """ Returns the quality index that represents the average correlation between of median beat and interpolated beats
    """
    score = []
    for i in range(0, np.shape(x_interp)[1]):
        corr = np.corrcoef(median_waveform, x_interp[:,i])[0, 1]
        score.append(corr)

    return np.mean(score)

def get_derivatives(x, dt=1):
    """Returns the arrays of 1st and 2nd derivatives of x. dt is time difference
    between neighboring samples of x.
    """
    # Ensure results has same length as input by prepending 0.
    xp = np.diff(x, axis=0, prepend=0).astype(float)
    # Replace prepended 0 with 1st actual difference.
    xp[0] = xp[1]
    # Scale value difference with time difference to get derivative.
    xp /= dt

    # Same for 2nd derivative
    xs = np.diff(xp, axis=0, prepend=0)
    xs[0:2] = xs[2]
    xs /= dt

    return xp, xs

def find_systolic_peaks(median_waveform, median_time, most_prominent=False):

    # Find systolic peak
    dt = median_time[1] - median_time[0]
    #first_der = np.gradient(median_waveform)
    #start_ind = np.argmax(first_der)
    peak_idx, _ = find_peaks(median_waveform, height=0.3, distance=int(330 / dt))
    #sp_idx = peak_idx[peak_idx >= start_ind][0]
    #peak_idx = peak_idx[peak_idx <= min_samples]

    if len(peak_idx)==0:
        sp_idx = np.nan
    elif(most_prominent):
        sp_idx = peak_idx[median_waveform[peak_idx].argmax()]
    else:
        sp_idx = peak_idx[0]

    return sp_idx

def find_beat_onset(median_wave, median_time):

    dt = median_time[1] - median_time[0]
    first_der, second_der = get_derivatives(median_wave, dt)
    sp_idx = find_systolic_peaks(median_wave, median_time)
    min_interval = int(100/dt) # got the idea from pyPPG library, they said that 
    #systolic upslope's duration should be 100-120ms

    if(np.isnan(sp_idx)):
        return np.nan
    
    if(sp_idx - min_interval > 0):
        ind_interval = sp_idx - min_interval
    else:
        ind_interval = sp_idx
    
    max_idx = np.where(first_der[:ind_interval] == np.max(first_der[:ind_interval]))[0][0]
    on_idx = np.where(np.diff(np.sign(first_der[:max_idx]))>0)[0]

    if len(on_idx)==0:
        return np.argmin(median_wave[:sp_idx])
    else:
        return on_idx[-1]

def find_dicrotic_notch(median_waveform, median_time):

    dt = median_time[1]-median_time[0]
    first, second = get_derivatives(median_waveform, dt)

    # Find dicrotic notch after systolic peak
    sp_idx = find_systolic_peaks(median_waveform, median_time)
    diastolic_region = second[sp_idx:]
    notch_indices = np.where(np.diff(np.sign(diastolic_region)) > 0)[0]

    if len(notch_indices)==0:
        return np.argmax(first[sp_idx:])+sp_idx # incisure
    else:
        ni = np.argmin(median_waveform[notch_indices+sp_idx])
        print(ni)
        #return notch_indices[ni] + sp_idx
        return notch_indices[ni] + sp_idx

def _is_max_abs_1st_der_neg(x):
    """Return True if the peak in the 1st derivative of x with the highest
    absolute value is negative.

    Helper function used by is_inverted.
    """
    xp, _ = get_derivatives(x)
    return np.abs(xp.min()) > np.abs(xp.max())

def is_inverted(r_peak_idx, ppg, sampling_freq=500):
    """Returns True if the PPG sequence is inverted.

    Parameters
    ----------
    r_peak_idx : 1d array
        Indices of the PPG signals that correspond to R-peaks in the ECG.
    ppg : 1d array
        Sequence of PPG samples.
    sampling_frequency : float
        Frequency at which the PPG is sampled, in Hz.

    Returns
    -------
    bool
        True if the sequence is inverted.
    """
    # Filter ppg to remove respiration artefact
    ppg_ = np.copy(ppg)
    ppg_ = low_pass_filter(ppg_, cutoff_freq=10, sampling_freq=sampling_freq)
    ppg_ = high_pass_filter(ppg_, cutoff_freq=0.5, sampling_freq=sampling_freq)

    # For each beat (delimited by 2 consecutive r_pead_idx),
    # check if PPG has the expected shape corresponding to an
    # inverted signal: rises, then falls off, then rises again.
    num_rr_intervals = len(r_peak_idx) - 1
    n_inverted = 0

    # Loop over individual heartbeat intervals
    for i in range(num_rr_intervals):
        # Slice out the part of signal corresponding to the current RR-interval
        x = ppg_[r_peak_idx[i]: r_peak_idx[i + 1]]
        n_inverted += _is_max_abs_1st_der_neg(x)

    inverted = (n_inverted > num_rr_intervals / 2)

    return inverted


def _get_dic_notch_and_dias_peak(ppg, rel_sys_peak, ppg_sd, verbose=False):
    """Find the dicrotic notch (DN) and diastolic peak (DP) in a segment of PPG.

    Parameters
    ----------
    ppg : 1d array
        One segment of the PPG signal, corresponding to one RR interval.
    rel_sys_peak : int
        Index of the point in ppg which corresponds to the systolic peak.
    ppg_sd : 1d array
        Second derivative of PPG.
    verbose : bool, optional
        If True, intermediate results are printed on the console, by default False

    Returns
    -------
    rel_dic_notch, rel_dias_peak : int, int
        Indices of DN and DP relative to the passed PPG segment.
    """
    verboseprint = print if verbose else lambda *a, **k: None
    # DP must be at least 'distance' apart from SP, and prominent enough.
    prom_threshold = (ppg.max() - ppg.min()) * 0.005
    # Find peak offsets relative to systolic peak
    peaks, properties = find_peaks(ppg[rel_sys_peak:], prominence=prom_threshold)
    # Move peaks so they are relative to beat beginning
    peaks += rel_sys_peak
    if len(peaks) > 0:
        verboseprint(f"Found peaks at {peaks}, with prominences {properties['prominences']}, right of SP. Declaring 1st to be DP.")

        # The first peak is the SP (alternative: most prominent one)
        # rel_dias_peak = peaks[properties['prominences'].argmax()]
        rel_dias_peak = peaks[0]

        # DN is the min between the two peaks (SP and DP)
        dn_from_sp = np.argmin(ppg[rel_sys_peak+1:rel_dias_peak])
        rel_dic_notch = rel_sys_peak + dn_from_sp

    else:
        # TODO this always returns a results, but thing about how to filter
        # them, so that we only get DN and SP if we're confident that they
        # are really there.
        verboseprint("No peaks right of SP. Using 2nd derivative to search for DN and SP.")

        # There are a few options for finding the DN:
        # 1) max of 2nd derivative to the right of SP
        # 2) 1st local max of 2nd derivative to the right of SP
        # 3) most promiment peak of 2nd derivative right of SP

        # # Max of 2nd derivative to the right of SP is DN.
        # dn_from_sp = np.argmax(xs[rel_sys_peak:])
        # rel_dic_notch = rel_sys_peak + dn_from_sp

        prom_threshold = np.ptp(ppg_sd) * 0.005
        xs_peaks, properties = find_peaks(ppg_sd[rel_sys_peak:], prominence=prom_threshold)
        if len(xs_peaks) == 0:
            verboseprint("No peaks in 2nd derivative right of SP. Skipping to next interval.")
            rel_dic_notch = np.nan
            rel_dias_peak = np.nan
        
        else:
            # # 1st local max of 2nd derivative right of SP is DN.
            # verboseprint(f"Found {len(xs_peaks)} peaks at {xs_peaks} from SP in 2nd derivative. Setting DN to 1st one.")
            # dn_from_sp = xs_peaks[0]

            # Most prominent local max of 2nd derivative right of SP is DN.
            verboseprint(f"Found {len(xs_peaks)} peaks at {xs_peaks} from SP in 2nd derivative. Setting DN to most prominent one.")
            dn_from_sp = xs_peaks[properties['prominences'].argmax()]
            
            # Distance of DN from start of RR-interval 
            rel_dic_notch = rel_sys_peak + dn_from_sp

            # min of 2nd derivative to the right of DN is DP.
            dp_from_dn = np.argmin(ppg_sd[rel_dic_notch:])
            rel_dias_peak = rel_dic_notch + dp_from_dn       

    return rel_dic_notch, rel_dias_peak

