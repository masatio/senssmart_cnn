from turtle import clear, color, end_fill
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import scipy.signal
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch
from pathlib import Path
import os
import pandas as pd
import csv
from ecg_ppg_preprocessing import median_beat_interp
from ppg_testing import read_data

def read_mic_data(root, filename):
    filepath = os.path.join(root, filename)
    mic_data = pd.read_csv(filepath)

    mic = mic_data['M1'].to_numpy()
    tm = mic_data['Tm'].to_numpy()

    filename_ecg = filename[:-7]+"ecg.csv"
    filepath = os.path.join(root, filename_ecg)
    ecg_data = pd.read_csv(filepath)

    ecg_signal = ecg_data.to_numpy()
    t_ecg = ecg_data['Tm'].to_numpy()

    return tm, mic, t_ecg, ecg_signal

def mic_indices_of_r_peak_times(mic_timestamps, r_peak_times):
    """Among the mic_timestamps, find the points closest to the R-peak
    timestamps.
    """
    # Find the indices into the PPG signals that correspond to R peaks in ECG.
    return np.searchsorted(mic_timestamps, r_peak_times, side='left')

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

def filter_signal(mic_data, fs_mic):
    """Filters the signal, without adding phase delay, using 6th order Butterworth bandpass filter with
    cutoff frequencies 25Hz and 100Hz. Intended for microphone signal.

    Returns
    -------
    mic_data_filt: filtered and normalized mic signal
    """
    # Creating a bandpass Butterworth filter with cutoff frequencies 25 and 100
    sos = sp.signal.butter(6, np.array([25, 100]), 'bandpass', fs=fs_mic, output='sos')
    mic_data_filt = sp.signal.sosfilt(sos, mic_data)
    # b,a = sp.signal.butter(3,np.array([25,100]),btype = 'bandpass',fs=fs_mic)
    # Filtering the data without phase delay
    # mic_data_filt = scipy.signal.filtfilt(b,a,mic_data)
    mic_data_filt = mic_data_filt / np.max(np.abs(mic_data_filt))
    return mic_data_filt

def simple_moving_average(signal, window=150):
    """Implements a simple moving average filter, with the defined window size
    """
    return np.convolve(signal, np.ones(window) / window, mode='same')

def shannon_envelope(x, window_size=100):
    """Returns a normalized Shannon energy envelope of the signal, with the defined window size
    """
    E = -(x ** 2) * np.log(x ** 2 + 1e-8)
    E_ma = simple_moving_average(E, window_size)
    E_ma = E_ma / np.max(E_ma)

    return E_ma


def detect_S1S2(E, R_ts, mic_axis):
    """Detects time stamps of S1 and S2 beginnings.
    Iterates through every cycle (R to R peak) to find three most prominent peaks.
    Peak that comes first in the cycle is assumed to be S1, and the one after that S2.
    In certain signals a heart sound in the place of S3 appears that is more prominent that S2. By finding 3 most
    prominent peaks, we make sure that S2 is found even when it is the third most prominent.

    Parameters
    ----------
    E: normalized energy envelope,
    R_ts: array of time stamps of R peaks in ECG

    Returns
    -------
    s1_ts: array of time stamps of S1 beginnings
    s2_ts: array of time stamps of S2 beginnings

    """
    s1_ts = np.array([])
    s2_ts = np.array([])
    s1_peak_ts = np.array([])
    s2_peak_ts = np.array([])

    """

    """

    for i in range(len(R_ts) - 1):
        dt = 0.06
        cycle = E[(mic_axis > R_ts[i] - dt) & (
                    mic_axis < R_ts[i + 1] - dt)]  # Energy of one cycle (from one R peak to the next one)
        cycle_axis = mic_axis[(mic_axis > R_ts[i] - dt) & (mic_axis < R_ts[i + 1] - dt)]  # Corresponding axis
        num_sam = 50  # minimal number of samples between two peaks -> 50ms
        peaks, _ = find_peaks(cycle, distance=num_sam)
        prominences, left_bases, _ = peak_prominences(cycle, peaks, wlen=300)
        idx = np.argsort(prominences)
        idx = np.flip(idx)  # indexes of three dominant peaks (starting with most prominent)
        left_bases = left_bases[idx]  # sorts peaks and left bases based on prominence in descending order
        left_bases = cycle_axis[left_bases]  # time stamps of left bases
        peaks = peaks[idx]
        peaks = cycle_axis[peaks]  # time stamps of peaks
        # time stamps of 3 most prominent peaks and their left bases
        # we assume that the peak that comes first in the cycle is S1, than S2
        s1s2_peaks = peaks[0:2]

        if(len(s1s2_peaks) == 0):
            s1_ts = np.append(s1_ts, np.nan)
            s2_ts = np.append(s2_ts, np.nan)
            s1_peak_ts = np.append(s1_peak_ts, np.nan)
            s2_peak_ts = np.append(s2_peak_ts, np.nan)
            
        else:
            s1s2_left_bases = left_bases[0:2]
            # s1s2_left_bases = peaks[0:2]
            idx1 = np.argmin(s1s2_peaks)
            idx2 = np.argmax(s1s2_peaks)
            # print(s1s2_left_bases[idx1],s1s2_left_bases[idx2])
            s1_ts = np.append(s1_ts, s1s2_left_bases[idx1])
            s2_ts = np.append(s2_ts, s1s2_left_bases[idx2])
            s1_peak_ts = np.append(s1_peak_ts, s1s2_peaks[idx1])
            s2_peak_ts = np.append(s2_peak_ts, s1s2_peaks[idx2])
    
    return s1_ts, s2_ts, s1_peak_ts, s2_peak_ts


def plot_mic_ecg(mic_data, E, mic_axis, s1_ts, s2_ts, ecg, ecg_axis, R_ts):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    axs[0].plot(mic_axis, E, label='Energy envelope')
    axs[0].scatter(s1_ts, np.zeros(len(s1_ts)), color='red', s=30, zorder=3, label='S1')
    axs[0].scatter(s2_ts, np.zeros(len(s2_ts)), color='green', s=30, zorder=3, label='S2')
    axs[0].vlines(x=R_ts, ymin=0, ymax=1, colors='orange', label='R peak (ECG)', linestyles='dashed', zorder=3)
    axs[1].plot(mic_axis, mic_data / np.max(np.abs(mic_data)), 'black')
    axs[1].scatter(s1_ts, np.zeros(len(s1_ts)), color='red', s=20, zorder=3, label='S1')
    axs[1].scatter(s2_ts, np.zeros(len(s2_ts)), color='green', s=20, zorder=3, label='S2')
    #axs[2].plot(ecg_axis, ecg[:, 1], label='Energy envelope')
    # axs[1].vlines(x = s1_ts, ymin = -1, ymax = 1,colors = 'black',label = 'S1',zorder = 3)
    # xs[1].vlines(x = s2_ts, ymin = -1, ymax = 1,colors = 'red',label = 'S2',zorder = 3)
    # axs[1].plot(mic_axis,mic_orig/np.max(np.abs(mic_orig)))
    axs[1].vlines(x=R_ts, ymin=-1, ymax=1, colors='orange', label='R peak (ECG)', linestyles='dashed', zorder=3)
    axs[0].legend()
    axs[1].legend()
    plt.xlabel('t [s]')
    plt.show()
    return

if __name__ == '__main__':
    root = 'C:/Users/masat/Desktop/Master rad/Klinika/Klinika csv'
    root_out = 'C:/Users/masat/Desktop/ETRAN_2025'

    info = pd.read_csv(root_out + '/Datasets/information.csv')
    subjects = info.iloc[:, 0].tolist()
    filenames = [name + "_mic.csv" for name in subjects]

    ecg_data = read_data(root_out, '/ECG_parameters/r_peaks.csv', subjects)
    
    counter = 0
    intervals = []

    for filename in filenames:
        tm, mic_data, t_ecg, ecg = read_mic_data(root, filename)
        fs_mic = np.ceil(len(tm) / tm[len(tm) - 1]/0.001)
        mic_orig = mic_data
        mic_data = filter_signal(mic_data, fs_mic)
        # modified 1:
        #mic_data, tm, x_interp = median_beat_interp(mic_data, tm, ecg_data[counter])

        # Find signal envelope
        window_size = 120
        E = shannon_envelope(mic_data, window_size)
        b, a = sp.signal.butter(3, 200, btype='low', fs=fs_mic)
        E = sp.signal.filtfilt(b, a, E)
        # E = scipy.signal.savgol_filter(E, window_length=151, polyorder=5, mode="nearest") #savitzky-golay filter
        # ecg = np.zeros((len(ecg_axis),13))
        # to use the QRJ annotator we need ECG data in the form of time + 12 columns
        # since we only have 8 channels in the original ECG, the other 4 are copies of channel 1

        # Modify 2: commented this R_ts
        R_ts = tm[mic_indices_of_r_peak_times(tm, ecg_data[counter])]*0.001  # R_ts are time stamps of R peaks
        #R_ts = [tm[0]*0.001, tm[-1]*0.001]
        counter = counter + 1

        s1_ts, s2_ts, _, _ = detect_S1S2(E, R_ts, tm*0.001)
        #arr = s1_ts - R_ts[:-1]
        intervals.append(s2_ts - R_ts[:-1])

        # R_s1_ptp = np.append(R_s1_ptp,np.sum(R_ts[0:len(s1_ts)]-s1_ts)/len(s1_ts))
        plot_mic_ecg(mic_data, E, tm*0.001, s1_ts, s2_ts, ecg, t_ecg*0.001, R_ts)
        # break
# plt.hist(R_s1_ptp)
# plt.show()
  #  with open(root_out + "/PCG_parameters/s2_r_median_beat.csv", "w", newline="") as csvfile:
  #      writer = csv.writer(csvfile)

        # Writing header
        #writer.writerow(["Name", "QS2"])

        # Writing sorted data
   #     for i in range(len(intervals)):
   #         writer.writerow([subjects[i]] + list(intervals[i]*1000))
pass

