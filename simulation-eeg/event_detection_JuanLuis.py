#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from brian2 import *
import scipy
from scipy import signal
import ast
from typing import List, Tuple
from LennartToellke_files.Event_detection_Toellke import band_filter, window_rms

def frequency_band_analysis(
    event: np.ndarray, 
    low: float, 
    high: float, 
    samp_freq: float, 
    scaling: str = "density"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Analyzes the frequency band of a given event by applying a bandpass filter and computing the power spectrum.

        Parameters:
            event (array-like): The input signal or event data to analyze.
            low (float): The lower cutoff frequency of the bandpass filter.
            high (float): The upper cutoff frequency of the bandpass filter.
            samp_freq (float): The sampling frequency of the input signal.
            scaling (str): The scaling method for the power spectrum, default is "density".

        Returns:
            tuple: A tuple containing:
                - array: The frequencies corresponding to the power spectrum.
                - array: The power spectrum of the filtered signal.
                - array: The filtered signal after applying the bandpass filter, or empty lists if analysis fails.
    """
    try:
        filtered_sig = band_filter(event, low, high, samp_freq)
        frequencies, power_spectrum = signal.periodogram(
            filtered_sig, samp_freq / Hz, 'flattop', scaling=scaling)

        return frequencies, power_spectrum, filtered_sig
    except:
        return [], [], []

def sharp_wave_detection(
    sig: np.ndarray, 
    boundary_condition: float, 
    peak_condition: float, 
    record_dt: float
) -> List[np.ndarray]:
    """
        Detects sharp waves in a signal based on specified boundary and peak conditions.

        Parameters:
            sig (array-like): The input signal from which sharp waves are to be detected.
            boundary_condition (float): The multiplier for standard deviation to define the boundary for detection.
            peak_condition (float): The multiplier for standard deviation to define the peak condition for detection.
            record_dt (float): The time step of the recorded signal, used for RMS calculation.

        Returns:
            list: A list of arrays, each containing the segments of the signal that correspond to detected sharp waves.
    """
    # calculation of root-mean-square
    start_plot_time = 50 * msecond
    start_ind = int(start_plot_time / record_dt)
    sig_rms = window_rms(
        sig[start_ind:] - mean(sig[start_ind:]), int(10 * ms / record_dt))
    sig_std = std(sig_rms)

    boundary_value = boundary_condition * sig_std
    peak_value = peak_condition * sig_std

    # detection of sharp waves
    begin = 0
    peak = False
    all_sharp_waves = []

    for ind in range(len(sig_rms)):
        rms = sig_rms[ind]
        if rms > boundary_value and begin == 0:
            # event start
            begin = ind
        if rms > peak_value and not peak:
            # event fulfills peak condition
            peak = True
        elif rms < boundary_value and peak:
            # sharp wave detected
            sharp_wave_signal = sig[begin + start_ind:ind + start_ind]
            all_sharp_waves.append(sharp_wave_signal)
            begin = 0
            peak = False
        elif rms < boundary_value and not peak:
            # event without sufficient peak
            begin = 0
            peak = False

    return all_sharp_waves

def event_detection(
    sig: np.ndarray,
    scaling: str = "density",
    theta_band: Tuple[float, float] = (5, 10),
    gamma_band: Tuple[float, float] = (30, 100),
    ripple_band: Tuple[float, float] = (100, 250)
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Detects sharp waves, theta waves, and gamma waves in a given signal and performs frequency analysis.
    This function needs a sample frequency of 1024 Hz.
    
    Parameters:
        sig (np.ndarray): The input signal from which events are to be detected.
        scaling (str): The scaling method for the power spectrum, default is "density".
    Returns:
        Tuple (containing lists):
            - List of detected sharp wave events.
            - List of theta wave events with their peaks and durations.
            - List of gamma wave events with their peaks and durations.
            - List of sharp wav e ripple events with their peaks and durations.
            - List of power spectra for theta, gamma, and ripple bands.
    """
    # model specific signal properties
    sample_frequency = 1024 * Hz
    record_dt = 1 / sample_frequency

    event_signals = sharp_wave_detection(sig, 3, 4.5, record_dt)

    # frequency analysis
    filtered_events = []
    all_spectrum_peaks = []
    all_durations = []

    theta_waves = []
    theta_wave_peaks = []
    theta_wave_durations = []
    
    gamma_waves = []
    gamma_wave_peaks = []
    gamma_wave_durations = []
    
    sharp_wave_ripples = []
    sharp_wave_ripple_peaks = []
    sharp_wave_ripple_durations = []

    theta_spectrum = []
    gamma_spectrum = []
    ripple_spectrum = []

    for event in event_signals:
        # filter in broad frequency range
        frequencies, power_spectrum, filtered_event = frequency_band_analysis(
            event, 1, 400, sample_frequency, scaling=scaling)
        if len(frequencies) != 0 and len(power_spectrum) != 0:
            # collect general event data
            filtered_events.append(filtered_event)
            duration = len(event) * record_dt * 1000
            all_durations.append(duration)
            peak_frequency = frequencies[argmax(power_spectrum)]
            all_spectrum_peaks.append(peak_frequency)

            # identify waves
            if 5 <= peak_frequency <= 10:
                theta_waves.append(event)
                theta_wave_peaks.append(peak_frequency)
                theta_wave_durations.append(duration)
            elif 30 <= peak_frequency < 100:
                gamma_waves.append(event)
                gamma_wave_peaks.append(peak_frequency)
                gamma_wave_durations.append(duration)
            elif 100 <= peak_frequency <= 250:
                sharp_wave_ripples.append(event)
                sharp_wave_ripple_peaks.append(peak_frequency)
                sharp_wave_ripple_durations.append(duration)

        # collect power of frequency bands
        theta_spectrum.extend(
            frequency_band_analysis(event, theta_band[0], theta_band[1], sample_frequency, scaling=scaling)[1]
        )
        gamma_spectrum.extend(
            frequency_band_analysis(event, gamma_band[0], gamma_band[1], sample_frequency, scaling=scaling)[1]
        )
        ripple_spectrum.extend(
            frequency_band_analysis(event, ripple_band[0], ripple_band[1], sample_frequency, scaling=scaling)[1]
        )

    # structure result data
    all_event_data = [event_signals, filtered_events, all_spectrum_peaks, all_durations]
    theta_data = [theta_waves, theta_wave_peaks, theta_wave_durations]
    gamma_data = [gamma_waves, gamma_wave_peaks, gamma_wave_durations]
    swr_data = [sharp_wave_ripples, sharp_wave_ripple_peaks, sharp_wave_ripple_durations]
    band_spectra = [theta_spectrum, gamma_spectrum, ripple_spectrum]

    return all_event_data, theta_data, gamma_data, swr_data, band_spectra

