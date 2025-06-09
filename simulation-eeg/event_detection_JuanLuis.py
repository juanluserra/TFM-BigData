#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from brian2 import *
import scipy
from scipy import signal
import ast
from typing import List, Tuple
from LennartToellke_files.Event_detection_Toellke import sharp_wave_detection, frequency_band_analysis


def event_detection(sig: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Detects sharp waves, theta waves, and gamma waves in a given signal and performs frequency analysis.
    This function needs a sample frequency of 1024 Hz.
    
    Parameters:
        sig (np.ndarray): The input signal from which events are to be detected.
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
            event, 1, 500, sample_frequency)
        if len(frequencies) != 0 and len(power_spectrum) != 0:
            # collect general event data
            filtered_events.append(filtered_event)
            duration = len(event) * record_dt * 1000
            all_durations.append(duration)
            peak_frequency = frequencies[argmax(power_spectrum)]
            all_spectrum_peaks.append(peak_frequency)

            # identify sharp wave ripples
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
        theta_spectrum.extend(frequency_band_analysis(
            event, 5, 10, sample_frequency)[1])
        gamma_spectrum.extend(frequency_band_analysis(
            event, 30, 100, sample_frequency)[1])
        ripple_spectrum.extend(frequency_band_analysis(
            event, 100, 250, sample_frequency)[1])

    # structure result data
    all_event_data = [event_signals, filtered_events, all_spectrum_peaks, all_durations]
    theta_data = [theta_waves, theta_wave_peaks, theta_wave_durations]
    gamma_data = [gamma_waves, gamma_wave_peaks, gamma_wave_durations]
    swr_data = [sharp_wave_ripples, sharp_wave_ripple_peaks, sharp_wave_ripple_durations]
    band_spectra = [theta_spectrum, gamma_spectrum, ripple_spectrum]

    return all_event_data, theta_data, gamma_data, swr_data, band_spectra

