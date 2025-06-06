import ast

import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
from .Event_detection_Toellke import *
import os


def create_list_from_timeSeries(file_path):
    """
    Reads a file containing a single line list of floats in scientific notation and returns it as a list of floats.

    Parameters:
        file_path (str): The root path of the file to read from.

    Returns:
        list: A list of floats contained in the file.
    """
    with open(file_path, 'r') as file:
        # Read the single line from the file
        data_line = file.readline().strip()

        # Evaluate the line to convert it into a list
        # ast.literal_eval safely evaluates an expression node or a string containing a Python expression
        data_list = ast.literal_eval(data_line)

        return data_list


def plot_power_spectral_density(frequencies, power_densities):
    """
        Plots the power spectral density (PSD) against frequency.

        Parameters:
            frequencies (array-like): A list or array of frequency values (in Hz).
            power_densities (array-like): A list or array of power spectral density values corresponding to the frequencies.

        Returns:
            None: This function does not return a value; it directly creates and displays a line plot of the PSD.
    """
    plt.figure(figsize=(10, 5))
    # Log scale for better visibility of peaks
    plt.plot(frequencies, power_densities)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V/Hz]')
    plt.show()


def plot_lfp(recordings, sim_label, save=False):
    """
    Plots local field potentials over time and saves the plot as a PNG file.

    Parameters:
        recordings (list of float): The local field potential data.
        sim_label (str): Specific name to include in the file name and plot title.
    """

    timestep_ms = 0.9765625
    # Generate time points starting from 0, incrementing by the specified timestep for each data point
    time_points = [i * timestep_ms for i in range(len(recordings))]

    # Create a plot
    plt.figure(figsize=(10, 4))
    plt.plot(time_points, recordings, label='Local Field Potentials')
    plt.xlabel('Time (ms)')
    plt.ylabel('Potential (V)')
    plt.title(f'{sim_label}')
    plt.grid(True)

    # Set y-axis tick labels at multiples of 1e-6 from -1e-6 to 5e-6
    tick_values = [-1e-4, 0, 1e-4, 2e-4, 3e-4]
    tick_labels = ['-1', '0', '1', '2', '3']
    plt.yticks(tick_values, tick_labels)

    # Add scaling factor above y-axis
    plt.text(0.01, 1.02, '1e-4', transform=plt.gca().transAxes,
             ha='left', va='bottom')

    if not save:
        plt.show()
    else:
        path = 'images'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'images/{sim_label}_LFP.png')


def plot_full_length_lfp(file_path, save=False):
    """
        Plots the full-length local field potential (LFP) data from a specified file in frames.

        Parameters:
            file_path (str): The path to the file containing the LFP time series data.

        Returns:
            None: This function does not return a value; it directly plots the LFP frames.
    """
    recordings = create_list_from_timeSeries(file_path)

    recordings_per_second = 1024
    lfp_frame_length = 10 * recordings_per_second
    lfp_frames_in_recording = len(recordings) // lfp_frame_length

    for i in range(lfp_frames_in_recording):
        lfp_frame = recordings[(
            0 + i) * lfp_frame_length: (1 + i) * lfp_frame_length]
        plot_lfp(lfp_frame, f"Frame nr. {i + 1}", save=True)


def plot_line_diagram(label_value_list, y_label, x_label="", title="", comp_values=[], axis=0):
    """
        Plots a line diagram with optional comparison values, displaying mean values and standard deviations.

        Parameters:
            label_value_list (list of tuples): A list where each tuple contains a label and a list of values
                                                (e.g., [("label1", [value1, value2, ...]), ...]).
            y_label (str): The label for the y-axis.
            x_label (str, optional): The label for the x-axis. Defaults to an empty string.
            title (str, optional): The title of the plot. Defaults to an empty string.
            comp_values (list of tuples, optional): A list of comparison values in the same format as label_value_list.
                                                     Defaults to an empty list.
            axis (matplotlib.axes.Axes, optional): The axis to plot on. If 0, a new figure is created. Defaults to 0.

        Returns:
            None: This function does not return a value; it directly creates and displays a plot.
    """

    if len(comp_values) == 0:
        for x in label_value_list:
            comp_values.append(("", np.nan))

    categories = [x[0] for x in label_value_list]
    values = [mean(x[1]) for x in label_value_list]
    standard_deviations = [np.std(x[1]) for x in label_value_list]
    comp_mean_values = [mean(x[1]) for x in comp_values]
    comp_std_values = [np.std(x[1]) for x in comp_values]

    # Create figure and axis
    if axis == 0:
        # Adjust figure size to ensure everything fits well
        fig, ax = plt.subplots(figsize=(3, 4))
    else:
        ax = axis

    # Plotting the line diagram
    ax.plot(categories, values, marker='o', linestyle='-',
            color="green")  # Line with markers
    ax.plot(categories, comp_mean_values,
            marker='o', linestyle='-')  # comp values
    # ax.errorbar(categories, values, yerr=standard_deviations, fmt='o', capsize=7)

    # Setting labels and title
    if title != "":
        ax.set_title(f'{title}', fontsize=28)
    ax.set_ylabel(f"{y_label}")
    if axis == 0:
        ax.set_xlabel(f"{x_label}")
    # Ensure ticks are set correctly before setting labels
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)  # Rotate labels to prevent overlap

    # Modify these values to compress or expand the x-axis
    ax.set_xlim(-0.5, len(categories)-0.5)

    # Adjust layout to make sure nothing gets cut off
    if axis == 0:
        plt.tight_layout()
        plt.show()


def plot_occurrence_frequencies(occurrence_frequencies, parameter_label="", title="", comp_frequencies=[], axis=0, save=False):
    """
        Plots occurrence frequencies with error bars representing standard deviations, including optional comparison frequencies.

        Parameters:
            occurrence_frequencies (list of tuples): A list where each tuple contains a category label and a list of frequency values
                                                      (e.g., [("label1", [value1, value2, ...]), ...]).
            parameter_label (str, optional): The label for the x-axis. Defaults to an empty string.
            title (str, optional): The title of the plot. Defaults to an empty string.
            comp_frequencies (list of tuples, optional): A list of comparison frequencies in the same format as occurrence_frequencies.
                                                          Defaults to an empty list.
            axis (matplotlib.axes.Axes, optional): The axis to plot on. If 0, a new figure is created. Defaults to 0.

        Returns:
            None: This function does not return a value; it directly creates and displays a plot.
    """

    if len(comp_frequencies) == 0:
        for x in occurrence_frequencies:
            comp_frequencies.append(("", np.nan))

    categories = [x[0] for x in occurrence_frequencies]
    means = [np.mean(x[1]) for x in occurrence_frequencies]
    std_devs = [np.std(x[1]) for x in occurrence_frequencies]

    comp_means = [np.mean(x[1]) for x in comp_frequencies]
    comp_std_devs = [np.std(x[1]) for x in comp_frequencies]

    # Create figure and axis
    if axis == 0:
        # Adjust figure size to ensure everything fits well
        fig, ax = plt.subplots(figsize=(3, 4))
    else:
        ax = axis

    # Plotting the error bars
    ax.errorbar(categories, means, yerr=std_devs, fmt='o', capsize=7)
    ax.errorbar(categories, comp_means, yerr=comp_std_devs,
                fmt='o', capsize=7, color='green')
    # ax.errorbar(categories, comp_means, fmt='o', color='green')

    # Setting labels and title
    if title != "":
        ax.set_title(f'{title}', fontsize=28)
    ax.set_ylabel('Frequency of Occurrence (Hz)')
    if axis == 0:
        ax.set_xlabel(f"{parameter_label}")
    # Ensure ticks are set correctly before setting labels
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)  # Rotate labels to prevent overlap

    # Modify these values to compress or expand the x-axis
    ax.set_xlim(-0.5, len(categories)-0.5)

    # Adjust layout to make sure nothing gets cut off
    if axis == 0:
        plt.tight_layout()
        plt.show()


def plot_frequency_distribution(frequencies, label, save=False, concatenate=False):
    """
        Plots the frequency distribution of given frequencies using a histogram with specified bins.

        Parameters:
            frequencies (array-like): A list or array of frequency values to be plotted.
            label (str): The label for the title of the plot, indicating the context of the frequencies.

        Returns:
            None: This function does not return a value; it directly creates and displays a histogram plot.
    """
    # Define the bins for 20 Hz intervals from 20 Hz to 300 Hz
    bins = np.arange(20, 251, 10)

    # Count the occurrences of frequencies in each bin
    counts, _ = np.histogram(frequencies, bins=bins)

    # Create bar positions (the center of each bin)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Create a color array for the bars
    colors = ['green' if (100 <= center <= 250)
              else 'grey' for center in bin_centers]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.bar(bin_centers, counts, width=7.5, color=colors, edgecolor='black')

    # Setting the labels and title
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Peak Frequency Occurrences')
    plt.title(f'Event Peak Frequencies in: {label}')
    plt.xticks(bins)  # Set x-ticks to be at the edges of the bins
    # plt.ylim(0, 5)  # Set y-axis limit from 0 to 5

    # Show grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    if save:
        path = 'images'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'images/{label}_Frequency_Distribution.png')
    elif concatenate:
        pass
    else:
        plt.show()


def plot_peak_frequencies(peak_frequencies, parameter_label="", title="", comp_frequencies=[],
                          axis=0, figsize=(3, 4), xrotation=0, ylim=(0, 200), save=False):
    """
        Plots peak frequencies with error bars representing standard deviations, including optional comparison frequencies.

        Parameters:
            peak_frequencies (list of tuples): A list where each tuple contains a category label and a list of peak frequency values
                                                (e.g., [("label1", [value1, value2, ...]), ...]).
            parameter_label (str, optional): The label for the x-axis. Defaults to an empty string.
            title (str, optional): The title of the plot. Defaults to an empty string.
            comp_frequencies (list of tuples, optional): A list of comparison frequencies in the same format as peak_frequencies.
                                                          Defaults to an empty list.
            axis (matplotlib.axes.Axes, optional): The axis to plot on. If 0, a new figure is created. Defaults to 0.

        Returns:
            None: This function does not return a value; it directly creates and displays a plot.
    """

    if len(comp_frequencies) == 0:
        for x in peak_frequencies:
            comp_frequencies.append(("", np.nan))

    categories = [x[0] for x in peak_frequencies]
    means = [np.mean(x[1]) for x in peak_frequencies]
    std_devs = [np.std(x[1]) for x in peak_frequencies]
    comp_means = [np.mean(x[1]) for x in comp_frequencies]
    comp_std_devs = [np.std(x[1]) for x in comp_frequencies]

    # Create figure and axis
    if axis == 0:
        # or (4, 4) Adjust figure size to ensure everything fits well
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = axis

    # Plotting the error bars
    ax.errorbar(categories, means, yerr=std_devs, fmt='o', capsize=7)
    # ax.errorbar(categories, comp_means, yerr=comp_std_devs, fmt='o', capsize=7, color='green')
    ax.errorbar(categories, comp_means, fmt='o-', color='green')

    # Setting labels and title
    if title != "":
        ax.set_title(f'{title}', fontsize=28)
    ax.set_ylabel('Peak Frequency (Hz)')
    if axis == 0:
        ax.set_xlabel(f"{parameter_label}")
    # Ensure ticks are set correctly before setting labels
    ax.set_xticks(range(len(categories)))
    # Rotate labels to prevent overlap
    ax.set_xticklabels(categories, rotation=xrotation)

    # Modify these values to compress or expand the x-axis
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.set_yticks(np.arange(0, 176, 25))
    ax.set_ylim(ylim[0], ylim[1])

    # Adjust layout to make sure nothing gets cut off
    if save:
        path = 'images'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'images/{parameter_label}_Peak_Frequencies.png')
    else:
        plt.show()


def plot_power_spectral_density_bands(psd_bands, label="", title="", axis=0, figsize=(4, 4), xrotation=0, save=True):
    """
        Plots the power spectral density (PSD) for different frequency bands (Theta, Gamma, Ripple) with error bars.

        Parameters:
            psd_bands (list of tuples): A list where each tuple contains a parameter label and a list of lists representing
                                         the power values for different bands (e.g., [("label1", [[theta_band], [gamma_band], [ripple_band]]), ...]).
            label (str, optional): The label for the x-axis. Defaults to an empty string.
            title (str, optional): The title of the plot. Defaults to an empty string.
            axis (matplotlib.axes.Axes, optional): The axis to plot on. If 0, a new figure is created. Defaults to 0.

        Returns:
            None: This function does not return a value; it directly creates and displays a bar plot with error bars.
    """

    categories = [x[0] for x in psd_bands]
    # Extract means and standard deviations for Theta (0), Gamma (1), Ripple (2) bands
    theta_means = [np.mean(x[1][0]) for x in psd_bands]
    theta_stds = [np.std(x[1][0]) for x in psd_bands]

    gamma_means = [np.mean(x[1][1]) for x in psd_bands]
    gamma_stds = [np.std(x[1][1]) for x in psd_bands]

    ripple_means = [np.mean(x[1][2]) for x in psd_bands]
    ripple_stds = [np.std(x[1][2]) for x in psd_bands]

    # Number of groups
    n_groups = len(categories)
    if axis == 0:
        fig, ax = plt.subplots(figsize=figsize)  # for 1: (3, 4)
    else:
        ax = axis

    # Set position of bar on X axis
    index = np.arange(n_groups)
    bar_width = 0.16  # for 1: 0.075
    cap_size = 3

    rects1 = ax.bar(index - bar_width, ripple_means, bar_width, yerr=ripple_stds,
                    color='orange', label='Ripple band', capsize=cap_size,
                    error_kw={'zorder': 2}, zorder=3)

    rects2 = ax.bar(index, gamma_means, bar_width, yerr=gamma_stds,
                    color='green', label='Gamma band', capsize=cap_size,
                    error_kw={'zorder': 2}, zorder=3)

    rects3 = ax.bar(index + bar_width, theta_means, bar_width, yerr=theta_stds,
                    color='blue', label='Theta band', capsize=cap_size,
                    error_kw={'zorder': 2}, zorder=3)

    # Axis labels etc.
    ax.set_ylabel('Power ($V^2$)')

    if title != "":
        ax.set_title(f'{title}', fontsize=28)
    if axis == 0:
        ax.set_xlabel(f"{label}")
    # Ensure ticks are set correctly before setting labels
    ax.set_xticks(index)
    # Rotate labels to prevent overlap
    ax.set_xticklabels(categories, rotation=xrotation)
    ax.legend(loc='upper right', fontsize=9, labelspacing=0.15)

    # Set Y-axis to log scale to match the original plot scale
    ax.set_yscale('log')

    # Set limits and ticks on y-axis to match original plot
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.set_ylim(1e-18, 1e-8)  # Adjust as necessary based on your data range
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    if axis == 0:
        fig.tight_layout()
        if save:
            path = 'images'
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(f'images/{label}_PSD_Bands.png')
        else:
            plt.show()


def combine_plots(peak_input, occurrence_input, duration_input, power_input, parameter_label):
    """
        Combines multiple plots into a single figure with subplots arranged in a grid layout.

        Parameters:
            peak_input (tuple): A tuple containing peak frequency data and optional comparison frequencies.
            occurrence_input (tuple): A tuple containing occurrence frequency data and optional comparison frequencies.
            duration_input (list): A list of mean sharp wave ripple durations to be plotted.
            power_input (list): A list of power spectral density bands to be plotted.
            parameter_label (str): The label to be displayed below the combined plots, indicating the parameter being analyzed.

        Returns:
            None: This function does not return a value; it directly creates and displays a combined plot with multiple subplots.
    """
    # Create a figure with subplots (1 row and 4 columns)
    fig = plt.figure(figsize=(15, 5))  # 12,4 way better readability
    gs = GridSpec(1, 13)  # Total of width units (3+3+3+4=13)

    # Define widths based on the ratio
    ax1 = fig.add_subplot(gs[0:1, :3])  # Plot 1 takes up the first three units
    ax2 = fig.add_subplot(gs[0:1, 3:6])  # Plot 2 takes up the next three units
    ax3 = fig.add_subplot(gs[0:1, 6:9])  # Plot 3 takes up the next three units
    ax4 = fig.add_subplot(gs[0:1, 9:13])  # Plot 4 takes up the last four units

    # Call each plotting function with the corresponding axis
    plot_peak_frequencies(
        peak_input[0], comp_frequencies=peak_input[1], title="A", axis=ax1)
    plot_occurrence_frequencies(
        occurrence_input[0], comp_frequencies=occurrence_input[1], title="B", axis=ax2)
    plot_line_diagram(
        duration_input, "Mean SWR Duration (ms)", title="C", axis=ax3)
    plot_power_spectral_density_bands(power_input, title="D", axis=ax4)

    fig.text(0.5, 0.05, parameter_label, ha='center', va='center', fontsize=16)
    # Adjust layout
    plt.tight_layout(rect=[0.0, 0.075, 1.0, 1.0])

    # Show the combined plot
    plt.show()


def single_sim_analysis(file_path, showLFP, showEventLFP):
    """
        Performs a single simulation analysis by extracting data from a specified file, detecting events, and generating various plots.

        Parameters:
            file_path (str): The path to the file containing the simulation data.
            showLFP (bool): A flag indicating whether to plot the local field potential (LFP) recording.
            showEventLFP (bool): A flag indicating whether to plot individual sharp wave ripple events and non-ripple events.

        Returns:
            None: This function does not return a value; it directly creates and displays multiple plots based on the analysis.
    """
    sim_label, research_param = file_path.split("/")[-2:]

    # extract and analyse data
    recordings = create_list_from_timeSeries(file_path)
    [events, filtered_events, all_spectrum_peaks, all_duration], [sharp_wave_ripples,
                                                                  sharp_wave_ripple_peaks, sharp_wave_ripple_durations], band_spectra = event_detection(recordings)

    # prepare plotting parameters
    if len(recordings) > 40960:
        # 10s recording, starting at 20s
        lfp_recording_sample = recordings[20480:30720]
    elif len(recordings) < 6144:
        lfp_recording_sample = recordings  # if shorter than 6s, plot all
    else:
        # 5s recording, starting at 1s
        lfp_recording_sample = recordings[1024:6144]

    # generate plots
    if showLFP:
        plot_lfp(recordings, f"{sim_label} = {research_param}", save=True)

    if showEventLFP:
        for event in sharp_wave_ripples[:2]:
            plot_lfp(event, "Sharp wave ripple", save=True)
        for x in [x for x in events if x not in sharp_wave_ripples][:2]:
            plot_lfp(x, "No SWR", save=True)

    plot_frequency_distribution(all_spectrum_peaks, sim_label, save=True)
    plot_peak_frequencies(
        [(research_param, all_spectrum_peaks),], sim_label, save=True)
    plot_power_spectral_density_bands(
        [(research_param, band_spectra),], sim_label, save=True)


def sim_collection_analysis(collection_folder_path, chat_output, do_plots):
    """
        Facilitates the analysis of a collection of simulations with the same configurations, by aggregating data from multiple files and generating summary statistics.

        Parameters:
            collection_folder_path (str): The path to a folder containing only simulation data files of same parameter configuration.
            chat_output (bool): A flag indicating whether to print summary statistics to the console.
            do_plots (bool): A flag indicating whether to generate and display plots based on the aggregated data.

        Returns:
            tuple: A tuple containing:
                - list: Aggregated data including event counts, occurrence frequencies, peak frequencies, and durations.
                - list: Aggregated data for sharp wave ripples including counts, occurrence frequencies, peak frequencies, and durations.
                - list: A list of band powers for theta, gamma, and ripple frequency bands.
    """

    research_parameter, parameter_value = collection_folder_path.split(
        "/")[-2:]
    parameter_value = parameter_value.lstrip("0")
    sim_label = f'{research_parameter} = {parameter_value}'

    all_num = []
    all_occ_freq = []
    all_peaks = []
    all_dur = []

    swr_num = []
    swr_occ_freq = []
    swr_peaks = []
    swr_dur = []

    band_powers = [[], [], []]  # theta, gamma, ripple

    # aggregate data from collection
    for entity in os.listdir(collection_folder_path)[:8]:

        file_path = f'{collection_folder_path}/{entity}'
        recordings = create_list_from_timeSeries(file_path)
        [events, filtered_events, all_spectrum_peaks, all_duration], [sharp_wave_ripples,
                                                                      sharp_wave_ripple_peaks, sharp_wave_ripple_durations], band_spectra = event_detection(recordings)

        num_all = len(events)
        all_num.append(num_all)
        all_occ_freq.append(num_all/60)
        all_peaks.extend(all_spectrum_peaks)
        all_dur.extend(all_duration)

        num_swrs = len(sharp_wave_ripples)
        swr_num.append(num_swrs)
        swr_occ_freq.append(num_swrs/60)
        swr_peaks.extend(sharp_wave_ripple_peaks)
        swr_dur.extend(sharp_wave_ripple_durations)

        for i, x in enumerate(band_spectra):
            band_powers[i].extend(x)

    # display data
    if chat_output:
        print(f'\n___ {sim_label} ___\n')
        print(f"Average Events per minute : {mean(all_num)}")
        print(f"Average peak frequency : {mean(all_peaks)}")
        print(f"Average Event duration : {mean(all_dur)} ms")
        print("--------------------------------------------")
        print(f"Average Sharp Wave Ripples per minute : {mean(swr_num)}")
        print(f"Average Sharp Wave Ripple peak frequency : {mean(swr_peaks)}")
        print(f"Average Sharp Wave Ripple duration : {mean(swr_dur)} ms")

    if do_plots:
        plot_frequency_distribution(all_peaks, sim_label)

    all_data = [all_num, all_occ_freq, all_peaks, all_dur]
    swr_data = [swr_num, swr_occ_freq, swr_peaks, swr_dur]

    return all_data, swr_data, band_powers


def parameter_comparison(main_folder_path, reverse_analysis, do_chat, do_plots):
    """
        Compares simulation parameters by analyzing multiple simulation configurations and generating comparative plots.

        Parameters:
            main_folder_path (str): The path to the main folder containing parameter-specific subfolders.
            reverse_analysis (bool): A flag indicating whether to sort parameter values in reverse order.
            do_chat (bool): A flag indicating whether to print summary statistics to the console.
            do_plots (bool): A flag indicating whether to generate and display plots based on the analysis.

        Returns:
            None: This function does not return a value; it directly creates and displays comparative plots of simulation results.
    """
    parameter_label = main_folder_path.split("/")[-1].split("(")[0]

    parm_units = {"gCAN": "($µS/cm^2$)", "G_ACh": "(factor)", "maxN": "(count)", "g_max_e": "($pS$)", "gCAN-G_ACh": "($µS/cm^2$ - factor)",
                  "maxN-g_max_e": "(count - $pS$)", "Full Attack": "(intensity)"}
    parameter_with_unit = f"{parameter_label} {parm_units[parameter_label]}"

    all_peak_lists = []
    all_occ_freq_lists = []
    swr_peak_lists = []
    swr_occ_freq_lists = []
    swr_dur_lists = []
    band_power_lists = []

    [all_num, all_occ_freq, all_peaks, all_dur], [swr_num, swr_occ_freq, swr_peaks, swr_dur], band_powers = sim_collection_analysis(
        "sorted_output/sleep/healthy", do_chat, do_plots)
    all_peak_lists.append(("healthy", all_peaks))
    all_occ_freq_lists.append(("healthy", all_occ_freq))
    swr_peak_lists.append(("healthy", swr_peaks))
    swr_occ_freq_lists.append(("healthy", swr_occ_freq))
    swr_dur_lists.append(("healthy", swr_dur))
    band_power_lists.append(("healthy", band_powers))

    parameter_values = sorted(os.listdir(
        main_folder_path), reverse=reverse_analysis)

    for parameter in parameter_values:
        parameter_folder_path = f"{main_folder_path}/{parameter}"
        clean_param_string = parameter.lstrip("0").split("(")[0]

        [all_num, all_occ_freq, all_peaks, all_dur], [swr_num, swr_occ_freq, swr_peaks,
                                                      swr_dur], band_powers = sim_collection_analysis(parameter_folder_path, do_chat, do_plots)
        all_peak_lists.append((clean_param_string, all_peaks))
        all_occ_freq_lists.append((clean_param_string, all_occ_freq))
        swr_peak_lists.append((clean_param_string, swr_peaks))
        swr_occ_freq_lists.append((clean_param_string, swr_occ_freq))
        swr_dur_lists.append((clean_param_string, swr_dur))
        band_power_lists.append((clean_param_string, band_powers))

    # Plot All
    combine_plots((all_peak_lists, swr_peak_lists), (all_occ_freq_lists,
                  swr_occ_freq_lists), swr_dur_lists, band_power_lists, parameter_with_unit)

    # uncomment following lines to output specific plots on their own
    # Peak Frequencies
    # plot_peak_frequencies(all_peak_lists, parameter_with_unit, title="A", comp_frequencies=swr_peak_lists)
    # plot_peak_frequencies(all_peak_lists, parameter_label, title="All Events")
    # plot_peak_frequencies(swr_peak_lists, parameter_label, title="Sharp Wave Ripples")

    # Event Occurrence
    # plot_occurrence_frequencies(all_occ_freq_lists, parameter_with_unit, title="B", comp_frequencies=swr_occ_freq_lists)
    # plot_occurrence_frequencies(swr_occ_freq_lists, parameter_label, title="Sharp Wave Ripples")

    # Durations
    # plot_line_diagram(swr_dur_lists, "Mean SWR Duration (ms)", x_label=parameter_with_unit, title="C")

    # Power in Oscillation bands
    # plot_power_spectral_density_bands(band_power_lists, parameter_with_unit, title="D")

    # More
    # plot_line_diagram(swr_occ_freq_lists, parameter_label, "Occurrence Frequency (Hz)", 1, [mean(x[1]) for x in all_occ_freq_lists])


if __name__ == '__main__':

    doChat = 1
    doPlots = 1
    reversed_analysis = 0

    # Attack parameter analysis
    # parameter_comparison("sorted_output/sleep/gCAN", 0, 0, 0)
    # parameter_comparison("sorted_output/sleep/G_ACh", 0, 0, 0)
    # parameter_comparison("sorted_output/sleep/gCAN-G_ACh", 0, 0, 0)
    # parameter_comparison("sorted_output/sleep/g_max_e", 1, 0, 0)
    # parameter_comparison("sorted_output/sleep/maxN", 1, 0, 0)
    # parameter_comparison("sorted_output/sleep/maxN-g_max_e", 1, 0, 0)
    # parameter_comparison("sorted_output/sleep/Full Attack(maxN-g_max_e-gCAN-G_ACh)", reversed_analysis, doChat, doPlots)

    single_sim_analysis("sorted_output/sleep/healthy/LFP_08-11_[0].txt", 1, 0)
