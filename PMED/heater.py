import os
import sys
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt

from config import sampling_rate, baseline_temp, window_secs
from read_data import read_synchronised_data

def round_temp(number):
    """Round a number to the closest half integer.
    >>> round_temp(1.3)
    1.5
    >>> round_temp(2.6)
    2.5
    >>> round_temp(3.0)
    3.0
    >>> round_temp(4.1)
    4.0"""
    return np.round(number * 2) / 2

def clean_heater_signal(y, plot= False, method= "temperature_raise", temp_threshold = 36, raise_threshold= 3):
    """Function to clean the heater signal of the Medoc software.

    Parameters
    ----------
    y: Np. Signal to clean.
    plot: Bool. Whether to plot the preprocessing process or not. Plt.show() is not called.
    method: String. Method to detect the flanks.
                    Method `temperature_threshold` detects flanks when the temperature exceeds the given `temp_threshold` temperature (C°).
                    Method `temperature_raise` detects flanks where the temperature raises at least `temp_threshold` temperature (C°) at a stretch.
    temp_threshold: Int. Temperature (C°) threshold to detect a stimulus. Used for method `temperature_threshold`. Default is 36.
    raise_threshold: Int. Temperature (C°) threshold to detect a stimulus. Used for method `temperature_raise`. Default is 3.

    Returns
    -------
    Np: Preprocessed signal
    """
    if plot:
        plt.plot(y, label= "Raw")

    if method == "temperature_threshold":
        def crossings_nonzero_pos2neg(data):
            pos = data > 0
            return (pos[:-1] & ~pos[1:]).nonzero()[0]

        def crossings_nonzero_neg2pos(data):
            npos = data < 0
            return (npos[:-1] & ~npos[1:]).nonzero()[0]

        # retrieve flanks where the signal exceeds or falls of 'temp_threshold' C° -> painful stimulus
        start_flanks = crossings_nonzero_neg2pos(y - temp_threshold)
        end_flanks = crossings_nonzero_pos2neg(y - temp_threshold)

        # Remove first window if starting flank is missing
        if start_flanks[0]>end_flanks[0]:
            end_flanks = end_flanks[1:]
        # Remove last window if ending flank is missing
        if end_flanks[-1]<start_flanks[-1]:
            start_flanks = start_flanks[:-1]

        # remove wrong flanks - remove if there are too many flanks in a certain time window. - Temp values are fluctuating a lot.
        start_flanks = [start_flanks[0]] + [start_flanks[i] for i in np.arange(1, len(start_flanks)) if abs(start_flanks[i]-start_flanks[i-1]) > (sampling_rate * window_secs)]
        end_flanks = [end_flanks[0]] + [end_flanks[i] for i in np.arange(1, len(end_flanks)) if abs(end_flanks[i]-end_flanks[i-1]) > (sampling_rate * window_secs)]

    elif method == "temperature_raise":
        # preprocess the signal - round, smooth (convolve), pad missing values because of convolve
        y = np.round(y, decimals= 2)
        kernel_size = 100
        kernel = np.ones(kernel_size) / kernel_size
        y = np.pad(y, (kernel_size//2, kernel_size//2 - 1), 'edge')
        y = np.convolve(y, kernel, 'valid')

        if plot:
            plt.plot(y, label= "Rounded")

        def zero_crossing(data):
            pos = data > 0
            return ((pos[:-1] & ~pos[1:]) | (~pos[:-1] & pos[1:])).nonzero()[0]

        # compute diff between successive values
        deriv = y[:-1] - y[1:]

        # replace zeros with previous value
        for i in range(1, len(deriv)):
            deriv[i] = deriv[i] if deriv[i] != 0 else deriv[i-1]

        # possible flanks are where there is a change in diffs
        possible_flanks = zero_crossing(deriv)

        # insert a possible flank at the start - when the signal is raising from the beginning there is no diff flank
        possible_flanks = np.insert(possible_flanks, 0, 0, axis=0)

        # filter flanks for the ones were there is an increase over `threshold` C°
        start_flanks = [j for i, j in zip(possible_flanks[:-1], possible_flanks[1:]) if (y[i] - y[j]) < - raise_threshold]
        # filter flanks for the ones were there is an decrease over `threshold` C°
        end_flanks = [i for i, j in zip(possible_flanks[:-1], possible_flanks[1:]) if (y[i] - y[j]) > raise_threshold]

    else:
        raise NotImplementedError(f"Method '{method}' is not implemented.")

    if plot:
        plt.vlines(x=start_flanks, ymin= min(y), ymax= max(y), color= "green", linestyle='--', label= "Starts")
        plt.vlines(x=end_flanks, ymin= min(y), ymax= max(y), color= "red", linestyle='--', label= "Ends")

    old_end = 0

    # if the first segment is missing its starting flank -> remove it
    if end_flanks[0] < start_flanks[0]:
        end_flanks = end_flanks[1:]

    # for each found start and end flanks
    for start_flank, end_flank in zip(start_flanks, end_flanks):

        # create a offset to extract the middle values of a window. On the corners values are fluctuating more.
        offset = 2 * sampling_rate
        # calculate stim temperature
        temp = round_temp(np.mean(y[start_flank + offset: end_flank - offset]))

        # calculate middle
        mid = start_flank + ((end_flank - start_flank)//2)

        # calculate start and end of window. 10 seconds window centered around middle of stim.
        start, end = mid - ((window_secs*sampling_rate)//2), mid + ((window_secs*sampling_rate)//2)

        # set the values before the window as baseline
        y[old_end:start]= baseline_temp
        y[start:end]= temp
        old_end= end

    # set everything after the last stim to baseline
    y[old_end:]= baseline_temp

    if plot:
        plt.plot(y, label= "Cleaned")
        plt.legend()

    return y


if __name__ == "__main__":
    """Main function.
    """

    # Set working directory to path of file
    os.chdir(Path(sys.path[0]))

    # Read in the data stream of one subject
    df = read_synchronised_data(subject_id= 1)
    cleaned_signal = clean_heater_signal(df["Heater [C]"])
    print("Unique temperatures: ", np.unique(cleaned_signal))
