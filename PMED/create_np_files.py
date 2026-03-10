import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from config import window_secs, sensor_names, baseline_temp, sampling_rate, num_repetitions, raw_dir, np_dir
from read_data import np_pmhdb_exists

#-------------------------------------------------------------------------------------------------------
# Functions PainMonit Experimental Dataset (PMED)
#-------------------------------------------------------------------------------------------------------

def to_categorical(y, num_classes=None, dtype="float32"):
    """Function to convert a label vector to the one-hot encoding format.

    Parameters
    ----------
    y: numpy/list. Vector of labels assumed to be in integer format.
    num_classes: Int. Number of classes. If set to None, the number of classes is automatically set to max(y)+1

    Returns
    -------
    np: categorical. Matrix of labels in the the one-hot encoding format.
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def crossings_nonzero_neg2pos(data):
    """Function to find zero crossings (from negative to positive) in a signal.

    Parameters
    ----------
    data: numpy/list. The given numpy to search for zero crossings.

    Returns
    -------
    np: Crossings. Indices where crossings from negative to positive can be found.
    """
    npos = data < 0
    return (npos[:-1] & ~npos[1:]).nonzero()[0]

def segment_pmhdb(df, baseline_shift= 5, plot= False):
    """Function to segment stim and baseline windows from synchronised PMED.

    Parameters
    ----------
    df: Panda. Synchronised PMED stream of one subject.
    baseline_shift: Int. Number of seconds by which the "10 seconds baseline"-window is shifted backwards. Value describes the time between "baseline" and "stim". Default is 5.
    plot: Bool. Boolean to describe whether the segmentation process is plotted or not.
    """

    X = []
    y_heater = []
    y_covas = []

    # start of a stimuli
    stim = (df["Heater_cleaned"] != baseline_temp).astype("int")
    stim[stim==False]=-1
    stim_starts = crossings_nonzero_neg2pos(stim.values)
    num_baseline_windows = 0

    if plot:
        df["Heater_cleaned"].plot()

    # extract 10 seconds before stim
    window = int(window_secs * sampling_rate)
    for start in stim_starts:
        # extract window before as baseline
        # move the baseline window about "baseline_shift" secs
        baseline_start = start - (baseline_shift * sampling_rate)
        if (num_baseline_windows < num_repetitions) and (baseline_start > window) and (df["Heater_cleaned"].values[baseline_start - window: baseline_start]==baseline_temp).all():
            X.append(df[sensor_names].values[baseline_start - window: baseline_start])
            y_covas.append(0)
            y_heater.append(0)
            num_baseline_windows += 1

            if plot:
                plt.axvspan(baseline_start - window, baseline_start, facecolor='#2ca02c', alpha=0.5, label= "Baseline")

        # extract window afterwards as stimulus
        start += 1
        temp = df["Heater_cleaned"].values[start]
        end = int(start + window)
        if (df["Heater_cleaned"].values[start: end]== temp).all():
            X.append(df[sensor_names].values[start: end])
            y_covas.append(sum(df["COVAS"].values[start: end]))
            y_heater.append(temp)

            if plot:
                plt.axvspan(start, start + window, facecolor='#d62728', alpha=0.5, label= "Stim")

    if plot:
        plt.show()

    # --- Convert heater from temp to class
    # extract baseline + 1 one non painful stim and 4 pain stim temperatures
    temps = np.unique(y_heater)

    conversion = { x : i for i, x in enumerate(temps)}
    # convert y from temperature to class label -> baseline: 0, no pain stimulus: 1, pain stimuli: 2-5
    y_heater = np.vectorize(conversion.get)(y_heater)

    # --- Convert COVAS label into quartiles 0:No covas; 1: 1.Quartile, 2: 2. Quartile, [...], 4: 4.Quartile
    # Normalize values between 0 and 100
    y_covas = np.array(y_covas)
    y_covas = y_covas/y_covas.max()
    y_covas *= 100
    y_covas = np.array([int(x//(25)) + 1 if x > 0 else 0 for x in y_covas])
    y_covas[y_covas==5] = 4

    X = np.array(X)

    return X, y_heater, y_covas

def create_np_pmhdb(overwrite= False, raw_dir= raw_dir, np_dir= np_dir):
    """Function to create np files of the PMED and save them.

    Parameters
    ----------
    overwrite: bool. Whether to overwrite existing np files or not. Defaults to False.
    raw_dir: Str/Path. Path to define the directory containing the raw files. Defaults to 'raw_dir'.
    np_dir: Str/Path. Path to define the output directory to create the NP files. Defaults to 'np_dir'.
    """

    if not Path(raw_dir).exists():
        print(f"There is no directory '{raw_dir.resolve()}'. Please place the datasets correctly.")
        return

    if np_pmhdb_exists(np_dir) and not overwrite:
        print(f"There is already a numpy dataset under '{np_dir.resolve()}'. Dataset will not be overwritten. Use the '--overwrite' flag to overwrite the dataset.")
        return

    data_list = []
    heater_list = []
    covas_list = []
    subjects_list = []

    print("Create PMED np files...")

    file_names = glob.glob(str(Path(raw_dir, "*.csv")))

    # sort the list - lexicographically
    file_names.sort()

    for index, filename in enumerate(tqdm(file_names)):
        subject_data = pd.read_csv(filename, sep=";", decimal=",")

        X, y_heater, y_covas = segment_pmhdb(subject_data)

        data_list.append(X)
        heater_list.append(y_heater)
        covas_list.append(y_covas)
        subjects_list.append([index] * X.shape[0])

    data = np.concatenate(data_list, axis=0)
    heater = np.concatenate(heater_list, axis=0)
    covas = np.concatenate(covas_list, axis=0)
    subjects = np.concatenate(subjects_list, axis=0)

    assert len(data)==len(heater)==len(covas)==len(subjects)

    data = np.nan_to_num(data,)

    # Data: Add channel axis
    data = data[..., np.newaxis]

    # Labels to categorical
    heater = to_categorical(heater)
    covas = to_categorical(covas)

    if not np_dir.exists():
        os.makedirs(np_dir)

    np.save(Path(np_dir, "X"), data)
    np.save(Path(np_dir, "y_heater"), heater)
    np.save(Path(np_dir, "y_covas"), covas)
    np.save(Path(np_dir, "subjects"), subjects)

    print("\nData shape: ", data.shape)
    print("heater shape: ", heater.shape)
    print("covas shape: ", covas.shape)
    print("Subjects shape: ", subjects.shape)

    print(f"Np dataset created and saved under '{np_dir.resolve()}'.")

#-------------------------------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """Main function.
    """

    # Set working directory to path of file
    os.chdir(Path(sys.path[0]))

    # Initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--overwrite", help="Overwrites existing NP files.", action="store_true")

    # Read arguments from the command line
    args = parser.parse_args()

    create_np_pmhdb(overwrite= args.overwrite)
