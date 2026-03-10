import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from config import raw_dir, np_dir, sensor_names, sampling_rate
from read_data import read_raw_data, np_pmpdb_exists

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

#-------------------------------------------------------------------------------------------------------
# Functions PainMonit Clinical Dataset (PMCD)
#-------------------------------------------------------------------------------------------------------

def segment_pmpdb(df, window_secs= 4, step_size= 2, sampling_rate= sampling_rate, plot= False):
    """Function to segment stimulus and baseline windows from synchronised 'PainMonit' data.

    Parameters
    ----------
    df: Panda. Synchronised PainMonit data of one subject.
    window_width: float. Width of the segmented windows in seconds. Defaults to 4.
    step_size: float. Step size given in seconds. Defaults to 2.
    sampling_rate: int. The sampling rate of the data. Defaults to the data sampling rate (250).
    plot: Bool. Boolean to describe whether the segmentation process is plotted or not. Defaults to False.

    Returns
    -------
    X: np. A numpy file containing the segmented data.
    """

    # calculate window size
    window_size = int(window_secs * sampling_rate)
    # calculate the distance between window start
    distance = step_size * sampling_rate
    # calculate the length of the input data
    input_data_length = df.shape[0]

    if plot:
        df.plot()
        colors = ["red", "green"]
        i = 0

    X = []
    for start in range(0, input_data_length, distance):
        end = start + window_size

        if end > input_data_length:
            continue

        X.append(df.values[start: end])

        if plot:
            plt.axvspan(df.index.values[start], df.index.values[end], facecolor=colors[i%len(colors)], alpha=0.5)
            i+=1

    if plot:
        plt.show()

    X = np.array(X)

    return X

def process_segments(x, columns, selected_sensors= sensor_names):
    """Function to process the segments of the PMCD dataset.
    Filters segments to the ones with pain label and extract only relevant data.

    Parameters
    ----------
    x: np. numpy file with the segmented data (from 'segment_pmpdb()').
    columns: list. Names of the columns.
    selected_sensors: list. List of sensors to select from the data. Defaults to 'sensor_names'.

    Returns
    ----------
    data, labels: np. Numpy files with the data and labels.
    """

    data = []
    labels = []

    for i in x:
        # all pain labels in the current segment
        pain_labels = i[:, columns.index("Pain labels")]

        # maximum value
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_label = np.nanmax(pain_labels)

        # skip if there are only nan values        
        if np.isnan(max_label):
            continue

        labels.append(int(max_label))
        data.append(i)

    labels = np.array(labels)
    data = np.array(data)

    # select only the sensor data
    data = np.stack([data[:, :, columns.index(i)] for i in selected_sensors], axis= -1)

    return data, labels

def create_np_pmpdb(overwrite= False, raw_dir= raw_dir, np_dir= np_dir):
    """Function to create np files of the PMCD dataset and save them.

    Parameters
    ----------
    overwrite: bool. Whether to overwrite existing np files or not. Defaults to False.
    raw_dir: Str/Path. Path to define the directory containing the raw files. Defaults to 'raw_dir'.
    np_dir: Str/Path. Path to define the output directory to create the NP files. Defaults to 'np_dir'.
    """

    if not Path(raw_dir).exists():
        print(f"There is no directory '{raw_dir.resolve()}'. Please place the datasets correctly.")
        return

    if np_pmpdb_exists(np_dir) and not overwrite:
        print(f"There is already a numpy dataset under '{np_dir.resolve()}'. Dataset will not be overwritten. Use the '--overwrite' flag to overwrite the dataset.")
        return

    data_list = []
    labels_list = []
    subjects_list = []

    print("Create PMCD np files...")

    for i in tqdm(range(1, 50)):

        subject_data = read_raw_data(subject_id= i)

        for repetition in range(2):
            if subject_data[repetition] is None:
                continue

            # select the correct data for subject and repetition
            x = subject_data[repetition]["data"]
            # save the column headers for later use
            columns = list(x.columns)
            # extract segments and save them in a numpy
            x = segment_pmpdb(x)

            # extract only needed sensors, filter segments with no pain labels
            data, labels = process_segments(x, columns= columns)

            # remove no pain data
            mask = labels != 0
            data = data[mask]
            labels = labels[mask]

            # extract the baseline windows
            x_baseline = segment_pmpdb(subject_data[repetition]["baseline"])
            # select only the sensor data
            data_baseline = np.stack([x_baseline[:, :, columns.index(i)] for i in sensor_names], axis= -1)
            labels_baseline = [0] * len(data_baseline)
            subjects_baseline = [i] * len(data_baseline)

            # add the pain data
            data_list.append(data)
            labels_list.append(labels)
            subjects_list.append([i] * len(data))

            # add the non pain data
            data_list.append(data_baseline)
            labels_list.append(labels_baseline)
            subjects_list.append(subjects_baseline)

    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    subjects = np.concatenate(subjects_list, axis=0)

    assert len(data)==len(labels)==len(subjects)

    data = np.nan_to_num(data,)
    labels = np.nan_to_num(labels,)

    # Data: Add channel axis
    data = data[..., np.newaxis]

    labels = to_categorical(labels)

    if not np_dir.exists():
        os.makedirs(np_dir)

    np.save(Path(np_dir, "X"), data)
    np.save(Path(np_dir, "y"), labels)
    np.save(Path(np_dir, "subjects"), subjects)

    print("\nData shape: ", data.shape)
    print("Labels shape: ", labels.shape)
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

    create_np_pmpdb(overwrite= args.overwrite)
