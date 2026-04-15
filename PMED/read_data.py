import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from config import np_dir

def np_pmhdb_exists(np_dir= np_dir):
    """Function to check if np files of the PMED dataset already exist.

    Parameters
    ----------
    np_dir: string. String describing the location of the files.

    Returns
    -------
    bool: True if the dataset exists as np file, False otherwise.
    """

    data = Path(np_dir, "X.npy")
    heater = Path(np_dir, "y_heater.npy")
    covas = Path(np_dir, "y_covas.npy")
    subjects = Path(np_dir, "subjects.npy")

    return data.exists() and heater.exists() and covas.exists() and subjects.exists()

def read_segmented_np(label):
    ''' Function to read the segmented PMED in form of numpy files created by script "create_np_files".
    Parameters
    ----------

    label: String. Either 'covas' or 'heater' depending on the label you want to use.

    Returns
    -------
    X, y, subjects: np.
    '''

    if not np_pmhdb_exists():
        raise FileExistsError("Data has not been segmented before. Please run the 'create_np_files.py' script.")

    np_dir = Path("dataset", "np-dataset")
    return np.load(Path(np_dir, "X.npy")), np.load(Path(np_dir, "y_{}.npy".format(label))), np.load(Path(np_dir, "subjects.npy"))

def read_synchronised_data(subject_id):
    ''' Function to read the data streams from one subject found under "raw-data".
    Parameters
    ----------

    subject_id: int. Subject code.

    Returns
    -------
    subject data: data frame.
    '''
    
    filename = Path("dataset", "raw-data", f"S_{str(subject_id).zfill(2)}-synchronised-data.csv")
    df = pd.read_csv(filename, sep=";", decimal=",")
    df = df.set_index("Seconds")
    df.index = pd.to_timedelta(df.index, unit='s')
    df.index.name = "Secs"

    return df
    
if __name__ == "__main__":
    """Main function.
    """

    # Set working directory to path of file
    os.chdir(Path(sys.path[0]))

    # Read in the segmented data
    print("_"*50)
    print("Segmented data")
    X, y, subjects = read_segmented_np(label= "heater")
    print("Shape of the segmented data:")
    print(X.shape)
    print("(Samples, Secs * Hz, Num sensors, 1)")
    print("Sensors order: 'Bvp', 'Eda_E4', 'Resp', 'Eda_RB', 'Ecg', 'Emg'")

    # Read in the data stream of one subject
    print("_"*50)
    print("Data streams")
    subject_id = 1
    df = read_synchronised_data(subject_id= subject_id)
    print(f"Shape of the data stream from subject '{subject_id}':")
    print(df.shape)
    print("Columns of the data stream:")
    print(list(df.columns))
    # Plot data
    print("Trying to plot data stream...")
    axes = df.resample(rule= "250L").mean().plot(subplots=True)
    plt.legend()
    plt.show()