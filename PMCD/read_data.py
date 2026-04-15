import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from config import np_dir

def np_pmpdb_exists(np_dir= np_dir):
    """Function to check if np files of the PMCD dataset already exist.

    Parameters
    ----------
    np_dir: string. String describing the location of the files.

    Returns
    -------
    bool: True if the dataset exists as np file, False otherwise.
    """

    data = Path(np_dir, "X.npy")
    labels = Path(np_dir, "y.npy")
    subjects = Path(np_dir, "subjects.npy")

    return data.exists() and labels.exists() and subjects.exists()

def read_segmented_np():
    ''' Function to read the segmented PMED in form of numpy files created by script "create_np_files".
    Parameters
    ----------

    Returns
    -------
    X, y, subjects: np.
    '''

    if not np_pmpdb_exists():
        raise FileExistsError("Data has not been segmented before. Please run the 'create_np_files.py' script.")
    
    data = np.load(Path(np_dir, "X.npy"))
    labels = np.load(Path(np_dir, "y.npy"))
    subjects = np.load(Path(np_dir, "subjects.npy"))

    return data, labels, subjects

def read_txt(file_path):
    if not file_path.exists():
        raise FileExistsError(f"File '{file_path}' does not exists.")
    
    with open(file_path, "r") as f:
        return f.read()

def read_raw_data(subject_id):
    ''' Function to read the data streams from one subject.
    Parameters
    ----------

    subject_id: int. Subject code.

    Returns
    -------
    subject data: list.
    '''

    data = []

    for i in range(2):
        name = f"P{str(subject_id).zfill(2)}_{i+1}"
        file_name = f"{name}.csv"
        file_dir = Path("dataset", "raw-data", name)
        file_path = Path(file_dir, file_name)

        if not file_path.exists():
            data.append(None)
            continue

        df = pd.read_csv(file_path, sep=";", decimal=",")
        df = set_index(df)

        df_baseline = pd.read_csv(Path(file_dir, f"{name}_runUp.csv"), sep=";", decimal=",")
        df_baseline = set_index(df_baseline)

        noPainThreshold = int(read_txt(Path(file_dir, "noPainThreshold.txt")))
        severePainThreshold = int(read_txt(Path(file_dir, "severePainThreshold.txt")))

        data.append({"data": df, "baseline": df_baseline, "noPainThreshold": noPainThreshold, "severePainThreshold": severePainThreshold})

    return data

def set_index(df):
    ''' Function to correctly set the index of the PMCD data frames.
    Parameters
    ----------

    df: pandas. Pandas data frame to manipulate.

    Returns
    -------
    df: pandas.
    '''

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
    print("")
    print("_"*50)
    print("Segmented data")
    X, y, subjects = read_segmented_np()
    print("Shape of the segmented data:")
    print("Data shape: ", X.shape)
    print("Labels shape: ", y.shape)
    print("Subjects shape: ", subjects.shape)
    print("Sample count per class:")
    y = np.argmax(y, axis= 1)
    values, counts = np.unique(y, return_counts=True)
    for value, count in zip(values, counts):
        print(f"Class '{value}': {count}")

    # Read in the data stream of one subject
    print("_"*50)
    print("Data streams")
    subject_id = 1
    data = read_raw_data(subject_id= subject_id)
    df = data[0]["data"]
    print(f"Shape of the first data stream from subject '{subject_id}':")
    print(df.shape)
    # Plot data
    print("Trying to plot data stream...")
    axes = df.drop(columns=['Pain rates', 'Pain labels']).resample(rule= "250L").mean().plot(subplots=True)
    plt.legend()
    plt.show()