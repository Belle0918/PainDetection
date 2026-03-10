"""
Feature extraction: convert raw physiological signal windows into a tabular
(pandas DataFrame) representation suitable for classical ML and LLM analysis.

Input:  X of shape (N, T, C) or (N, T, C, 1)  — windows × time × sensors
Output: DataFrame of shape (N, num_features)    — one row per window
"""
import numpy as np
import pandas as pd
from scipy import stats, signal as sp_signal


def _band_power(sig: np.ndarray, fs: int, low: float, high: float) -> float:
    """Return the fraction of total power in [low, high] Hz band."""
    freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=min(len(sig), 256))
    idx = (freqs >= low) & (freqs <= high)
    total = np.trapz(psd, freqs)
    if total == 0:
        return 0.0
    return float(np.trapz(psd[idx], freqs[idx]) / total)


def _spectral_entropy(sig: np.ndarray, fs: int) -> float:
    _, psd = sp_signal.welch(sig, fs=fs, nperseg=min(len(sig), 256))
    psd_norm = psd / (psd.sum() + 1e-12)
    return float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))


def _dominant_freq(sig: np.ndarray, fs: int) -> float:
    freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=min(len(sig), 256))
    return float(freqs[np.argmax(psd)])


def extract_window_features(window: np.ndarray, fs: int = 250) -> dict:
    """Extract features from a single window of shape (T, C).

    Returns a flat dict: {<sensor>_<feature>: value, ...}
    """
    features = {}
    n_sensors = window.shape[1]
    for c in range(n_sensors):
        sig = window[:, c].astype(float)
        prefix = f"s{c}"

        # --- Statistical ---
        features[f"{prefix}_mean"]     = float(np.mean(sig))
        features[f"{prefix}_std"]      = float(np.std(sig))
        features[f"{prefix}_min"]      = float(np.min(sig))
        features[f"{prefix}_max"]      = float(np.max(sig))
        features[f"{prefix}_range"]    = float(np.max(sig) - np.min(sig))
        features[f"{prefix}_median"]   = float(np.median(sig))
        features[f"{prefix}_iqr"]      = float(stats.iqr(sig))
        features[f"{prefix}_skew"]     = float(stats.skew(sig))
        features[f"{prefix}_kurtosis"] = float(stats.kurtosis(sig))

        # --- Energy / temporal ---
        features[f"{prefix}_rms"]  = float(np.sqrt(np.mean(sig ** 2)))
        features[f"{prefix}_mad"]  = float(np.mean(np.abs(sig - np.mean(sig))))
        zcr = ((sig[:-1] * sig[1:]) < 0).sum()
        features[f"{prefix}_zcr"]  = float(zcr / len(sig))

        # --- Frequency ---
        features[f"{prefix}_dom_freq"]       = _dominant_freq(sig, fs)
        features[f"{prefix}_spec_entropy"]   = _spectral_entropy(sig, fs)
        features[f"{prefix}_band_low"]       = _band_power(sig, fs, 0.0,  5.0)
        features[f"{prefix}_band_high"]      = _band_power(sig, fs, 5.0, 40.0)

    return features


def extract_features(
    X: np.ndarray,
    sensor_names: list[str],
    fs: int = 250,
) -> pd.DataFrame:
    """Extract features from all windows.

    Parameters
    ----------
    X            : (N, T, C) or (N, T, C, 1) array
    sensor_names : list of C sensor name strings
    fs           : sampling frequency in Hz

    Returns
    -------
    DataFrame of shape (N, num_features) with columns named
    <sensor>_<feature>.
    """
    if X.ndim == 4:
        X = X[..., 0]  # drop channel dimension

    rows = []
    for i in range(len(X)):
        row = extract_window_features(X[i], fs=fs)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Rename columns from s0_, s1_, … to <sensor_name>_
    rename = {f"s{c}_": f"{sensor_names[c]}_" for c in range(len(sensor_names))}
    for old_prefix, new_prefix in rename.items():
        df.columns = [col.replace(old_prefix, new_prefix, 1) for col in df.columns]

    return df
