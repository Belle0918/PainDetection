"""
Dataset loading helpers that abstract over PMED and PMCD numpy files.

Label schemes
─────────────
PMED (y_heater, 6 classes)
  full    : 0=baseline, 1=NP, 2=P1, 3=P2, 4=P3, 5=P4
  3class  : 0=no-pain {0,1}, 1=lower-pain {2,3}, 2=higher-pain {4,5}
            NOTE: P1-P4 are evenly spaced between pain threshold (T_P) and tolerance
            (T_T). The {2,3}/{4,5} split is a pragmatic 2+2 grouping, NOT a
            clinically validated moderate/severe boundary. For severity grounded in
            subjective experience, prefer label='covas' instead.
  binary  : 0=no-pain {0,1}, 1=pain {2,3,4,5}

PMED (y_covas, 5 classes) — subjective CoVAS rating (quartiles of 0-100 scale)
  full    : 0=no-rating, 1=Q1(0-25%), 2=Q2(25-50%), 3=Q3(50-75%), 4=Q4(75-100%)
  3class  : 0=no-pain {0}, 1=moderate {1,2}, 2=severe {3,4}
            This is the more defensible severity split — it uses the subject's
            own pain report rather than the stimulus temperature level.
  binary  : 0=no-pain {0}, 1=pain {1,2,3,4}

PMCD (y, 3 classes) — per-subject NRS thresholds applied by dataset authors
  full / 3class : 0=no-pain, 1=moderate, 2=severe   (already 3-level)
  binary        : 0=no-pain {0}, 1=pain {1,2}

The 3-class scheme is the natural common ground for cross-domain experiments.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

from .config import PMED_NP_DIR, PMCD_NP_DIR, PMED_SENSORS, PMCD_SENSORS


# ── Mapping tables ────────────────────────────────────────────────────────────

# PMED heater (6 → N classes)
_PMED_HEATER_FULL   = {i: i for i in range(6)}                      # identity
_PMED_HEATER_3CLASS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
_PMED_HEATER_BINARY = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1}

# PMED covas (5 → N classes)
_PMED_COVAS_FULL    = {i: i for i in range(5)}                       # identity
_PMED_COVAS_3CLASS  = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
_PMED_COVAS_BINARY  = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}

# PMCD (3 → N classes) — already 3-level (no-pain / moderate / severe)
_PMCD_FULL          = {i: i for i in range(3)}                       # identity
_PMCD_3CLASS        = {0: 0, 1: 1, 2: 2}                            # identity
_PMCD_BINARY        = {0: 0, 1: 1, 2: 1}

# Human-readable class names
PMED_HEATER_NAMES = {
    "full":   {0: "Baseline", 1: "NP", 2: "P1", 3: "P2", 4: "P3", 5: "P4"},
    "3class": {0: "No-pain", 1: "Lower-pain (P1-P2)", 2: "Higher-pain (P3-P4)"},
    "binary": {0: "No-pain", 1: "Pain"},
}
PMED_COVAS_NAMES = {
    "full":   {0: "No rating", 1: "Q1 (0-25%)", 2: "Q2 (25-50%)", 3: "Q3 (50-75%)", 4: "Q4 (75-100%)"},
    "3class": {0: "No-pain", 1: "Moderate", 2: "Severe"},
    "binary": {0: "No-pain", 1: "Pain"},
}
PMCD_NAMES = {
    "full":   {0: "No-pain", 1: "Moderate", 2: "Severe"},
    "3class": {0: "No-pain", 1: "Moderate", 2: "Severe"},
    "binary": {0: "No-pain", 1: "Pain"},
}


def _remap(y: np.ndarray, mapping: dict) -> np.ndarray:
    return np.vectorize(mapping.__getitem__)(y).astype(int)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_pmed(
    label: str = "heater",
    scheme: str = "3class",
    np_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load PMED numpy dataset.

    Parameters
    ----------
    label  : 'heater' (stimulus intensity) or 'covas' (subjective rating)
    scheme : 'full' | '3class' | 'binary'
    np_dir : override default path

    Returns
    -------
    X        : (N, 2500, 6)  float32
    y        : (N,)           int — remapped labels
    subjects : (N,)           int
    """
    d = np_dir or PMED_NP_DIR
    X        = np.load(d / "X.npy")[..., 0].astype(np.float32)  # (N,2500,6)
    y        = np.load(d / f"y_{label}.npy").argmax(axis=1).astype(int)
    subjects = np.load(d / "subjects.npy").astype(int)

    if label == "heater":
        mapping = {"full": _PMED_HEATER_FULL, "3class": _PMED_HEATER_3CLASS,
                   "binary": _PMED_HEATER_BINARY}[scheme]
    else:  # covas
        mapping = {"full": _PMED_COVAS_FULL, "3class": _PMED_COVAS_3CLASS,
                   "binary": _PMED_COVAS_BINARY}[scheme]

    return X, _remap(y, mapping), subjects


def load_pmcd(
    scheme: str = "3class",
    np_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load PMCD numpy dataset.

    Parameters
    ----------
    scheme : 'full' | '3class' | 'binary'
             PMCD is natively 3-class (no-pain / moderate / severe)

    Returns
    -------
    X        : (N, 1000, 7)  float32
    y        : (N,)           int
    subjects : (N,)           int
    """
    d = np_dir or PMCD_NP_DIR
    X        = np.load(d / "X.npy")[..., 0].astype(np.float32)  # (N,1000,7)
    y        = np.load(d / "y.npy").argmax(axis=1).astype(int)
    subjects = np.load(d / "subjects.npy").astype(int)

    mapping = {"full": _PMCD_FULL, "3class": _PMCD_3CLASS, "binary": _PMCD_BINARY}[scheme]
    return X, _remap(y, mapping), subjects


def select_sensors(
    X: np.ndarray,
    source_sensors: list[str],
    target_sensors: list[str],
) -> np.ndarray:
    """Return only the columns of X corresponding to target_sensors.

    Parameters
    ----------
    X              : (N, T, C)
    source_sensors : full list of sensor names for X
    target_sensors : subset to keep (must all be in source_sensors)
    """
    idx = [source_sensors.index(s) for s in target_sensors]
    return X[:, :, idx]
