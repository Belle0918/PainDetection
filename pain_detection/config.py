"""
Shared configuration for the pain detection project.
"""
from pathlib import Path

# ── Dataset paths ─────────────────────────────────────────────────────────────
PMED_NP_DIR = Path(__file__).parent.parent / "PMED" / "dataset" / "np-dataset"
PMCD_NP_DIR = Path(__file__).parent.parent / "PMCD" / "dataset" / "np-dataset"

# ── Sensor names ──────────────────────────────────────────────────────────────
PMED_SENSORS = ["Bvp", "Eda_E4", "Resp", "Eda_RB", "Ecg", "Emg"]
PMCD_SENSORS = ["Bvp", "Eda_E4", "Tmp", "Resp", "Eda_RB", "Bvp_RB", "Emg"]

# Sensors present in both datasets (used for cross-domain experiments)
COMMON_SENSORS = ["Bvp", "Eda_E4", "Resp", "Eda_RB", "Emg"]

# Indices of COMMON_SENSORS within each dataset's sensor list
PMED_COMMON_IDX = [PMED_SENSORS.index(s) for s in COMMON_SENSORS]   # [0,1,2,3,5]
PMCD_COMMON_IDX = [PMCD_SENSORS.index(s) for s in COMMON_SENSORS]   # [0,1,3,4,6]

# ── Sampling ───────────────────────────────────────────────────────────────────
SAMPLING_RATE = 250  # Hz

# ── PMED label details ────────────────────────────────────────────────────────
# y_heater classes: 0=baseline, 1=NP (non-painful), 2-5=P1-P4 (painful)
# Binary pain: class 0-1 → no-pain (0), class 2-5 → pain (1)
PMED_PAIN_THRESHOLD = 2   # heater class index at which pain begins
