"""
Cross-domain pain detection: train on PMED (experimental), evaluate on PMCD (clinical).

Both datasets share 5 common sensors: Bvp, Eda_E4, Resp, Eda_RB, Emg.
Feature extraction normalises for the different window lengths
(PMED: 10 s / 2500 samples vs PMCD: 4 s / 1000 samples).

The 3-class scheme is the natural alignment:
  PMED heater  →  0=no-pain {baseline+NP},  1=moderate {P1+P2},  2=severe {P3+P4}
  PMCD         →  0=no-pain,                1=moderate,           2=severe
                  (per-subject NRS thresholds applied by dataset authors)

Usage
-----
python run_cross_domain.py --model rf
python run_cross_domain.py --model svm --scheme binary
python run_cross_domain.py --model rf  --scheme 3class --also-within
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np

from pain_detection.config import PMED_SENSORS, PMCD_SENSORS, COMMON_SENSORS
from pain_detection.data_loader import (
    load_pmed, load_pmcd, select_sensors,
    PMED_HEATER_NAMES, PMCD_NAMES,
)
from pain_detection.features import extract_features
from pain_detection.models import build_rf, build_svm
from pain_detection.evaluate import held_out_eval, loso_cv


def parse_args():
    p = argparse.ArgumentParser(description="Cross-domain PMED → PMCD experiment")
    p.add_argument("--model",       choices=["rf", "svm"], default="rf")
    p.add_argument("--scheme",      choices=["binary", "3class"], default="3class",
                   help="Label scheme (3class is the natural common ground)")
    p.add_argument("--also-within", action="store_true",
                   help="Also run within-dataset LOSO-CV for comparison")
    return p.parse_args()


def main():
    args = parse_args()
    factory = build_rf if args.model == "rf" else build_svm
    scheme  = args.scheme

    label_names = PMED_HEATER_NAMES[scheme]   # same as PMCD_NAMES[scheme] for 3class/binary

    print("─" * 62)
    print("Cross-domain: PMED (train)  →  PMCD (test)")
    print(f"Label scheme : {scheme}  —  {list(label_names.values())}")
    print(f"Common sensors: {COMMON_SENSORS}")
    print("─" * 62)

    # ── Load and align ────────────────────────────────────────────────────────
    print("\nLoading PMED...")
    X_pmed, y_pmed, subj_pmed = load_pmed(label="heater", scheme=scheme)
    X_pmed = select_sensors(X_pmed, PMED_SENSORS, COMMON_SENSORS)
    _dist_str = lambda y: "  ".join(f"{label_names[c]}:{(y==c).sum()}" for c in sorted(set(y)))
    print(f"  PMED  {X_pmed.shape}  →  {_dist_str(y_pmed)}")

    print("Loading PMCD...")
    X_pmcd, y_pmcd, subj_pmcd = load_pmcd(scheme=scheme)
    X_pmcd = select_sensors(X_pmcd, PMCD_SENSORS, COMMON_SENSORS)
    print(f"  PMCD  {X_pmcd.shape}  →  {_dist_str(y_pmcd)}")

    # ── Feature extraction ────────────────────────────────────────────────────
    print(f"\nExtracting features (5 sensors × 16 = 80 features each)...")
    feat_pmed = extract_features(X_pmed, sensor_names=COMMON_SENSORS, fs=250).values.astype(np.float32)
    feat_pmcd = extract_features(X_pmcd, sensor_names=COMMON_SENSORS, fs=250).values.astype(np.float32)
    print(f"  PMED features: {feat_pmed.shape}  |  PMCD features: {feat_pmcd.shape}")

    # ── Cross-domain ──────────────────────────────────────────────────────────
    print(f"\n[Cross-domain]  Train: PMED  →  Test: PMCD  |  {args.model.upper()}")
    xd = held_out_eval(factory(), feat_pmed, y_pmed, feat_pmcd, y_pmcd)

    # ── Optional within-dataset comparison ───────────────────────────────────
    pmed_within = pmcd_within = None
    if args.also_within:
        print(f"\n[Within PMED LOSO-CV]  {args.model.upper()}")
        pmed_within = loso_cv(factory, feat_pmed, y_pmed, subj_pmed)

        print(f"\n[Within PMCD LOSO-CV]  {args.model.upper()}")
        pmcd_within = loso_cv(factory, feat_pmcd, y_pmcd, subj_pmcd)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  Model         : {args.model.upper()}  |  Scheme: {scheme}")
    print(f"  Classes       : {list(label_names.values())}")
    print(f"  Cross-domain  acc={xd['accuracy']:.4f}  F1={xd['f1']:.4f}" +
          (f"  AUC={xd['auc']:.4f}" if xd["auc"] else ""))
    if pmed_within:
        print(f"  Within-PMED   acc={pmed_within['accuracy']:.4f}  F1={pmed_within['f1']:.4f}")
    if pmcd_within:
        print(f"  Within-PMCD   acc={pmcd_within['accuracy']:.4f}  F1={pmcd_within['f1']:.4f}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
