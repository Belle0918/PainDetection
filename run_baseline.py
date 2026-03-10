"""
Baseline pain detection experiment using Leave-One-Subject-Out CV (LOSO-CV).

Label schemes
─────────────
  binary  : pain vs no-pain (2 classes)
  3class  : no-pain / moderate / severe  (PMED heater: {0,1}→0  {2,3}→1  {4,5}→2;
                                          PMCD already natively 3-class)
  full    : all original classes (PMED heater: 6, PMED covas: 5, PMCD: 3)

Usage examples
--------------
# PMED, RF, 3-class severity, all 6 sensors
python run_baseline.py --dataset pmed --label heater --scheme 3class --model rf

# PMED, RF, binary, 3 specific sensors
python run_baseline.py --dataset pmed --scheme binary --model rf \
    --sensors Bvp Eda_E4 Emg

# PMED, SVM, subjective CoVAS 3-class
python run_baseline.py --dataset pmed --label covas --scheme 3class --model svm

# PMCD, RF, 3-class (natural for clinical data)
python run_baseline.py --dataset pmcd --scheme 3class --model rf

# PMED, 1D-CNN on raw windows (requires torch)
python run_baseline.py --dataset pmed --scheme binary --model cnn
"""
import argparse
import sys
import numpy as np

from pain_detection.config import PMED_SENSORS, PMCD_SENSORS
from pain_detection.data_loader import (
    load_pmed, load_pmcd, select_sensors,
    PMED_HEATER_NAMES, PMED_COVAS_NAMES, PMCD_NAMES,
)
from pain_detection.features import extract_features
from pain_detection.models import build_rf, build_svm, CNN1D
from pain_detection.evaluate import loso_cv


def parse_args():
    p = argparse.ArgumentParser(description="Baseline pain detection with LOSO-CV")
    p.add_argument("--dataset", choices=["pmed", "pmcd"], default="pmed")
    p.add_argument("--label",   choices=["heater", "covas"], default="heater",
                   help="PMED label type (ignored for PMCD)")
    p.add_argument("--scheme",  choices=["binary", "3class", "full"], default="3class",
                   help="Label granularity: binary | 3class | full")
    p.add_argument("--model",   choices=["rf", "svm", "cnn"], default="rf")
    p.add_argument("--sensors", nargs="+", default=None,
                   help="Sensor subset (≥3 required; default: all)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.dataset.upper()} ({args.scheme})...")
    if args.dataset == "pmed":
        X, y, subjects = load_pmed(label=args.label, scheme=args.scheme)
        all_sensors = PMED_SENSORS
        label_names = (PMED_HEATER_NAMES if args.label == "heater" else PMED_COVAS_NAMES)[args.scheme]
        label_desc  = f"{args.label} / {args.scheme}"
    else:
        X, y, subjects = load_pmcd(scheme=args.scheme)
        all_sensors = PMCD_SENSORS
        label_names = PMCD_NAMES[args.scheme]
        label_desc  = f"NRS thresholds / {args.scheme}"

    classes = sorted(np.unique(y).tolist())
    print(f"  X shape : {X.shape}")
    print(f"  Classes : {[label_names[c] for c in classes]}  (integers {classes})")
    counts = {label_names[c]: int((y == c).sum()) for c in classes}
    print(f"  Counts  : {counts}")

    # ── Sensor selection ─────────────────────────────────────────────────────
    chosen = args.sensors or all_sensors
    bad = [s for s in chosen if s not in all_sensors]
    if bad:
        print(f"Unknown sensors: {bad}.  Available: {all_sensors}")
        sys.exit(1)
    if len(chosen) < 3:
        print("At least 3 sensors are required.")
        sys.exit(1)

    X = select_sensors(X, all_sensors, chosen)
    print(f"  Sensors ({len(chosen)}): {chosen}")

    # ── Model factory ─────────────────────────────────────────────────────────
    n_classes = len(classes)
    n_sensors = X.shape[2]

    if args.model == "cnn":
        def model_factory():
            return CNN1D(n_sensors=n_sensors, n_classes=n_classes)
        print(f"\nRunning LOSO-CV with 1D-CNN (raw windows, {X.shape[1]} steps × {n_sensors} ch)...")
        results = loso_cv(model_factory, X, y, subjects)

    else:
        print(f"\nExtracting features ({len(chosen)} sensors × 16 = {len(chosen)*16} features)...")
        feat = extract_features(X, sensor_names=chosen, fs=250).values.astype(np.float32)
        print(f"  Feature matrix: {feat.shape}")

        factory = build_rf if args.model == "rf" else build_svm
        print(f"\nRunning LOSO-CV with {args.model.upper()}...")
        results = loso_cv(factory, feat, y, subjects)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  Dataset : {args.dataset.upper()}  |  Label : {label_desc}")
    print(f"  Model   : {args.model.upper()}")
    print(f"  Classes : {[label_names[c] for c in classes]}")
    print(f"  Accuracy (LOSO) : {results['accuracy']:.4f}")
    print(f"  Macro F1        : {results['f1']:.4f}")
    if results["auc"] is not None:
        print(f"  Macro AUC       : {results['auc']:.4f}")
    print(f"{'═'*58}\n")


if __name__ == "__main__":
    main()
