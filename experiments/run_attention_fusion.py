"""
Attention-fusion model experiment with LOSO-CV.

Usage examples
--------------
# PMED, 4 modalities (BVP, EDA, EMG, Resp), binary
python run_attention_fusion.py --dataset pmed --scheme binary

# PMCD, 3-class
python run_attention_fusion.py --dataset pmcd --scheme 3class

# PMCD with focal loss
python run_attention_fusion.py --dataset pmcd --scheme 3class --focal-loss

# Single modality ablation (BVP only)
python run_attention_fusion.py --dataset pmed --scheme binary --sensors Bvp
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np

from pain_detection.config import PMED_SENSORS, PMCD_SENSORS
from pain_detection.data_loader import (
    load_pmed, load_pmcd, select_sensors,
    PMED_HEATER_NAMES, PMED_COVAS_NAMES, PMCD_NAMES,
)
from pain_detection.models import AttentionFusionModel
from pain_detection.evaluate import loso_cv


def parse_args():
    p = argparse.ArgumentParser(description="Attention-fusion pain detection with LOSO-CV")
    p.add_argument("--dataset", choices=["pmed", "pmcd"], default="pmed")
    p.add_argument("--label",   choices=["heater", "covas"], default="heater",
                   help="PMED label type (ignored for PMCD)")
    p.add_argument("--scheme",  choices=["binary", "3class", "full"], default="binary")
    p.add_argument("--sensors", nargs="+", default=None,
                   help="Sensor subset (default: BVP, EDA_E4, EMG, Resp)")
    # Model hyperparameters
    p.add_argument("--latent-dim",  type=int, default=64)
    p.add_argument("--filters",     type=int, nargs="+", default=[32, 64])
    p.add_argument("--kernel-size", type=int, default=7)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--focal-loss",  action="store_true")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    return p.parse_args()


# Default 4 modalities used in the paper (shared across PMED & PMCD)
DEFAULT_SENSORS = ["Bvp", "Eda_E4", "Emg", "Resp"]


def main():
    args = parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.dataset.upper()} ({args.scheme})...")
    if args.dataset == "pmed":
        X, y, subjects = load_pmed(label=args.label, scheme=args.scheme)
        all_sensors = PMED_SENSORS
        label_names = (PMED_HEATER_NAMES if args.label == "heater" else PMED_COVAS_NAMES)[args.scheme]
        default_sensors = DEFAULT_SENSORS
    else:
        X, y, subjects = load_pmcd(scheme=args.scheme)
        all_sensors = PMCD_SENSORS
        label_names = PMCD_NAMES[args.scheme]
        default_sensors = DEFAULT_SENSORS

    classes = sorted(np.unique(y).tolist())
    n_classes = len(classes)

    # ── Sensor selection ─────────────────────────────────────────────────────
    chosen = args.sensors or default_sensors
    bad = [s for s in chosen if s not in all_sensors]
    if bad:
        print(f"Unknown sensors: {bad}.  Available: {all_sensors}")
        return
    X = select_sensors(X, all_sensors, chosen)
    n_modalities = len(chosen)

    print(f"  X shape     : {X.shape}")
    print(f"  Classes     : {[label_names[c] for c in classes]}")
    counts = {label_names[c]: int((y == c).sum()) for c in classes}
    print(f"  Counts      : {counts}")
    print(f"  Modalities  : {chosen} ({n_modalities})")

    # ── Model factory ────────────────────────────────────────────────────────
    def model_factory():
        return AttentionFusionModel(
            n_modalities=n_modalities,
            n_classes=n_classes,
            latent_dim=args.latent_dim,
            filters=tuple(args.filters),
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
        )

    print(f"\nRunning LOSO-CV with AttentionFusion ({n_modalities} encoders)...")
    results = loso_cv(model_factory, X, y, subjects)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Dataset    : {args.dataset.upper()}")
    print(f"  Model      : AttentionFusion")
    print(f"  Modalities : {chosen}")
    print(f"  Classes    : {[label_names[c] for c in classes]}")
    print(f"  Accuracy   : {results['accuracy']:.4f}")
    print(f"  Macro F1   : {results['f1']:.4f}")
    if results["auc"] is not None:
        print(f"  Macro AUC  : {results['auc']:.4f}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
