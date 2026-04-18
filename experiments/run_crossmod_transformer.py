"""
CrossMod-Transformer experiment.

Implementation of Farmani et al. 2025 (Nature Scientific Reports,
s41598-025-14238-y) adapted to PMED / PMCD with LOSO-CV.

Usage
-----
# PMCD, 3-class (the hard one we want to push F1>60 on)
python experiments/run_crossmod_transformer.py --dataset pmcd --scheme 3class

# Same but with focal loss for the imbalanced moderate class
python experiments/run_crossmod_transformer.py --dataset pmcd --scheme 3class --focal-loss

# PMED binary, 4 modalities (default: Bvp, Eda_E4, Resp, Emg)
python experiments/run_crossmod_transformer.py --dataset pmed --scheme binary

# Faster: grouped 5-fold CV instead of full LOSO
python experiments/run_crossmod_transformer.py --dataset pmcd --scheme 3class --cv kfold --kfolds 5
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
from pain_detection.models import CrossModTransformer
from pain_detection.evaluate import loso_cv, grouped_kfold_cv


DEFAULT_SENSORS = ["Bvp", "Eda_E4", "Resp", "Emg"]


def parse_args():
    p = argparse.ArgumentParser(description="CrossMod-Transformer pain detection")
    p.add_argument("--dataset", choices=["pmed", "pmcd"], default="pmcd")
    p.add_argument("--label",   choices=["heater", "covas"], default="heater")
    p.add_argument("--scheme",  choices=["binary", "3class", "full"], default="3class")
    p.add_argument("--sensors", nargs="+", default=None,
                   help=f"Sensor subset (default: {DEFAULT_SENSORS})")

    p.add_argument("--cv", choices=["loso", "kfold"], default="loso")
    p.add_argument("--kfolds", type=int, default=5)

    # Architecture (paper defaults)
    p.add_argument("--d-model",      type=int, default=128)
    p.add_argument("--lstm-hidden",  type=int, default=64)
    p.add_argument("--lstm-layers",  type=int, default=2)
    p.add_argument("--n-heads",      type=int, default=8)
    p.add_argument("--ffn-hidden",   type=int, default=128)
    p.add_argument("--dropout",      type=float, default=0.3)

    # Training
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup",       type=int,   default=5)
    p.add_argument("--focal-loss",   action="store_true")
    p.add_argument("--focal-gamma",  type=float, default=2.0)
    p.add_argument("--target-length", type=int, default=250,
                   help="Downsample each window to this length (paper: 138 @ 25Hz). "
                        "Use 0 to disable.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.dataset.upper()} ({args.scheme})...")
    if args.dataset == "pmed":
        X, y, subjects = load_pmed(label=args.label, scheme=args.scheme)
        all_sensors = PMED_SENSORS
        label_names = (PMED_HEATER_NAMES if args.label == "heater" else PMED_COVAS_NAMES)[args.scheme]
    else:
        X, y, subjects = load_pmcd(scheme=args.scheme)
        all_sensors = PMCD_SENSORS
        label_names = PMCD_NAMES[args.scheme]

    chosen = args.sensors or DEFAULT_SENSORS
    bad = [s for s in chosen if s not in all_sensors]
    if bad:
        print(f"Unknown sensors: {bad}.  Available: {all_sensors}")
        return
    X = select_sensors(X, all_sensors, chosen)
    n_modalities = len(chosen)
    classes = sorted(np.unique(y).tolist())
    n_classes = len(classes)

    print(f"  X shape     : {X.shape}")
    print(f"  Classes     : {[label_names[c] for c in classes]}")
    counts = {label_names[c]: int((y == c).sum()) for c in classes}
    print(f"  Counts      : {counts}")
    print(f"  Modalities  : {chosen} ({n_modalities})")

    # ── Model factory ────────────────────────────────────────────────────────
    target_length = args.target_length if args.target_length > 0 else None

    def model_factory():
        return CrossModTransformer(
            n_modalities=n_modalities,
            n_classes=n_classes,
            d_model=args.d_model,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            n_heads=args.n_heads,
            ffn_hidden=args.ffn_hidden,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup,
            focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
            target_length=target_length,
        )

    # ── Run CV ───────────────────────────────────────────────────────────────
    print(f"\nRunning {'LOSO-CV' if args.cv == 'loso' else f'{args.kfolds}-fold grouped CV'} "
          f"with CrossMod-Transformer ({n_modalities} modalities)...")
    if args.cv == "loso":
        results = loso_cv(model_factory, X, y, subjects)
    else:
        results = grouped_kfold_cv(model_factory, X, y, subjects, n_splits=args.kfolds)

    print(f"\n{'═'*60}")
    print(f"  Dataset    : {args.dataset.upper()}  |  Scheme: {args.scheme}")
    print(f"  Model      : CrossMod-Transformer")
    print(f"  Modalities : {chosen}")
    print(f"  Classes    : {[label_names[c] for c in classes]}")
    print(f"  Accuracy   : {results['accuracy']:.4f}")
    print(f"  Macro F1   : {results['f1']:.4f}")
    if results["auc"] is not None:
        print(f"  Macro AUC  : {results['auc']:.4f}")
    print(f"  Confusion  :\n{results['confusion']}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
