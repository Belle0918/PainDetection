"""
Cross-context transfer learning: pretrain on PMED, fine-tune on PMCD.

Compares three strategies:
  1. from-scratch : train AttentionFusion on PMCD only (baseline)
  2. frozen       : pretrain on PMED, freeze encoders, train head on PMCD
  3. finetune     : pretrain on PMED, fine-tune all with reduced encoder lr

Usage
-----
python run_transfer.py --scheme 3class
python run_transfer.py --scheme binary --epochs 30
python run_transfer.py --scheme 3class --strategies frozen finetune
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np

from pain_detection.config import PMED_SENSORS, PMCD_SENSORS
from pain_detection.data_loader import (
    load_pmed, load_pmcd, select_sensors,
    PMED_HEATER_NAMES, PMCD_NAMES,
)
from pain_detection.models import AttentionFusionModel, TransferModel
from pain_detection.evaluate import loso_cv

# 4 shared modalities used in the paper
TRANSFER_SENSORS = ["Bvp", "Eda_E4", "Resp", "Emg"]


def parse_args():
    p = argparse.ArgumentParser(description="Transfer learning PMED → PMCD")
    p.add_argument("--src-scheme", choices=["binary", "3class"], default="binary",
                   help="Source (PMED) label scheme for pretraining")
    p.add_argument("--scheme",     choices=["binary", "3class"], default="3class",
                   help="Target (PMCD) label scheme")
    p.add_argument("--strategies", nargs="+",
                   default=["scratch", "frozen", "finetune"],
                   choices=["scratch", "frozen", "finetune"])
    p.add_argument("--latent-dim",  type=int, default=64)
    p.add_argument("--filters",     type=int, nargs="+", default=[32, 64])
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--focal-loss",  action="store_true")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    return p.parse_args()


def main():
    args = parse_args()
    n_modalities = len(TRANSFER_SENSORS)
    model_kwargs = dict(
        latent_dim=args.latent_dim,
        filters=tuple(args.filters),
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
    )

    print("═" * 65)
    print("  Transfer Learning: PMED → PMCD")
    print(f"  Source scheme : {args.src_scheme}  |  Target scheme : {args.scheme}")
    print(f"  Strategies    : {args.strategies}")
    print(f"  Modalities    : {TRANSFER_SENSORS}")
    print("═" * 65)

    # ── Load source (PMED) ───────────────────────────────────────────────────
    print("\nLoading PMED...")
    X_pmed, y_pmed, subj_pmed = load_pmed(label="heater", scheme=args.src_scheme)
    X_pmed = select_sensors(X_pmed, PMED_SENSORS, TRANSFER_SENSORS)

    src_classes = sorted(np.unique(y_pmed).tolist())
    n_classes_src = len(src_classes)
    print(f"  PMED  {X_pmed.shape}  classes={n_classes_src}")

    # ── Load target (PMCD) ───────────────────────────────────────────────────
    print("Loading PMCD...")
    X_pmcd, y_pmcd, subj_pmcd = load_pmcd(scheme=args.scheme)
    X_pmcd = select_sensors(X_pmcd, PMCD_SENSORS, TRANSFER_SENSORS)

    tgt_classes = sorted(np.unique(y_pmcd).tolist())
    n_classes_tgt = len(tgt_classes)
    tgt_names = PMCD_NAMES[args.scheme]
    print(f"  PMCD  {X_pmcd.shape}  classes={n_classes_tgt}")
    counts = {tgt_names[c]: int((y_pmcd == c).sum()) for c in tgt_classes}
    print(f"  Counts: {counts}")

    # ── Pretrain on PMED (shared across frozen/finetune) ─────────────────────
    encoder_state = None
    if any(s in args.strategies for s in ["frozen", "finetune"]):
        print(f"\n[Pretraining] AttentionFusion on PMED ({args.src_scheme})...")
        pretrain_model = AttentionFusionModel(
            n_modalities=n_modalities,
            n_classes=n_classes_src,
            **model_kwargs,
        )
        pretrain_model.fit(X_pmed, y_pmed)
        encoder_state = pretrain_model.get_encoder_state()
        print("  Pretraining done.")

    # ── Evaluate each strategy via LOSO-CV on PMCD ───────────────────────────
    all_results = {}

    for strategy in args.strategies:
        print(f"\n[{strategy.upper()}] LOSO-CV on PMCD...")

        if strategy == "scratch":
            def factory():
                return AttentionFusionModel(
                    n_modalities=n_modalities,
                    n_classes=n_classes_tgt,
                    **model_kwargs,
                )
        else:
            def factory(strat=strategy, state=encoder_state):
                tm = TransferModel(
                    n_modalities=n_modalities,
                    n_classes_src=n_classes_src,
                    n_classes_tgt=n_classes_tgt,
                    strategy=strat,
                    **model_kwargs,
                )
                # Inject pretrained state directly (skip re-pretraining each fold)
                tm._encoder_state = state
                return tm

        results = loso_cv(factory, X_pmcd, y_pmcd, subj_pmcd)
        all_results[strategy] = results

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  Transfer Learning Results  (PMCD {args.scheme})")
    print(f"  Modalities: {TRANSFER_SENSORS}")
    print(f"{'─'*65}")
    print(f"  {'Strategy':<12} {'Accuracy':>10} {'Macro F1':>10} {'AUC':>10}")
    print(f"{'─'*65}")
    for strategy, res in all_results.items():
        auc_str = f"{res['auc']:.4f}" if res['auc'] is not None else "  N/A"
        print(f"  {strategy:<12} {res['accuracy']:>10.4f} {res['f1']:>10.4f} {auc_str:>10}")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
