"""
LLM-based baseline for pain detection.

Serialises extracted features → text prompt → Claude API → classification.
Evaluates zero-shot and few-shot on a sampled subset (LLM calls are expensive).

Usage
-----
# Zero-shot on PMED, 50 samples per class
python run_llm_baseline.py --dataset pmed --scheme binary --n-shots 0 --samples-per-class 50

# Few-shot (3 examples/class) on PMCD
python run_llm_baseline.py --dataset pmcd --scheme 3class --n-shots 3 --samples-per-class 30

# With context info
python run_llm_baseline.py --dataset pmed --scheme binary --n-shots 3 --context

# Compare zero vs few-shot
python run_llm_baseline.py --dataset pmcd --scheme 3class --compare
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from pain_detection.config import PMED_SENSORS, PMCD_SENSORS
from pain_detection.data_loader import (
    load_pmed, load_pmcd, select_sensors,
    PMED_HEATER_NAMES, PMED_COVAS_NAMES, PMCD_NAMES,
)
from pain_detection.features import extract_features
from pain_detection.models import LLMBaseline

DEFAULT_SENSORS = ["Bvp", "Eda_E4", "Emg", "Resp"]


def parse_args():
    p = argparse.ArgumentParser(description="LLM baseline for pain detection")
    p.add_argument("--dataset", choices=["pmed", "pmcd"], default="pmed")
    p.add_argument("--label",   choices=["heater", "covas"], default="heater")
    p.add_argument("--scheme",  choices=["binary", "3class"], default="binary")
    p.add_argument("--n-shots", type=int, default=0,
                   help="Few-shot examples per class (0=zero-shot)")
    p.add_argument("--samples-per-class", type=int, default=50,
                   help="Test samples per class (to limit API cost)")
    p.add_argument("--context", action="store_true",
                   help="Include recording context in prompt")
    p.add_argument("--compare", action="store_true",
                   help="Run both zero-shot and few-shot (3) for comparison")
    p.add_argument("--model", default="claude-sonnet-4-20250514",
                   help="Claude model ID")
    p.add_argument("--prompt-mode", choices=["raw", "semantic"], default="semantic",
                   help="Prompt mode: 'raw' (64-feature dump) or 'semantic' (natural language)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sample_balanced(X, y, samples_per_class, rng):
    """Sample up to samples_per_class from each class."""
    indices = []
    for c in sorted(np.unique(y)):
        cls_idx = np.where(y == c)[0]
        n = min(samples_per_class, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n, replace=False)
        indices.extend(chosen.tolist())
    indices = sorted(indices)
    return X[indices], y[indices]


def run_one(feat_train, y_train, feat_test, y_test, feature_names,
            class_names, n_shots, context_str, model_id, seed,
            prompt_mode="semantic"):
    """Run a single LLM evaluation."""
    ctx_str = context_str  # None or a descriptive string

    llm = LLMBaseline(
        class_names=class_names,
        context=ctx_str,
        n_shots=n_shots,
        model=model_id,
        random_state=seed,
        prompt_mode=prompt_mode,
    )
    llm.fit(feat_train, y_train, feature_names=feature_names)
    preds = llm.predict(feat_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    print(classification_report(y_test, preds,
                                target_names=class_names, zero_division=0))
    return {"accuracy": acc, "f1": f1}


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.dataset.upper()} ({args.scheme})...")
    if args.dataset == "pmed":
        X, y, subjects = load_pmed(label=args.label, scheme=args.scheme)
        all_sensors = PMED_SENSORS
        label_names = (PMED_HEATER_NAMES if args.label == "heater"
                       else PMED_COVAS_NAMES)[args.scheme]
        context_str = "controlled experimental (heat pain)"
    else:
        X, y, subjects = load_pmcd(scheme=args.scheme)
        all_sensors = PMCD_SENSORS
        label_names = PMCD_NAMES[args.scheme]
        context_str = "clinical physiotherapy"

    X = select_sensors(X, all_sensors, DEFAULT_SENSORS)
    classes = sorted(np.unique(y).tolist())
    class_names = [label_names[c] for c in classes]

    print(f"  X shape : {X.shape}")
    print(f"  Classes : {class_names}")
    counts = {label_names[c]: int((y == c).sum()) for c in classes}
    print(f"  Counts  : {counts}")

    # ── Extract features ─────────────────────────────────────────────────────
    print(f"\nExtracting features ({len(DEFAULT_SENSORS)} sensors × 16)...")
    feat_df = extract_features(X, sensor_names=DEFAULT_SENSORS, fs=250)
    feature_names = feat_df.columns.tolist()
    feat = feat_df.values.astype(np.float32)

    # ── Split: use ~20% as train (for few-shot pool), rest as test ────────
    # Simple stratified split by subject
    unique_subj = np.unique(subjects)
    rng.shuffle(unique_subj)
    n_train_subj = max(3, len(unique_subj) // 5)
    train_subj = set(unique_subj[:n_train_subj])

    train_mask = np.array([s in train_subj for s in subjects])
    test_mask = ~train_mask

    feat_train, y_train = feat[train_mask], y[train_mask]
    feat_test_full, y_test_full = feat[test_mask], y[test_mask]

    # Sample test set to limit API calls
    feat_test, y_test = sample_balanced(
        feat_test_full, y_test_full, args.samples_per_class, rng)
    print(f"  Train pool : {len(feat_train)} samples ({n_train_subj} subjects)")
    print(f"  Test set   : {len(feat_test)} samples (sampled)")

    # ── Run experiments ──────────────────────────────────────────────────────
    if args.compare:
        configs = [(0, "Zero-shot"), (3, "Few-shot (k=3)"), (5, "Few-shot (k=5)")]
    else:
        shot_label = f"Few-shot (k={args.n_shots})" if args.n_shots > 0 else "Zero-shot"
        configs = [(args.n_shots, shot_label)]

    results = {}
    for n_shots, label in configs:
        print(f"\n{'─'*55}")
        print(f"  [{label}]  context={'yes' if args.context else 'no'}")
        print(f"{'─'*55}")
        ctx = context_str if args.context else None
        res = run_one(feat_train, y_train, feat_test, y_test,
                      feature_names, class_names, n_shots, ctx,
                      args.model, args.seed, args.prompt_mode)
        results[label] = res

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  LLM Baseline Results  ({args.dataset.upper()} {args.scheme})")
    print(f"  Model: {args.model}")
    print(f"{'─'*55}")
    print(f"  {'Config':<22} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"{'─'*55}")
    for label, res in results.items():
        print(f"  {label:<22} {res['accuracy']:>10.4f} {res['f1']:>10.4f}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
