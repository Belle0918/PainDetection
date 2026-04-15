"""
Evaluation utilities:
  - Leave-One-Subject-Out cross-validation (LOSO-CV)
  - Subject-grouped K-Fold cross-validation (faster alternative)
  - Metrics: accuracy, macro-F1, ROC-AUC (OvR), confusion matrix
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def loso_cv(
    model_factory,          # callable() → a fresh, unfitted model
    X: np.ndarray,          # (N, ...) — features or raw windows
    y: np.ndarray,          # (N,)     — integer class labels
    subjects: np.ndarray,   # (N,)     — subject IDs
    verbose: bool = True,
) -> dict:
    """Leave-One-Subject-Out cross-validation.

    Parameters
    ----------
    model_factory : zero-argument callable that returns a fresh model.
                    Model must implement fit(X, y) and predict(X).
                    Optionally predict_proba(X) for AUC.
    X             : feature matrix or raw windows
    y             : integer labels
    subjects      : per-sample subject ID (used to create the splits)
    verbose       : print per-fold summary

    Returns
    -------
    dict with keys: accuracy, f1, auc (if available), per_subject, predictions
    """
    unique_subjects = np.unique(subjects)
    all_true, all_pred, all_proba = [], [], []
    per_subject = []

    for sid in unique_subjects:
        test_mask  = subjects == sid
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if len(np.unique(y_train)) < 2:
            continue  # skip degenerate folds

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
            except Exception:
                pass

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        per_subject.append({"subject": sid, "accuracy": acc, "f1": f1, "n_test": int(test_mask.sum())})

        if verbose:
            print(f"  Subject {sid:3d} | acc={acc:.3f}  F1={f1:.3f}  n={test_mask.sum()}")

        all_true.append(y_test)
        all_pred.append(y_pred)
        if proba is not None:
            all_proba.append(proba)

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    overall_acc = accuracy_score(all_true, all_pred)
    overall_f1  = f1_score(all_true, all_pred, average="macro", zero_division=0)

    auc = None
    if all_proba:
        all_proba_arr = np.concatenate(all_proba)
        try:
            auc = roc_auc_score(all_true, all_proba_arr, multi_class="ovr", average="macro")
        except ValueError:
            pass

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  LOSO overall  | acc={overall_acc:.3f}  F1={overall_f1:.3f}" +
              (f"  AUC={auc:.3f}" if auc is not None else ""))
        print(classification_report(all_true, all_pred, zero_division=0))

    return {
        "accuracy":    overall_acc,
        "f1":          overall_f1,
        "auc":         auc,
        "per_subject": pd.DataFrame(per_subject),
        "predictions": (all_true, all_pred),
        "confusion":   confusion_matrix(all_true, all_pred),
    }


def grouped_kfold_cv(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    n_splits: int = 5,
    verbose: bool = True,
    random_state: int = 42,
) -> dict:
    """Subject-grouped K-Fold CV — faster alternative to LOSO.

    Subjects are randomly assigned to K folds so that no subject appears
    in both train and test.  This gives the same no-leak guarantee as LOSO
    but with K models instead of N_subjects models.
    """
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=n_splits)
    all_true, all_pred, all_proba = [], [], []
    per_fold = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=subjects)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        if len(np.unique(y_train)) < 2:
            continue

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
            except Exception:
                pass

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        n_subj = len(np.unique(subjects[test_idx]))
        per_fold.append({"fold": fold, "accuracy": acc, "f1": f1,
                         "n_test": len(test_idx), "n_subjects": n_subj})

        if verbose:
            print(f"  Fold {fold} | acc={acc:.3f}  F1={f1:.3f}  "
                  f"n={len(test_idx)}  subjects={n_subj}")

        all_true.append(y_test)
        all_pred.append(y_pred)
        if proba is not None:
            all_proba.append(proba)

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    overall_acc = accuracy_score(all_true, all_pred)
    overall_f1  = f1_score(all_true, all_pred, average="macro", zero_division=0)

    auc = None
    if all_proba:
        all_proba_arr = np.concatenate(all_proba)
        try:
            auc = roc_auc_score(all_true, all_proba_arr, multi_class="ovr", average="macro")
        except ValueError:
            pass

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  {n_splits}-Fold overall | acc={overall_acc:.3f}  F1={overall_f1:.3f}" +
              (f"  AUC={auc:.3f}" if auc is not None else ""))
        print(classification_report(all_true, all_pred, zero_division=0))

    return {
        "accuracy":    overall_acc,
        "f1":          overall_f1,
        "auc":         auc,
        "per_fold":    pd.DataFrame(per_fold),
        "predictions": (all_true, all_pred),
        "confusion":   confusion_matrix(all_true, all_pred),
    }


def held_out_eval(
    model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    verbose: bool = True,
) -> dict:
    """Simple train/test evaluation (used for cross-domain experiments)."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        except Exception:
            pass

    if verbose:
        print(f"  acc={acc:.3f}  F1={f1:.3f}" + (f"  AUC={auc:.3f}" if auc is not None else ""))
        print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "accuracy":  acc,
        "f1":        f1,
        "auc":       auc,
        "confusion": confusion_matrix(y_test, y_pred),
    }
