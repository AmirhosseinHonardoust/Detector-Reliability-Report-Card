from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)

def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    oh = np.zeros((len(y_true), n_classes), dtype=float)
    oh[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - oh) ** 2, axis=1)))

def compute_overall(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "ece": expected_calibration_error(y_true, proba, n_bins=10),
        "brier": multiclass_brier(y_true, proba, n_classes=len(labels)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def coverage_curve(y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    rows = []
    for t in thresholds:
        keep = conf >= t
        cov = float(keep.mean())
        if keep.sum() == 0:
            rows.append({"threshold": float(t), "coverage": cov, "accuracy": np.nan, "macro_f1": np.nan})
            continue
        rows.append({
            "threshold": float(t),
            "coverage": cov,
            "accuracy": float(accuracy_score(y_true[keep], pred[keep])),
            "macro_f1": float(f1_score(y_true[keep], pred[keep], average="macro")),
        })
    return pd.DataFrame(rows)
