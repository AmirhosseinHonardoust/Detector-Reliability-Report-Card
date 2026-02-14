from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix (test)")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=25, ha="right")
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()

def plot_reliability(y_true: np.ndarray, proba: np.ndarray, out_path: Path, n_bins: int = 10) -> None:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc, bin_conf = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_acc.append(correct[mask].mean())
        bin_conf.append(conf[mask].mean())
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(bin_conf, bin_acc, marker="o")
    plt.title("Reliability diagram (confidence vs accuracy)")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()

def plot_coverage(curve: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(curve["coverage"], curve["accuracy"], marker="o", label="accuracy")
    plt.plot(curve["coverage"], curve["macro_f1"], marker="o", label="macro_f1")
    plt.title("Coverage vs performance under abstention")
    plt.xlabel("Coverage (fraction auto-decided)")
    plt.ylabel("Performance")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()

def plot_confidence_hist(proba: np.ndarray, out_path: Path) -> None:
    conf = proba.max(axis=1)
    plt.figure(figsize=(7, 5))
    plt.hist(conf, bins=20, edgecolor="black")
    plt.title("Confidence histogram (max probability)")
    plt.xlabel("Max predicted probability")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()
