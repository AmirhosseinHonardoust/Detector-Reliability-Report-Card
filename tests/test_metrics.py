"""Unit tests for src.metrics with hand-computed expected values."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from src.metrics import (  # noqa: E402
    compute_overall,
    coverage_curve,
    expected_calibration_error,
    multiclass_brier,
)


def test_expected_calibration_error_hand_computed():
    y_true = np.array([0, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    # conf=[0.9,0.8], both correct -> bins (0.8,0.9] and (0.7,0.8]
    # ECE = 0.5*|1-0.9| + 0.5*|1-0.8| = 0.15
    assert expected_calibration_error(y_true, proba) == pytest.approx(0.15)


def test_ece_perfectly_calibrated_is_low():
    # confident (0.95) and always correct -> ECE equals the 0.05 gap
    proba = np.column_stack([np.full(50, 0.95), np.full(50, 0.05)])
    y_true = np.zeros(50, dtype=int)
    assert expected_calibration_error(y_true, proba) == pytest.approx(0.05, abs=1e-9)


def test_multiclass_brier_hand_computed():
    y_true = np.array([0, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    # row1: 0.01+0.01=0.02 ; row2: 0.04+0.04=0.08 ; mean=0.05
    assert multiclass_brier(y_true, proba, n_classes=2) == pytest.approx(0.05)


def test_coverage_curve_thresholding():
    y_true = np.array([0, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    df = coverage_curve(y_true, proba, np.array([0.0, 0.85]))
    rows = df.to_dict("records")
    assert rows[0]["coverage"] == pytest.approx(1.0)
    assert rows[0]["accuracy"] == pytest.approx(1.0)
    # at 0.85 only the 0.9-confidence sample is kept
    assert rows[1]["coverage"] == pytest.approx(0.5)
    assert rows[1]["accuracy"] == pytest.approx(1.0)


def test_coverage_curve_empty_subset_is_nan():
    y_true = np.array([0, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    df = coverage_curve(y_true, proba, np.array([0.99]))
    row = df.iloc[0]
    assert row["coverage"] == pytest.approx(0.0)
    assert np.isnan(row["accuracy"])
    assert np.isnan(row["macro_f1"])


def test_compute_overall_keys_and_values():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    out = compute_overall(y_true, y_pred, proba, labels=["ai", "human"])
    assert set(out) == {"accuracy", "macro_f1", "ece", "brier", "labels", "confusion_matrix"}
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["macro_f1"] == pytest.approx(1.0)
    assert out["labels"] == ["ai", "human"]
    assert out["confusion_matrix"] == [[1, 0], [0, 1]]
