"""End-to-end smoke test: runs the full pipeline on a tiny synthetic dataset.

Skips cleanly when the heavy/optional deps (scikit-learn, matplotlib) are absent.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

from src.pipeline import run  # noqa: E402


def _make_csv(path) -> None:
    rows = []
    per_class = 20
    for i in range(per_class):
        rows.append({"text": f"machine generated model output sample {i}", "label": "ai"})
        rows.append({"text": f"i went to the market today with friends {i}", "label": "human"})
        rows.append(
            {"text": f"machine output lightly revised by a person {i}", "label": "post_edited_ai"}
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_pipeline_end_to_end(tmp_path):
    csv = tmp_path / "tiny.csv"
    _make_csv(csv)
    out_dir = tmp_path / "outputs"
    fig_dir = tmp_path / "figures"

    res = run(
        input_path=str(csv),
        out_dir=str(out_dir),
        figures_dir=str(fig_dir),
        random_state=0,
    )

    # return contract
    assert res["primary_model"] in {"word", "char"}
    assert set(res["labels"]) == {"ai", "human", "post_edited_ai"}
    assert "recommended_threshold" in res["policy"]

    # artifacts written
    for name in (
        "metrics_overall.json",
        "abstention_policy.json",
        "splits_summary.json",
        "test_predictions.csv",
        "coverage_curve.csv",
    ):
        assert (out_dir / name).exists(), name
    for name in (
        "confusion_matrix.png",
        "reliability_diagram.png",
        "coverage_vs_accuracy.png",
        "probability_histograms.png",
    ):
        assert (fig_dir / name).exists(), name

    # metrics shape
    metrics = json.loads((out_dir / "metrics_overall.json").read_text(encoding="utf-8"))
    for key in ("accuracy", "macro_f1", "ece", "brier", "labels", "confusion_matrix"):
        assert key in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    # predictions table has per-class probability columns
    preds = pd.read_csv(out_dir / "test_predictions.csv")
    for lab in res["labels"]:
        assert f"p_{lab}" in preds.columns
    assert {"pred_label", "confidence", "disagree_word_char"}.issubset(preds.columns)
