"""Tests for model persistence and live inference."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from src.inference import load_bundle, predict_texts  # noqa: E402
from src.pipeline import run  # noqa: E402


def _make_csv(path) -> None:
    rows = []
    for i in range(20):
        rows.append({"text": f"machine generated model output sample {i}", "label": "ai"})
        rows.append({"text": f"i went to the market today with friends {i}", "label": "human"})
        rows.append(
            {"text": f"machine output lightly revised by a person {i}", "label": "post_edited_ai"}
        )
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("infer")
    csv = tmp / "tiny.csv"
    _make_csv(csv)
    res = run(input_path=str(csv), out_dir=str(tmp / "out"), figures_dir=str(tmp / "fig"))
    return res["model_path"]


def test_pipeline_writes_model_bundle(bundle_path):
    bundle = load_bundle(bundle_path)
    assert set(bundle) == {"primary_model", "other_model", "labels", "threshold", "primary_name"}
    assert set(bundle["labels"]) == {"ai", "human", "post_edited_ai"}
    assert 0.0 <= bundle["threshold"] <= 1.0


def test_predict_texts_contract(bundle_path):
    bundle = load_bundle(bundle_path)
    out = predict_texts(bundle, ["some neutral text", "another bit of text"])
    assert len(out) == 2
    for r in out:
        assert r["pred_label"] in bundle["labels"]
        assert 0.0 <= r["confidence"] <= 1.0
        assert pytest.approx(sum(r["probs"].values()), abs=1e-6) == 1.0
        assert isinstance(r["disagree"], bool)
        assert isinstance(r["abstain"], bool)


def test_abstain_rule_is_consistent_with_threshold(bundle_path):
    bundle = load_bundle(bundle_path)
    thr = float(bundle["threshold"])
    r = predict_texts(bundle, ["short"])[0]
    # below threshold and no disagreement -> must abstain
    if r["confidence"] < thr and not r["disagree"]:
        assert r["abstain"] is True
    # confident and agreeing -> must auto-decide
    if r["confidence"] >= thr and not r["disagree"]:
        assert r["abstain"] is False
