"""Live single-text inference using the persisted model bundle.

The pipeline writes ``outputs/model.joblib`` containing the fitted primary and
secondary models, the label order, and the recommended abstention threshold.
This module loads that bundle and scores raw text, applying the same abstention
rule the report card recommends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

# Extra confidence margin required to auto-decide when the two models disagree.
ABSTAIN_DELTA = 0.05


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load a model bundle saved by ``src.pipeline.run``."""
    return joblib.load(path)


def predict_texts(bundle: dict[str, Any], texts: list[str]) -> list[dict[str, Any]]:
    """Score raw texts and apply the abstention policy.

    Returns one dict per input text with the predicted label, confidence,
    per-class probabilities, model-disagreement flag, and abstain decision.
    """
    primary = bundle["primary_model"]
    other = bundle["other_model"]
    labels: list[str] = list(bundle["labels"])
    threshold = float(bundle["threshold"])

    primary_proba = primary.predict_proba(texts)
    other_pred = other.predict_proba(texts).argmax(axis=1)

    results: list[dict[str, Any]] = []
    for i in range(len(texts)):
        probs = primary_proba[i]
        pred_idx = int(probs.argmax())
        conf = float(probs.max())
        disagree = pred_idx != int(other_pred[i])
        abstain = (conf < threshold) or (disagree and conf < min(0.99, threshold + ABSTAIN_DELTA))
        results.append(
            {
                "pred_label": labels[pred_idx],
                "confidence": conf,
                "probs": {labels[j]: float(probs[j]) for j in range(len(labels))},
                "disagree": disagree,
                "abstain": abstain,
            }
        )
    return results
