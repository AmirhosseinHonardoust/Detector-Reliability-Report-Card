"""Unit tests for src.split (sizes, stratification, reproducibility)."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.split import SplitConfig, make_splits  # noqa: E402


def _frame(n: int = 100) -> pd.DataFrame:
    labels = ["a"] * 40 + ["b"] * 30 + ["c"] * 30
    return pd.DataFrame({"text": [f"t{i}" for i in range(n)], "label": labels})


def test_split_sizes_partition_the_data():
    df = _frame(100)
    s = make_splits(df, SplitConfig(test_size=0.2, val_size=0.2))
    n = len(s["train"]) + len(s["val"]) + len(s["test"])
    assert n == len(df)
    # no row appears in two splits
    texts = set(s["train"]["text"]) | set(s["val"]["text"]) | set(s["test"]["text"])
    assert len(texts) == len(df)


def test_all_classes_present_in_each_split():
    df = _frame(100)
    s = make_splits(df, SplitConfig())
    for part in ("train", "val", "test"):
        assert set(s[part]["label"]) == {"a", "b", "c"}


def test_reproducible_with_same_seed():
    df = _frame(100)
    s1 = make_splits(df, SplitConfig(random_state=42))
    s2 = make_splits(df, SplitConfig(random_state=42))
    assert s1["test"]["text"].tolist() == s2["test"]["text"].tolist()


def test_different_seed_changes_split():
    df = _frame(100)
    s1 = make_splits(df, SplitConfig(random_state=1))
    s2 = make_splits(df, SplitConfig(random_state=2))
    assert s1["test"]["text"].tolist() != s2["test"]["text"].tolist()
