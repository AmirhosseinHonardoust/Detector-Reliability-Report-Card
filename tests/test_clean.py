"""Unit tests for src.clean (named-column detection, normalization, guards)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.clean import clean_df


def test_named_columns_renamed_and_normalized():
    df = pd.DataFrame({"text": [" Hello ", "World"], "label": [" AI ", "Human"]})
    out = clean_df(df)
    assert list(out.columns[:2]) == ["text", "label"]
    assert out["text"].tolist() == ["Hello", "World"]
    # labels are stripped + lowercased
    assert out["label"].tolist() == ["ai", "human"]


def test_label_synonym_column_detected():
    df = pd.DataFrame({"text": ["a", "b"], "human_or_ai": ["ai", "human"]})
    out = clean_df(df)
    assert "label" in out.columns
    assert set(out["label"]) == {"ai", "human"}


def test_empty_and_whitespace_text_rows_dropped():
    df = pd.DataFrame({"text": ["keep", "   ", ""], "label": ["ai", "human", "ai"]})
    out = clean_df(df)
    assert out["text"].tolist() == ["keep"]
    # index is reset after dropping
    assert out.index.tolist() == [0]


def test_missing_text_column_raises():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        clean_df(df)


def test_fallback_text_column_is_longest_string_column():
    # no column named text/content/sentence -> pick the longest string column
    df = pd.DataFrame(
        {
            "message": ["a fairly long sentence here", "another long sentence too"],
            "category": ["ai", "human"],
        }
    )
    out = clean_df(df)
    assert out["text"].tolist() == [
        "a fairly long sentence here",
        "another long sentence too",
    ]
    assert set(out["label"]) == {"ai", "human"}


def test_fallback_label_column_by_cardinality():
    # label detected by 2..6 unique values among string columns
    df = pd.DataFrame(
        {
            "body": ["some long text one", "some long text two", "some long text three"],
            "kind": ["ai", "human", "ai"],
        }
    )
    out = clean_df(df)
    assert set(out["label"]) == {"ai", "human"}


def test_input_not_mutated():
    df = pd.DataFrame({"text": [" Hello "], "label": [" AI "]})
    before = df.copy(deep=True)
    clean_df(df)
    pd.testing.assert_frame_equal(df, before)
