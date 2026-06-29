from __future__ import annotations

import pandas as pd


def _is_text_like(s: pd.Series) -> bool:
    # Catches object dtype (pandas 2.x) and the "str"/StringDtype columns
    # that pandas 3.x infers, while excluding numeric/bool/datetime columns.
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # find text column
    text_col = None
    for c in out.columns:
        if c.lower() in {"text", "content", "sentence"}:
            text_col = c
            break
    if text_col is None:
        obj_cols = [c for c in out.columns if _is_text_like(out[c])]
        if not obj_cols:
            raise ValueError("No obvious text column found.")
        lengths = {c: out[c].astype(str).str.len().mean() for c in obj_cols}
        text_col = max(lengths, key=lambda c: lengths[c])

    # find label column
    label_col = None
    for c in out.columns:
        if c.lower() in {"label", "class", "human_or_ai", "target"}:
            label_col = c
            break
    if label_col is None:
        for c in out.columns:
            if c == text_col:
                continue
            if _is_text_like(out[c]):
                nun = out[c].nunique(dropna=True)
                if 2 <= nun <= 6:
                    label_col = c
                    break
    if label_col is None:
        raise ValueError("No obvious label column found.")

    out = out.rename(columns={text_col: "text", label_col: "label"})
    out["text"] = out["text"].astype(str).fillna("").str.strip()
    out["label"] = out["label"].astype(str).str.strip().str.lower()

    out = out[out["text"].str.len() > 0].copy()
    return out.reset_index(drop=True)
