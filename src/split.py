from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42

def make_splits(df: pd.DataFrame, cfg: SplitConfig) -> dict:
    y = df["label"]
    strat = y if y.nunique() > 1 else None

    train_val, test = train_test_split(df, test_size=cfg.test_size, random_state=cfg.random_state, stratify=strat)

    y_tv = train_val["label"]
    strat_tv = y_tv if y_tv.nunique() > 1 else None
    val_rel = cfg.val_size / (1.0 - cfg.test_size)

    train, val = train_test_split(train_val, test_size=val_rel, random_state=cfg.random_state, stratify=strat_tv)

    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True), "test": test.reset_index(drop=True)}
