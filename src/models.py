from __future__ import annotations
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from src.features import FeatureConfig, make_word_vectorizer, make_char_vectorizer

@dataclass(frozen=True)
class ModelConfig:
    C: float = 3.0
    max_iter: int = 2000
    calibrate: bool = True
    calibration_method: str = "sigmoid"  # sigmoid or isotonic

def build_word_model(fcfg: FeatureConfig, mcfg: ModelConfig):
    base = Pipeline([
        ("tfidf", make_word_vectorizer(fcfg)),
        ("clf", LogisticRegression(C=mcfg.C, max_iter=mcfg.max_iter))
    ])
    return CalibratedClassifierCV(base, method=mcfg.calibration_method, cv=3) if mcfg.calibrate else base

def build_char_model(fcfg: FeatureConfig, mcfg: ModelConfig):
    base = Pipeline([
        ("tfidf", make_char_vectorizer(fcfg)),
        ("clf", LogisticRegression(C=mcfg.C, max_iter=mcfg.max_iter))
    ])
    return CalibratedClassifierCV(base, method=mcfg.calibration_method, cv=3) if mcfg.calibrate else base
