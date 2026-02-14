from __future__ import annotations
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass(frozen=True)
class FeatureConfig:
    word_ngram_max: int = 2
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    max_features: int = 60000

def make_word_vectorizer(cfg: FeatureConfig) -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, ngram_range=(1, cfg.word_ngram_max), max_features=cfg.max_features)

def make_char_vectorizer(cfg: FeatureConfig) -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, analyzer="char", ngram_range=(cfg.char_ngram_min, cfg.char_ngram_max), max_features=cfg.max_features)
