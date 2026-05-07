"""
Models.BaselineModel
--------------------
Architecture / pipeline definitions for the classical-ML baselines.

Pipelines:
    TF-IDF (word, 1-2 grams)  ->  classifier

Available classifiers (key -> factory):
    "svm"    : LinearSVC         (class_weight=balanced)
    "logreg" : LogisticRegression(class_weight=balanced)
    "rf"     : RandomForestClassifier(class_weight=balanced)
"""

from __future__ import annotations

from typing import Callable, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


BASELINE_MODELS: Dict[str, Callable[[], object]] = {
    "svm":    lambda: LinearSVC(C=1.0, class_weight="balanced", max_iter=4000),
    "logreg": lambda: LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", n_jobs=-1
    ),
    "rf":     lambda: RandomForestClassifier(
        n_estimators=400, n_jobs=-1, class_weight="balanced", random_state=42
    ),
}


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        analyzer="word",
    )


def build_baseline_pipeline(model_key: str) -> Pipeline:
    if model_key not in BASELINE_MODELS:
        raise ValueError(
            f"Unknown baseline {model_key!r}. "
            f"Choose one of {list(BASELINE_MODELS)}"
        )
    return Pipeline([
        ("tfidf", _tfidf()),
        ("clf",   BASELINE_MODELS[model_key]()),
    ])
