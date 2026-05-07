"""
Models.BaselineOps
------------------
All operations for the classical-ML baselines: train, evaluate, predict,
save / load.

Usage from RUN.py:

    ops = BaselineOps(model_key="svm")
    ops.train(X_train, y_train)
    metrics = ops.evaluate(X_test, y_test, classes)
    ops.save(ARTIFACTS_DIR / "svm.joblib")

    # later, for inference:
    ops = BaselineOps.load(ARTIFACTS_DIR / "svm.joblib")
    labels, conf = ops.predict_with_confidence(["تغريدة جديدة"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from DAL.results_viz import plot_confusion
from .BaselineModel import build_baseline_pipeline


class BaselineOps:
    """Train / evaluate / persist a single TF-IDF + classifier pipeline."""

    def __init__(self, model_key: str = "svm"):
        self.model_key = model_key
        self.pipe = build_baseline_pipeline(model_key)

    # ---------------------- training ---------------------------------------
    def train(self, X_train: Sequence[str], y_train: np.ndarray) -> None:
        self.pipe.fit(X_train, y_train)

    # ---------------------- evaluation -------------------------------------
    def evaluate(
        self,
        X_test: Sequence[str],
        y_test: np.ndarray,
        classes: list[str],
        artifacts_dir: Path,
    ) -> dict:
        pred = self.pipe.predict(X_test)
        metrics = {
            "model":            self.model_key,
            "accuracy":         float((pred == y_test).mean()),
            "precision_macro":  precision_score(y_test, pred, average="macro",
                                                zero_division=0),
            "recall_macro":     recall_score(y_test, pred, average="macro",
                                             zero_division=0),
            "f1_macro":         f1_score(y_test, pred, average="macro"),
            "f1_weighted":      f1_score(y_test, pred, average="weighted"),
        }
        report = classification_report(y_test, pred, target_names=classes,
                                       digits=4, zero_division=0)
        cm = confusion_matrix(y_test, pred)

        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        (artifacts_dir / f"baseline_report_{self.model_key}.txt").write_text(
            f"=== {self.model_key.upper()} ===\n{report}\n\nConfusion:\n{cm}\n",
            encoding="utf-8",
        )
        (artifacts_dir / f"baseline_metrics_{self.model_key}.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        plot_confusion(
            cm, classes,
            artifacts_dir / "plots" / f"confusion_{self.model_key}.png",
            title=f"Confusion – {self.model_key}",
        )
        return metrics

    # ---------------------- prediction -------------------------------------
    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipe.predict(texts)

    def predict_with_confidence(
        self, texts: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        ids = self.pipe.predict(texts)

        if hasattr(self.pipe, "predict_proba"):
            proba = self.pipe.predict_proba(texts)
            conf  = proba.max(axis=1)
        else:
            scores = self.pipe.decision_function(texts)
            if scores.ndim == 1:
                conf = 1 / (1 + np.exp(-np.abs(scores)))
            else:
                conf = 1 / (1 + np.exp(-scores.max(axis=1)))
        return ids, conf

    # ---------------------- IO ---------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_key": self.model_key, "pipe": self.pipe}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaselineOps":
        blob = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model_key = blob["model_key"]
        obj.pipe      = blob["pipe"]
        return obj
