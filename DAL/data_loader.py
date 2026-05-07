"""
DAL.data_loader
---------------
Load the Arabic suicidal-ideation dataset from a local path or a Google
Drive path (when running on Colab the drive is mounted at /content/drive).

Public API
----------
load_dataset(path, text_col=None, label_col=None) -> pd.DataFrame
prepare_splits(df, text_col, label_col, test_size, val_size, seed)
                                       -> (train_df, val_df, test_df, classes)
ARTIFACTS_DIR  -- pathlib.Path to ./artifacts (auto-created)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .preprocessing import batch_clean

# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path(os.environ.get("NLP_ARTIFACTS_DIR", "artifacts")).resolve()
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Heuristic column-name guessers (auto-detect)
TEXT_HINTS  = ["text", "tweet", "content", "message", "post", "نص", "تغريدة"]
LABEL_HINTS = ["label", "class", "target", "y", "category", "تصنيف", "فئة"]


# ---------------------------------------------------------------------------
def _guess_column(cols: list[str], hints: list[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for h in hints:
        for lc, orig in low.items():
            if h in lc:
                return orig
    return None


def _read_any(path: str | Path) -> pd.DataFrame:
    suf = Path(path).suffix.lower()
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf == ".json":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file type: {suf}")


# ---------------------------------------------------------------------------
def load_dataset(
    path: str | Path,
    text_col: str | None = None,
    label_col: str | None = None,
    *,
    drop_duplicates: bool = True,
    clean: bool = True,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Load + (optionally) clean the dataset.

    Returns
    -------
    df       : DataFrame with at least columns [text_col, label_col, "text_clean"]
    text_col : resolved text column name
    label_col: resolved label column name
    """
    if not os.path.exists(path):
        sys.exit(f"Dataset file not found: {path}")

    df = _read_any(path)
    cols = list(df.columns)

    text_col  = text_col  or _guess_column(cols, TEXT_HINTS)
    label_col = label_col or _guess_column(cols, LABEL_HINTS)
    if text_col is None or label_col is None:
        sys.exit(
            f"Could not auto-detect columns "
            f"(text={text_col}, label={label_col}). "
            f"Available: {cols}. Pass --text-col / --label-col explicitly."
        )

    df = df[[text_col, label_col]].dropna()
    if drop_duplicates:
        df = df.drop_duplicates(subset=[text_col])

    if clean:
        df = df.copy()
        df["text_clean"] = batch_clean(df[text_col])
        df = df[df["text_clean"].str.len() > 0]

    df = df.reset_index(drop=True)
    return df, text_col, label_col


# ---------------------------------------------------------------------------
def prepare_splits(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    *,
    test_size: float = 0.2,
    val_size:  float = 0.1,
    seed: int = 42,
    save_label_encoder: bool = True,
):
    """
    Stratified train / val / test split. Persists the LabelEncoder so that
    every model in the project sees the same class-id mapping.
    """
    le = LabelEncoder()
    y  = le.fit_transform(df[label_col].astype(str))
    classes = list(le.classes_)

    if save_label_encoder:
        joblib.dump(le, ARTIFACTS_DIR / "label_encoder.joblib")

    df = df.copy()
    df["label_id"] = y

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=y
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=seed,
        stratify=train_df["label_id"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        classes,
    )


# Smoke-test
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--text-col",  default=None)
    ap.add_argument("--label-col", default=None)
    args = ap.parse_args()

    df, t, l = load_dataset(args.path, args.text_col, args.label_col)
    print(f"Rows={len(df)}  text_col={t!r}  label_col={l!r}")
    print(df.head())
