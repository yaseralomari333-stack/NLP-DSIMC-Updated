"""
DAL.eda
-------
Exploratory Data Analysis with plots.

Generated artifacts (under  artifacts/eda/):
    01_class_balance.png        bar chart of class counts
    02_text_length_hist.png     histogram of token-length per class
    03_text_length_box.png      box plot of token-length per class
    04_top_tokens_<class>.png   top-20 most-frequent tokens, one PNG per class
    05_dataset_overview.txt     numeric summary
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader  import ARTIFACTS_DIR, load_dataset
from .preprocessing import batch_clean, ARABIC_STOPWORDS

EDA_DIR = ARTIFACTS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def _mpl():
    """Late import + headless backend so we never crash on a server."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})
    return plt


# ---------------------------------------------------------------------------
def _plot_class_balance(df: pd.DataFrame, label_col: str) -> Path:
    plt = _mpl()
    counts = df[label_col].astype(str).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color="#3b7dd8")
    ax.set_title("Class balance")
    ax.set_ylabel("# samples")
    ax.set_xlabel(label_col)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f"{val:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = EDA_DIR / "01_class_balance.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_length_hist(df: pd.DataFrame, label_col: str) -> Path:
    plt = _mpl()
    df = df.copy()
    df["__len"] = df["text_clean"].str.split().str.len()

    fig, ax = plt.subplots(figsize=(7, 4))
    for cls, sub in df.groupby(label_col):
        if len(sub):
            ax.hist(sub["__len"], bins=40, alpha=0.55, label=str(cls))
    ax.set_title("Token-length distribution per class")
    ax.set_xlabel("# tokens")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    out = EDA_DIR / "02_text_length_hist.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_length_box(df: pd.DataFrame, label_col: str) -> Path:
    plt = _mpl()
    df = df.copy()
    df["__len"] = df["text_clean"].str.split().str.len()

    classes = sorted(df[label_col].astype(str).unique())
    data = [df.loc[df[label_col].astype(str) == c, "__len"].values for c in classes]
    # filter out empty groups (avoids matplotlib dim-mismatch errors)
    pairs = [(c, d) for c, d in zip(classes, data) if len(d)]
    if not pairs:
        return EDA_DIR / "03_text_length_box.png"
    classes, data = [c for c, _ in pairs], [d for _, d in pairs]

    fig, ax = plt.subplots(figsize=(6, 4))
    # `labels=` was renamed to `tick_labels=` in mpl 3.9; pass via positional
    bp = ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(classes) + 1))
    ax.set_xticklabels(classes)
    ax.set_title("Token-length per class (box plot)")
    ax.set_ylabel("# tokens")
    fig.tight_layout()
    out = EDA_DIR / "03_text_length_box.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_top_tokens(df: pd.DataFrame, label_col: str, k: int = 20) -> list[Path]:
    plt = _mpl()
    paths: list[Path] = []

    for cls, sub in df.groupby(label_col):
        tokens: Counter = Counter()
        for txt in sub["text_clean"]:
            for t in str(txt).split():
                if t in ARABIC_STOPWORDS or len(t) < 2:
                    continue
                tokens[t] += 1
        if not tokens:
            continue

        top = tokens.most_common(k)
        words, counts = zip(*top)

        fig, ax = plt.subplots(figsize=(7, 5))
        y_pos = np.arange(len(words))
        ax.barh(y_pos, counts, color="#d8693b")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_title(f"Top {k} tokens – class={cls}")
        ax.set_xlabel("Frequency")
        fig.tight_layout()

        safe = str(cls).replace("/", "_").replace(" ", "_")
        out = EDA_DIR / f"04_top_tokens_{safe}.png"
        fig.savefig(out)
        plt.close(fig)
        paths.append(out)

    return paths


# ---------------------------------------------------------------------------
def _write_overview(df: pd.DataFrame, text_col: str, label_col: str) -> Path:
    counts = df[label_col].astype(str).value_counts()
    pct    = (counts / counts.sum() * 100).round(2)
    dist   = pd.concat([counts.rename("count"), pct.rename("pct")], axis=1)

    lens = df["text_clean"].str.split().str.len()

    text = (
        f"Dataset overview\n================\n"
        f"Rows         : {len(df):,}\n"
        f"Text column  : {text_col!r}\n"
        f"Label column : {label_col!r}\n"
        f"Memory (MB)  : {df.memory_usage(deep=True).sum()/1e6:,.2f}\n\n"
        f"Class distribution\n------------------\n"
        f"{dist.to_string()}\n\n"
        f"Token-length stats\n------------------\n"
        f"{lens.describe(percentiles=[.5,.9,.95,.99]).round(2).to_string()}\n"
    )
    out = EDA_DIR / "05_dataset_overview.txt"
    out.write_text(text, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
def run_eda(
    data_path: str,
    text_col: str | None = None,
    label_col: str | None = None,
) -> dict:
    """End-to-end EDA. Returns a dict of generated artifacts."""
    df, t, l = load_dataset(data_path, text_col, label_col)
    out: dict = {"text_col": t, "label_col": l, "n_rows": len(df), "files": []}

    out["files"].append(str(_write_overview(df, t, l)))
    out["files"].append(str(_plot_class_balance(df, l)))
    out["files"].append(str(_plot_length_hist(df, l)))
    out["files"].append(str(_plot_length_box(df, l)))
    out["files"].extend(str(p) for p in _plot_top_tokens(df, l))
    return out


# Smoke-test
if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-col",  default=None)
    ap.add_argument("--label-col", default=None)
    args = ap.parse_args()

    print(json.dumps(run_eda(args.data, args.text_col, args.label_col),
                     indent=2, ensure_ascii=False))
