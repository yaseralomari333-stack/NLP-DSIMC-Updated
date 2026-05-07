"""
DAL.results_viz
---------------
Plots used after training / evaluation:

    plot_confusion(cm, classes, path)
    plot_comparison_bars(rows, path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .data_loader import ARTIFACTS_DIR

PLOTS_DIR = ARTIFACTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})
    return plt


# ---------------------------------------------------------------------------
def plot_confusion(
    cm: np.ndarray,
    classes: Sequence[str],
    path: str | Path,
    *,
    title: str | None = None,
    cmap: str = "Blues",
) -> Path:
    plt = _mpl()
    classes = [str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)

    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)
    fig.colorbar(im)
    fig.tight_layout()

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
def plot_comparison_bars(rows: list[dict], path: str | Path) -> Path:
    """
    rows = [
        {"model": "svm", "precision_macro": .., "recall_macro": ..,
         "f1_macro": .., "accuracy": ..},
        ...
    ]
    """
    plt = _mpl()
    df = pd.DataFrame(rows).set_index("model")
    cols = [c for c in
            ["precision_macro", "recall_macro", "f1_macro", "accuracy"]
            if c in df.columns]
    df = df[cols]

    fig, ax = plt.subplots(figsize=(9, 5))
    df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Models comparison – macro metrics")
    ax.set_xticklabels(df.index, rotation=20, ha="right")
    ax.legend(loc="lower right")

    # numeric labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p
