"""
DAL  --  Data Access Layer
==========================
Everything that touches the dataset lives here:

    data_loader.py   -- read .xlsx / .csv from local path or Google Drive
    preprocessing.py -- Arabic text cleaning + normalization
    eda.py           -- exploratory analysis + dataset plots
    results_viz.py   -- result plots (confusion matrix, model comparison)
"""

from .data_loader   import load_dataset, prepare_splits, ARTIFACTS_DIR
from .preprocessing import clean_text, batch_clean, ARABIC_STOPWORDS
from .eda           import run_eda
from .results_viz   import plot_confusion, plot_comparison_bars

__all__ = [
    "load_dataset",
    "prepare_splits",
    "ARTIFACTS_DIR",
    "clean_text",
    "batch_clean",
    "ARABIC_STOPWORDS",
    "run_eda",
    "plot_confusion",
    "plot_comparison_bars",
]
