"""
Models package
==============

For every model we keep a *pair* of files:

    <Name>Model.py   -- the architecture / pipeline definition
    <Name>Ops.py     -- the training, evaluation, save/load operations

Currently shipped:
    BaselineModel + BaselineOps   (TF-IDF + LinearSVC / LogReg / RandomForest)
    MARBERTModel  + MARBERTOps    (HF transformer, default UBC-NLP/MARBERT)
"""

from .BaselineModel import build_baseline_pipeline, BASELINE_MODELS
from .BaselineOps   import BaselineOps
from .MARBERTModel  import build_marbert
from .MARBERTOps    import MARBERTOps

__all__ = [
    "build_baseline_pipeline",
    "BASELINE_MODELS",
    "BaselineOps",
    "build_marbert",
    "MARBERTOps",
]
