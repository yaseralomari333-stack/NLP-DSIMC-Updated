"""
Models.MARBERTOps
-----------------
All operations around the MARBERT (or any HF Arabic encoder):

    * fine-tune with weighted CrossEntropy + early stopping
    * evaluate on a held-out test set
    * predict + return softmax confidence (used by the human-in-the-loop layer)
    * save / load the trained model directory
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from DAL.results_viz import plot_confusion
from .MARBERTModel import DEFAULT_MARBERT, build_marbert


# ---------------------------------------------------------------------------
@dataclass
class MARBERTConfig:
    model_name:    str   = DEFAULT_MARBERT
    max_length:    int   = 256
    batch_size:    int   = 8
    epochs:        int   = 4
    lr:            float = 1.5e-5
    weight_decay:  float = 0.01
    fp16:          bool  = False           # auto-disabled if no CUDA
    seed:          int   = 42
    early_stopping_patience: int = 2
    metric_for_best_model:   str = "f1_macro"


# ---------------------------------------------------------------------------
def _compute_metrics_factory():
    def _compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy":         float((preds == labels).mean()),
            "f1_macro":         f1_score(labels, preds, average="macro"),
            "f1_weighted":      f1_score(labels, preds, average="weighted"),
            "precision_macro":  precision_score(labels, preds, average="macro",
                                                zero_division=0),
            "recall_macro":     recall_score(labels, preds, average="macro",
                                             zero_division=0),
        }
    return _compute


# ---------------------------------------------------------------------------
class MARBERTOps:
    def __init__(self, config: MARBERTConfig | None = None):
        self.cfg = config or MARBERTConfig()
        self.tokenizer = None
        self.model     = None
        self.classes:  list[str] = []
        self._artifacts_dir: Path | None = None

    # ====================== TRAIN ==========================================
    def train(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        classes:  list[str],
        artifacts_dir: Path,
    ) -> None:
        """`*_df` must contain  text_clean  +  label_id  columns."""
        try:
            import torch
            from datasets import Dataset
            from transformers import (
                DataCollatorWithPadding,
                EarlyStoppingCallback,
                Trainer,
                TrainingArguments,
            )
        except ImportError as e:                              # pragma: no cover
            raise ImportError(
                f"HF stack missing: {e}. "
                "pip install 'transformers>=4.36' datasets accelerate torch"
            ) from e

        self.classes = classes
        self._artifacts_dir = Path(artifacts_dir)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        out_dir = self._artifacts_dir / "transformer"

        # ---- model + tokenizer
        self.tokenizer, self.model = build_marbert(
            num_labels=len(classes), model_name=self.cfg.model_name
        )

        # ---- HF datasets
        def _to_hf(d: pd.DataFrame):
            return Dataset.from_pandas(
                d[["text_clean", "label_id"]].rename(
                    columns={"text_clean": "text", "label_id": "labels"}
                ),
                preserve_index=False,
            )

        def _tok(batch):
            return self.tokenizer(
                batch["text"], truncation=True, max_length=self.cfg.max_length
            )

        ds_train = _to_hf(train_df).map(_tok, batched=True,
                                        remove_columns=["text"])
        ds_val   = _to_hf(val_df).map(_tok, batched=True,
                                      remove_columns=["text"])

        # ---- weighted loss
        counts = train_df["label_id"].value_counts().sort_index().values
        class_weights = torch.tensor(
            counts.sum() / (len(classes) * counts), dtype=torch.float
        )

        class _WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs,
                             return_outputs=False, **kw):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss = torch.nn.CrossEntropyLoss(
                    weight=class_weights.to(logits.device)
                )(logits, labels)
                return (loss, outputs) if return_outputs else loss

        # ---- training args
        has_cuda = torch.cuda.is_available()
        # training_args = TrainingArguments(
        #     output_dir=str(out_dir),
        #     learning_rate=self.cfg.lr,
        #     per_device_train_batch_size=self.cfg.batch_size,
        #     per_device_eval_batch_size=self.cfg.batch_size * 2,
        #     num_train_epochs=self.cfg.epochs,
        #     weight_decay=self.cfg.weight_decay,
        #     eval_strategy="epoch",
        #     save_strategy="epoch",
        #     load_best_model_at_end=True,
        #     metric_for_best_model=self.cfg.metric_for_best_model,
        #     greater_is_better=True,
        #     save_total_limit=2,
        #     logging_steps=50,
        #     report_to="none",
        #     fp16=self.cfg.fp16 and has_cuda,
        #     seed=self.cfg.seed,
        # )
        
        training_args = TrainingArguments(
        output_dir=str(out_dir),

        # learning
        learning_rate=self.cfg.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",

        # batches
        per_device_train_batch_size=self.cfg.batch_size,
        per_device_eval_batch_size=self.cfg.batch_size * 2,

        # epochs
        num_train_epochs=self.cfg.epochs,

        # regularization
        weight_decay=self.cfg.weight_decay,

        # evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,

        # best model selection
        metric_for_best_model=self.cfg.metric_for_best_model,
        greater_is_better=True,

        # checkpoints
        save_total_limit=2,

        # logging
        logging_steps=50,
        report_to="none",

        # performance
        fp16=self.cfg.fp16 and has_cuda,

        # reproducibility
        seed=self.cfg.seed,
    )

        self._trainer = _WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=_compute_metrics_factory(),
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.cfg.early_stopping_patience
            )],
        )
        self._trainer.train()

    # ====================== EVAL ===========================================
    def evaluate(
        self,
        test_df: pd.DataFrame,
        artifacts_dir: Path | None = None,
    ) -> dict:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Train or load the model before evaluate().")

        try:
            from datasets import Dataset
        except ImportError as e:                              # pragma: no cover
            raise ImportError(f"`datasets` missing: {e}") from e

        artifacts_dir = Path(artifacts_dir or self._artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        def _tok(batch):
            return self.tokenizer(
                batch["text"], truncation=True, max_length=self.cfg.max_length
            )

        ds_test = Dataset.from_pandas(
            test_df[["text_clean", "label_id"]].rename(
                columns={"text_clean": "text", "label_id": "labels"}
            ),
            preserve_index=False,
        ).map(_tok, batched=True, remove_columns=["text"])

        preds_out = self._trainer.predict(ds_test)
        preds  = preds_out.predictions.argmax(axis=-1)
        labels = preds_out.label_ids

        metrics = {
            "model":            self.cfg.model_name,
            "accuracy":         float((preds == labels).mean()),
            "precision_macro":  precision_score(labels, preds, average="macro",
                                                zero_division=0),
            "recall_macro":     recall_score(labels, preds, average="macro",
                                             zero_division=0),
            "f1_macro":         f1_score(labels, preds, average="macro"),
            "f1_weighted":      f1_score(labels, preds, average="weighted"),
            "n_test":           int(len(labels)),
        }

        report = classification_report(labels, preds, target_names=self.classes,
                                       digits=4, zero_division=0)
        cm = confusion_matrix(labels, preds)

        (artifacts_dir / "transformer_report.txt").write_text(
            f"=== {self.cfg.model_name} ===\n{report}\n\nConfusion:\n{cm}\n",
            encoding="utf-8",
        )
        (artifacts_dir / "transformer_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        plot_confusion(
            cm, self.classes,
            artifacts_dir / "plots" / "confusion_marbert.png",
            title="Confusion – MARBERT", cmap="Greens",
        )
        return metrics

    # ====================== PREDICT ========================================
    def predict_with_confidence(
        self, texts: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Train or load the model before predict().")

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device).eval()

        labels: list[int] = []
        conf:   list[float] = []
        bs = self.cfg.batch_size

        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = list(texts[i : i + bs])
                enc = self.tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True,
                    max_length=self.cfg.max_length, 
                    return_tensors="pt",
                ).to(device)
                logits = self.model(**enc).logits
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
                ids    = probs.argmax(axis=1)
                labels.extend(ids.tolist())
                conf.extend(probs.max(axis=1).tolist())
        return np.array(labels), np.array(conf)

    # ====================== IO =============================================
    def save(self, path: str | Path) -> Path:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Nothing to save.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        (path / "labels.json").write_text(
            json.dumps(self.classes, ensure_ascii=False), encoding="utf-8"
        )
        (path / "config_runtime.json").write_text(
            json.dumps(self.cfg.__dict__, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MARBERTOps":
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        path = Path(path)
        cfg_dict = {}
        cfg_path = path / "config_runtime.json"
        if cfg_path.exists():
            cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg = MARBERTConfig(**{k: v for k, v in cfg_dict.items()
                               if k in MARBERTConfig.__dataclass_fields__})

        obj = cls(cfg)
        obj.tokenizer = AutoTokenizer.from_pretrained(str(path))
        obj.model     = AutoModelForSequenceClassification.from_pretrained(str(path))
        labels_path = path / "labels.json"
        obj.classes = (json.loads(labels_path.read_text(encoding="utf-8"))
                       if labels_path.exists() else [])
        return obj
