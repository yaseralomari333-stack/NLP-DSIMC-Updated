"""
Models.MARBERTModel
-------------------
Architecture wrapper for the Arabic transformer.

Default checkpoint: UBC-NLP/MARBERT
    Pre-trained on ~6B Arabic tweets (MSA + dialects). Strongest publicly
    available encoder for short, informal Arabic text.

Use a different model by passing `model_name=` (e.g.
"aubmindlab/bert-base-arabertv2" or "asafaya/bert-base-arabic").
"""

from __future__ import annotations

DEFAULT_MARBERT = "UBC-NLP/MARBERTv2"


def build_marbert(num_labels: int, model_name: str = DEFAULT_MARBERT):
    """Return (tokenizer, model) for sequence classification."""
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as e:                                   # pragma: no cover
        raise ImportError(
            f"HF transformers not installed: {e}. "
            "Run: pip install 'transformers>=4.36' datasets accelerate torch"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return tokenizer, model
