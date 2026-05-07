"""
DAL.augmentation
----------------
Contextual augmentation for Arabic suicidal tweets.

Strategy:
- Replace only emotionally neutral words
- Preserve suicidal / mental-health keywords
"""

from __future__ import annotations

import random
import re

import nlpaug.augmenter.word as naw


# ---------------------------------------------------------
# كلمات ممنوع تتغير
# ---------------------------------------------------------
PROTECTED_WORDS = {
    "اموت",
    "الموت",
    "انتحر",
    "انتحار",
    "اقتل",
    "انهي",
    "حياتي",
    "تعبت",
    "يائس",
    "حزين",
    "مكتئب",
    "بموت",
    "اذبح",
    "اكتئاب",
}

# ---------------------------------------------------------
# AraBERT contextual augmenter
# ---------------------------------------------------------
aug = naw.ContextualWordEmbsAug(
    model_path="aubmindlab/bert-base-arabertv2",
    action="substitute",
    device="cuda",   # غيّرها cpu إذا ما عندك GPU
)


# ---------------------------------------------------------
def contextual_augment(text: str, aug_p: float = 0.15) -> str:
    """
    Contextual augmentation while preserving suicidal keywords.
    """

    if not isinstance(text, str) or not text.strip():
        return text

    words = text.split()

    # كلمات مسموح تتغير
    candidate_positions = [
        i for i, w in enumerate(words)
        if w not in PROTECTED_WORDS and len(w) > 2
    ]

    if not candidate_positions:
        return text

    n_changes = max(1, int(len(candidate_positions) * aug_p))
    positions = random.sample(
        candidate_positions,
        min(n_changes, len(candidate_positions))
    )

    temp_words = words[:]

    for pos in positions:
        original = temp_words[pos]

        try:
            augmented = aug.augment(original)

            if isinstance(augmented, list):
                augmented = augmented[0]

            # حماية إضافية
            if (
                augmented
                and augmented not in PROTECTED_WORDS
                and len(augmented) > 1
            ):
                temp_words[pos] = augmented

        except Exception:
            continue

    return " ".join(temp_words)