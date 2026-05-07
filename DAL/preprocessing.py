"""
DAL.preprocessing
-----------------
Arabic text cleaning utilities.

Cleaning pipeline (in order):
    1. Strip URLs, mentions (@user), hashtag symbol (keep the word).
    2. Remove HTML entities.
    3. Remove Arabic diacritics (tashkeel) and tatweel.
    4. Normalize letters: alef-variants -> alef, ya-variants -> ya,
                          ta-marbuta -> ha, etc.
    5. Collapse repeated chars (3+ -> 2).
    6. Strip non-Arabic-letter punctuation (digits + whitespace kept).
    7. Collapse whitespace.

Stop-words are kept by default (transformer models need context).
Pass remove_stopwords=True for the TF-IDF baseline.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Pre-compiled regexes  (Arabic chars expressed as \uXXXX for safety)
# ---------------------------------------------------------------------------
_URL_RE      = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE  = re.compile(r"@[\w_]+")
_HASHTAG_SYM = re.compile(r"#")
_HTML_RE     = re.compile(r"&\w+;")

# Arabic diacritics + Quranic marks (NOT the letters themselves):
#   U+0610-U+061A : Arabic marks
#   U+064B-U+065F : harakat
#   U+0670        : superscript alef
#   U+06D6-U+06ED : Quranic annotation signs
_DIACRITICS = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۭ]"
)
_TATWEEL     = re.compile("ـ")
_REPEAT_CHAR = re.compile(r"(.)\1{2,}")

# Keep only:
#   Arabic block          U+0600-U+06FF
#   Arabic Supplement     U+0750-U+077F
#   Arabic Extended-A     U+08A0-U+08FF
#   ASCII digits, whitespace
_NON_ARABIC = re.compile(
    "[^؀-ۿݐ-ݿࢠ-ࣿ0-9\\s]"
)
_MULTI_SPACE = re.compile(r"\s+")

# Arabic stop-word list (small but useful for TF-IDF)
ARABIC_STOPWORDS = {
    "في",                                  # fy
    "من",                                  # mn
    "على",                            # 3la
    "إلى",                            # ila
    "عن",                                  # 3n
    "هذا",                            # h*a
    "هذه",                            # h*h
    "ذلك",                            # *lk
    "تلك",                            # tlk
    "هو",                                  # hw
    "هي",                                  # hy
    "هم",                                  # hm
    "هن",                                  # hn
    "أنا",                            # ana
    "نحن",                            # n7n
    "أنت",                            # ant
    "أنتم",                      # antm
    "ما",                                  # ma
    "لا",                                  # la
    "لم",                                  # lm
    "لن",                                  # ln
    "قد",                                  # qd
    "كان",                            # kan
    "كانت",                      # kant
    "يكون",                      # ykwn
    "ثم",                                  # *m
    "أو",                                  # aw
    "أم",                                  # am
    "إن",                                  # in
    "أن",                                  # an
    "لكن",                            # lkn
    "بل",                                  # bl
    "كما",                            # kma
    "حتى",                            # 7tya
    "مع",                                  # m3
    "بين",                            # byn
    "عند",                            # 3nd
    "كل",                                  # kl
    "بعض",                            # b3d
    "غير",                            # gyr
    "نفس",                            # nfs
}

# Letter-form normalization map  (\uXXXX so encoding is robust)
_NORM_MAP = str.maketrans({
    "أ": "ا",  # alef-hamza-above -> alef
    "إ": "ا",  # alef-hamza-below -> alef
    "آ": "ا",  # alef-madda      -> alef
    "ٱ": "ا",  # alef-wasla      -> alef
    "ى": "ي",  # alef-maksura    -> ya
    "ئ": "ي",  # ya-hamza        -> ya
    "ؤ": "و",  # waw-hamza       -> waw
    "ة": "ه",  # ta-marbuta      -> ha
    "گ": "ك",  # gaf             -> kaf
    "ک": "ك",  # keheh           -> kaf
    "ڤ": "ف",  # veh             -> feh
    "پ": "ب",  # peh             -> beh
    "چ": "ج",  # tcheh           -> jeem
})


def clean_text(text: str, *, remove_stopwords: bool = False) -> str:
    """Clean a single Arabic string."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_SYM.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = text.translate(_NORM_MAP)
    text = _REPEAT_CHAR.sub(r"\1\1", text)
    text = _NON_ARABIC.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()

    if remove_stopwords and text:
        text = " ".join(t for t in text.split() if t not in ARABIC_STOPWORDS)

    return text


def batch_clean(
    series: Iterable[str], *, remove_stopwords: bool = False
) -> pd.Series:
    """Vectorised wrapper for a Series / list / iterable."""
    s = pd.Series(series, dtype="object").fillna("")
    return s.map(lambda x: clean_text(x, remove_stopwords=remove_stopwords))


if __name__ == "__main__":
    sample = (
        "حسّيت "
        "إني "
        "تعبان "
        "@user http://t.co/x #x"
    )
    print("IN :", sample)
    print("OUT:", clean_text(sample))
