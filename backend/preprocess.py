"""
preprocess.py
Text cleaning utilities for the Fake Medicine / Health Claim Detector.

This is the single source of truth for text preprocessing. Both the
training script and the Flask inference API should import from here,
so training and serving never drift out of sync.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_data() -> None:
    """Download required NLTK corpora if not already present."""
    for pkg in ("stopwords", "punkt", "wordnet"):
        try:
            nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

_NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def preprocess_text(text: str) -> str:
    """
    Clean a single piece of text for the model.

    Steps: lowercase -> strip punctuation/numbers -> tokenize ->
    remove stopwords -> lemmatize.

    Returns an empty string (not an error) for empty/non-string input,
    so downstream code can decide how to handle it.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    tokens = text.split()
    tokens = [_lemmatizer.lemmatize(w) for w in tokens if w not in _stop_words]
    return " ".join(tokens)


def preprocess_batch(texts) -> list:
    """Vectorized-friendly helper for a pandas Series or list of strings."""
    return [preprocess_text(t) for t in texts]