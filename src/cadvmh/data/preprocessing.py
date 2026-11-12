from __future__ import annotations

import hashlib
import html
import re
import unicodedata


_SPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_USER = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")


def clean_original_text(value: object) -> str:
    """Conservative decoding cleanup; preserves meaning-bearing punctuation and case."""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", "")
    return text.strip()


def canonicalize_for_duplicates(text: str) -> str:
    """Canonical form used only for grouping, never as transformer input."""
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    text = _URL.sub(" <url> ", text)
    text = _USER.sub(" <user> ", text)
    return _SPACE.sub(" ", text)


def stable_text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def conservative_augment(text: str) -> str:
    text = _URL.sub(" <URL> ", text)
    text = _USER.sub(" <USER> ", text)
    return _SPACE.sub(" ", text).strip()
