from __future__ import annotations

import hashlib
import html
import re
import unicodedata


_SPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_USER = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
_C1_CONTROLS = re.compile("[\u0080-\u009f]")
# Latin-1 reading of UTF-8 text: lead byte (Ã, Â, â) followed by a continuation-byte character.
_MOJIBAKE = re.compile("[\u00c2\u00c3\u00e2][\u0080-\u00bf\u20ac\u2122\u0153\u017e\u0178\u2018-\u201e]")


def mojibake_rate(texts) -> float:
    """Fraction of raw decoded strings showing the signature of a wrong single-byte decoding of UTF-8."""
    total = hits = 0
    for text in texts:
        total += 1
        hits += bool(_MOJIBAKE.search(str(text)))
    return hits / max(1, total)


def count_c1_controls(texts) -> int:
    return sum(len(_C1_CONTROLS.findall(str(text))) for text in texts)


def clean_original_text(value: object) -> str:
    """Conservative decoding cleanup; preserves meaning-bearing punctuation and case."""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", "")
    # U+0080-U+009F are non-printing C1 controls: in Dataset A they are residue of stripped emoji bytes.
    text = _C1_CONTROLS.sub("", text)
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
