from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .preprocessing import clean_original_text, count_c1_controls, mojibake_rate, stable_text_id
from .validation import ValidationSummary, validate_frame


def read_table(path: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding=encoding)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", encoding=encoding)
    if suffix == ".jsonl":
        with path.open("r", encoding=encoding) as handle:
            return pd.DataFrame(json.loads(line) for line in handle if line.strip())
    raise ValueError(f"Unsupported dataset format: {suffix}")


MOJIBAKE_LIMIT = 0.005


def load_dataset(dataset: dict[str, Any], encoding: str | None = None) -> tuple[pd.DataFrame, ValidationSummary]:
    """Load with an explicit, strict, per-dataset encoding.

    The dataset's own `encoding` key wins over the `encoding` argument (kept for call-site compatibility);
    the default is strict UTF-8. Decoding errors raise. A mojibake guard rejects files whose texts show the
    signature of UTF-8 decoded as a single-byte codec.
    """
    encoding = dataset.get("encoding") or encoding or "utf-8"
    frame = read_table(dataset["path"], encoding=encoding)
    rate = mojibake_rate(frame[dataset["text_column"]].dropna())
    if rate > MOJIBAKE_LIMIT:
        raise ValueError(f"{rate:.1%} of texts look like mojibake under encoding {encoding!r}; the encoding is wrong")
    frame, summary = validate_frame(
        frame,
        dataset["text_column"],
        dataset["label_column"],
        dataset["labels"],
        dataset.get("label_aliases", {}),
    )
    text_col, label_col = dataset["text_column"], dataset["label_column"]
    result = pd.DataFrame({"text": frame[text_col].map(clean_original_text), "label": frame[label_col]})
    id_col = dataset.get("id_column")
    source_ids = frame[id_col] if id_col and id_col in frame else pd.Series(index=frame.index, dtype=object)
    result["source_id"] = source_ids.astype("string").fillna("")
    result["sample_id"] = [f"{dataset['name']}:{idx}:{stable_text_id(text)[:16]}" for idx, text in zip(frame.index, result["text"])]
    result = result.reset_index(drop=True)
    result.attrs.update(encoding=encoding, mojibake_rate=rate, c1_controls_in_raw=count_c1_controls(frame[text_col]))
    return result, summary
