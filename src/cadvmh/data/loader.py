from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .preprocessing import clean_original_text, stable_text_id
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


def load_dataset(dataset: dict[str, Any], encoding: str = "utf-8") -> tuple[pd.DataFrame, ValidationSummary]:
    frame = read_table(dataset["path"], encoding=encoding)
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
    return result.reset_index(drop=True), summary
