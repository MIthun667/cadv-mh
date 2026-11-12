from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ValidationSummary:
    rows_read: int
    rows_valid: int
    missing_text: int
    empty_text: int
    invalid_labels: dict[str, int]


def normalize_labels(series: pd.Series, aliases: dict[str, str]) -> pd.Series:
    return series.map(lambda value: aliases.get(str(value), str(value).strip().lower()))


def validate_frame(
    frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    labels: list[str],
    aliases: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, ValidationSummary]:
    missing_columns = {text_column, label_column} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    aliases = aliases or {}
    result = frame.copy()
    missing_text = int(result[text_column].isna().sum())
    nonnull = result[text_column].notna()
    empty = nonnull & result[text_column].astype(str).str.strip().eq("")
    empty_text = int(empty.sum())
    result[label_column] = normalize_labels(result[label_column], aliases)
    invalid = result.loc[~result[label_column].isin(labels), label_column].value_counts(dropna=False)
    invalid_labels = {str(k): int(v) for k, v in invalid.items()}
    if invalid_labels:
        raise ValueError(f"Invalid labels: {invalid_labels}")
    result = result.loc[nonnull & ~empty].copy()
    return result, ValidationSummary(len(frame), len(result), missing_text, empty_text, invalid_labels)
