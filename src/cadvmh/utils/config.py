from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("inherits", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        if not parent_path.exists():
            parent_path = (path.parent.parent / parent).resolve()
        config = _merge(load_config(parent_path), config)
    validate_config(config)
    config["_config_path"] = str(path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if "dataset" in config:
        required = {"name", "path", "text_column", "label_column", "labels"}
        missing = required - config["dataset"].keys()
        if missing:
            raise ValueError(f"Missing dataset configuration keys: {sorted(missing)}")
        labels = config["dataset"]["labels"]
        if len(labels) != len(set(labels)) or len(labels) < 2:
            raise ValueError("Dataset labels must contain at least two unique values")
    fractions = config.get("data", {})
    if fractions:
        total = sum(float(fractions[k]) for k in ("train_fraction", "validation_fraction", "test_fraction"))
        if abs(total - 1.0) > 1e-8:
            raise ValueError("Train, validation, and test fractions must sum to 1")
