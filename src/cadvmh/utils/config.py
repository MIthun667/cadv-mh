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


# Every key below is consumed by the implementation. Unknown keys are rejected so a config can never
# describe behaviour that the code does not have.
KNOWN_KEYS: dict[str, set[str]] = {
    "project": {"name", "output_root", "seed"},
    "data": {"train_fraction", "validation_fraction", "test_fraction", "split_seed"},
    "model": {"encoder_name", "max_length", "selector_hidden_dim", "dropout", "interaction_dim", "selection_mode",
              "selection_ratio", "temperature_initial", "temperature_final", "gradient_checkpointing"},
    "training": {"microbatch_size", "gradient_accumulation", "encoder_learning_rate", "stage3_encoder_learning_rate",
                 "selector_learning_rate", "module_learning_rate", "weight_decay", "gradient_clip", "fp16", "patience",
                 "beta", "stage1_epochs", "stage2_epochs", "stage3_epochs"},
    "loss": {"classification", "original", "necessity", "sufficiency", "sparsity", "continuity", "consistency",
             "necessity_margin", "necessity_form", "necessity_margin_log"},
    "dataset": {"name", "path", "text_column", "label_column", "id_column", "labels", "label_aliases", "encoding",
                "manifest"},
    "transfer": {"source", "target", "direction", "labels", "dataset_a_mapping"},
    "baseline": {"kind", "encoder_name", "max_features", "ngram_range"},
}
NECESSITY_FORMS = {"absolute", "contrastive_prob", "contrastive_logprob"}
SELECTION_MODES = {"budgeted_st", "budgeted_st_sigmoid", "soft", "straight_through"}
# The splitter is a 5-fold outer x 8-fold inner StratifiedGroupKFold, i.e. 70/10/20 by construction.
IMPLEMENTED_FRACTIONS = {"train_fraction": 0.70, "validation_fraction": 0.10, "test_fraction": 0.20}


def validate_config(config: dict[str, Any]) -> None:
    for section, values in config.items():
        if section.startswith("_") or section == "ablation":
            continue
        if section not in KNOWN_KEYS:
            raise ValueError(f"Unknown configuration section: {section!r}")
        unknown = set(values) - KNOWN_KEYS[section]
        if unknown:
            raise ValueError(f"Unknown keys in [{section}] (not consumed by the implementation): {sorted(unknown)}")
    mode = config.get("model", {}).get("selection_mode")
    if mode is not None and mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {sorted(SELECTION_MODES)}")
    if mode in {"budgeted_st", "budgeted_st_sigmoid"} and float(config.get("loss", {}).get("sparsity", 0.0)) != 0.0:
        raise ValueError("loss.sparsity must be 0.0 with selection_mode=budgeted_st: the budget is enforced exactly, "
                         "so a sparsity penalty is inert at best and harmful to the straight-through gradient at worst")
    form = config.get("loss", {}).get("necessity_form")
    if form is not None and form not in NECESSITY_FORMS:
        raise ValueError(f"necessity_form must be one of {sorted(NECESSITY_FORMS)}")
    for key, expected in IMPLEMENTED_FRACTIONS.items():
        if key in config.get("data", {}) and abs(float(config["data"][key]) - expected) > 1e-8:
            raise ValueError(f"{key} must be {expected}: the group splitter implements exactly 70/10/20")
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
