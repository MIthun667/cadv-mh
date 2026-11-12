from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def group_stratified_split(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    if frame["group_id"].duplicated().any() and frame.groupby("group_id")["label"].nunique().gt(1).any():
        raise ValueError("Conflicting-label groups must be excluded or resolved before splitting")
    work = frame.reset_index(drop=True).copy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(work, work["label"], groups=work["group_id"]))
    train_val = work.iloc[train_val_idx].reset_index().rename(columns={"index": "original_index"})
    inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed + 1)
    train_sub, val_sub = next(inner.split(train_val, train_val["label"], groups=train_val["group_id"]))
    work["partition"] = ""
    work.loc[test_idx, "partition"] = "test"
    work.loc[train_val.iloc[train_sub]["original_index"], "partition"] = "train"
    work.loc[train_val.iloc[val_sub]["original_index"], "partition"] = "validation"
    assert not (work["partition"] == "").any()
    verify_split_disjointness(work)
    return work


def verify_split_disjointness(manifest: pd.DataFrame) -> None:
    counts = manifest.groupby("group_id")["partition"].nunique()
    if counts.gt(1).any():
        raise AssertionError("Duplicate group leakage detected across partitions")


def dataset_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest(manifest: pd.DataFrame, path: str | Path, metadata: dict[str, object]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable split manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest[["sample_id", "group_id", "partition", "label"]].to_csv(path, index=False)
    meta_path = path.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
