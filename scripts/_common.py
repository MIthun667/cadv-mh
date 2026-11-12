from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def resolve_dataset_path(config: dict) -> Path:
    path = Path(config["dataset"]["path"])
    return path if path.is_absolute() else (ROOT / path).resolve()


def manifest_path(config: dict) -> Path:
    """Immutable split manifest named by the dataset config (v2 manifests match the repaired decoding)."""
    return ROOT / config["dataset"]["manifest"]


def run_root(config: dict) -> Path:
    """Root for new run artifacts; historical artifacts under experiments/ are never written to."""
    return ROOT / config["project"]["output_root"]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
