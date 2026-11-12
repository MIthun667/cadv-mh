from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

import torch


def hardware_report() -> dict[str, Any]:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        report.update(
            gpu_name=torch.cuda.get_device_name(0),
            gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            cuda_version=torch.version.cuda,
        )
    return report


def git_report(cwd: str) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"revision": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "status": run("status", "--short")}
