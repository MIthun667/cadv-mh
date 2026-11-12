from __future__ import annotations

import os
import random

import numpy as np
import torch

RESEARCH_SEED = 42
CONFIRMATORY_SEEDS = (42, 43, 44, 45, 46)   # authorised for the Dataset B multi-seed confirmation (docs/confirmatory_protocol.md)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    if seed not in CONFIRMATORY_SEEDS:
        raise ValueError(f"CADV-MH training is restricted to seeds {CONFIRMATORY_SEEDS}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
