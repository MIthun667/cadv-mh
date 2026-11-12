from __future__ import annotations

import os
import random

import numpy as np
import torch

RESEARCH_SEED = 42


def seed_everything(seed: int, deterministic: bool = True) -> None:
    if seed != RESEARCH_SEED:
        raise ValueError(f"CADV-MH training is restricted to seed {RESEARCH_SEED}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
