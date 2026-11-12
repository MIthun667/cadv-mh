#!/usr/bin/env python
"""Regenerate configs/ablations/*.yaml from the ablation specs (single source of truth)."""
from __future__ import annotations

import yaml

from _common import ROOT
from cadvmh.training.ablations import ABLATIONS, spec_to_config

NAMES = {"A0": "a0_original", "A1": "a1_selector", "A2": "a2_counterfactual", "A3": "a3_necessity", "A4": "a4_sufficiency",
         "A5": "a5_interaction", "A6": "a6_continuity", "A7": "a7_consistency", "A8": "a8_complete"}

for name, spec in ABLATIONS.items():
    path = ROOT / "configs" / "ablations" / f"{NAMES[name]}.yaml"
    path.write_text("# Generated from src/cadvmh/training/ablations.py; edit the spec, not this file.\n"
                    + yaml.safe_dump({"ablation": spec_to_config(spec)}, sort_keys=False), encoding="utf-8")
