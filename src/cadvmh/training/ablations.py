"""Single source of truth for ablation definitions (A0-A8).

Each spec states which components are active. `train.py` consumes these specs and
`configs/ablations/*.yaml` are generated from / tested against them, so a config file cannot
claim a component that the implementation does not run.
"""
from __future__ import annotations

from dataclasses import dataclass

LOSS_TERMS = ("necessity", "sufficiency", "sparsity", "continuity", "consistency")
# Terms that vary across ablations under the repaired (exact-budget) selector. "sparsity" is structural there
# (weight 0, see utils/config.py), so it is neither added nor removed by any ablation.
ABLATED_TERMS = ("necessity", "sufficiency", "continuity", "consistency")


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    stages: tuple[int, ...]
    fusion_mode: str
    active_losses: tuple[str, ...]
    fixed_temperature: bool = False
    matched_encoder_epochs: bool = False  # A0: train stage 1 for stage1+stage3 epochs
    alias_of_canonical_run: bool = False  # A8: identical to the canonical full run

    @property
    def loss_overrides(self) -> dict[str, float]:
        """Weights forced to zero; active terms keep their base-config weight."""
        return {term: 0.0 for term in ABLATED_TERMS if term not in self.active_losses}


_SPECS = [
    AblationSpec("A0", "Original-view encoder classifier (matched encoder epochs)", (1,), "full", (), matched_encoder_epochs=True),
    AblationSpec("A1", "Learned selector with evidence-only prediction", (1, 2, 3), "evidence", ("sufficiency",)),
    AblationSpec("A2", "Counterfactual branch with mean fusion", (1, 2, 3), "mean", ()),
    AblationSpec("A3", "A2 plus necessity supervision", (1, 2, 3), "mean", ("necessity",)),
    AblationSpec("A4", "A3 plus evidence-only sufficiency supervision", (1, 2, 3), "mean", ("necessity", "sufficiency")),
    AblationSpec("A5", "A4 plus full context-effect interaction", (1, 2, 3), "full", ("necessity", "sufficiency")),
    AblationSpec("A6", "A5 plus continuity regularization", (1, 2, 3), "full", ("necessity", "sufficiency", "continuity")),
    AblationSpec("A7", "A6 plus consistency at fixed temperature", (1, 2, 3), "full", ABLATED_TERMS, fixed_temperature=True),
    AblationSpec("A8", "Complete model with temperature annealing (alias of canonical run)", (1, 2, 3), "full", ABLATED_TERMS, alias_of_canonical_run=True),
]
ABLATIONS: dict[str, AblationSpec] = {spec.name: spec for spec in _SPECS}


def evidence_required(weights_sufficiency: float, fusion_mode: str) -> bool:
    """The evidence-only branch runs only when it is trained on or used for the prediction."""
    return bool(weights_sufficiency) or fusion_mode == "evidence"


def spec_to_config(spec: AblationSpec) -> dict:
    """Plain-dict form written to configs/ablations/*.yaml and compared against it in tests."""
    return {
        "name": spec.name,
        "description": spec.description,
        "stages": list(spec.stages),
        "fusion_mode": spec.fusion_mode,
        "active_losses": list(spec.active_losses),
        "fixed_temperature": spec.fixed_temperature,
        "matched_encoder_epochs": spec.matched_encoder_epochs,
        "alias_of_canonical_run": spec.alias_of_canonical_run,
    }
