"""Pure scheduling helpers (no torch state) so they can be unit-tested exactly."""
from __future__ import annotations


def temperature_for(stage: int, epoch: int, stage2_epochs: int, stage3_epochs: int, initial: float, final: float, fixed: bool = False) -> float:
    """Linear anneal over the combined stage-2 + stage-3 epoch index g = 1..G, ending exactly at `final`.

    Stage 1 has no selector, so it reports `initial`. `fixed` (ablation A7) holds `initial`.
    """
    if fixed or stage < 2:
        return float(initial)
    total = stage2_epochs + stage3_epochs
    index = epoch if stage == 2 else stage2_epochs + epoch
    if not 1 <= index <= total:
        raise ValueError(f"epoch {epoch} outside stage {stage} schedule")
    if total == 1:
        return float(final)
    return float(initial + (final - initial) * (index - 1) / (total - 1))


def accumulation_window(step: int, total_steps: int, accumulation: int) -> tuple[int, bool]:
    """For 1-indexed micro-batch `step` of `total_steps`, return (size of its accumulation window, is_last_in_window).

    Windows are consecutive groups of `accumulation` micro-batches; the final window may be shorter.
    Dividing each micro-batch loss by the window size gives every window an unbiased mean gradient.
    """
    if accumulation < 1 or not 1 <= step <= total_steps:
        raise ValueError("invalid accumulation arguments")
    start = ((step - 1) // accumulation) * accumulation + 1
    size = min(accumulation, total_steps - start + 1)
    return size, step == start + size - 1


def encoder_learning_rate(stage: int, stage1_lr: float, stage3_lr: float) -> float:
    """Stage 1 fine-tunes at `stage1_lr`; stage 3 at `stage3_lr`. Stage 2 has a frozen encoder (value unused)."""
    return stage3_lr if stage == 3 else stage1_lr
