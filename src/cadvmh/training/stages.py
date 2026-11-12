from __future__ import annotations

from torch import nn


def _set(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def configure_stage(model: nn.Module, stage: int, train_upper_encoder_layers: int = 0) -> dict[str, int]:
    if stage not in {1, 2, 3}:
        raise ValueError("stage must be 1, 2, or 3")
    _set(model, False)
    if stage == 1:
        _set(model.encoder, True)
        _set(model.pooling, True)
        _set(model.auxiliary_head, True)
    elif stage == 2:
        _set(model.selector, True)
        if train_upper_encoder_layers:
            layers = getattr(getattr(model.encoder.model, "encoder", None), "layer", [])
            for layer in list(layers)[-train_upper_encoder_layers:]:
                _set(layer, True)
    else:
        _set(model, True)
    return {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
