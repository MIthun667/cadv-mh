from __future__ import annotations

import torch


def build_optimizer(
    model,
    encoder_lr: float = 1e-5,
    selector_lr: float = 5e-5,
    module_lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    groups = []
    assignments = [
        (model.encoder, encoder_lr, "encoder"),
        (model.selector, selector_lr, "selector"),
        (model.pooling, module_lr, "pooling"),
        (model.auxiliary_head, module_lr, "auxiliary_head"),
        (model.interaction, module_lr, "interaction"),
        (model.final_head, module_lr, "final_head"),
    ]
    seen: set[int] = set()
    for module, lr, name in assignments:
        parameters = [p for p in module.parameters() if p.requires_grad and id(p) not in seen]
        seen.update(id(p) for p in parameters)
        if parameters:
            groups.append({"params": parameters, "lr": lr, "name": name})
    if not groups:
        raise ValueError("No trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)
