from __future__ import annotations

import torch
import torch.nn.functional as F


def necessity_loss(
    original_logits: torch.Tensor,
    counterfactual_logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
    detach_original: bool = True,
) -> torch.Tensor:
    original = F.softmax(original_logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
    counterfactual = F.softmax(counterfactual_logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
    if detach_original:
        original = original.detach()
    return F.relu(float(margin) - original + counterfactual).mean()


def contrastive_necessity_loss(
    learned_logits: torch.Tensor,
    random_logits: torch.Tensor,
    labels: torch.Tensor,
    form: str,
    margin: float = 0.2,
    margin_log: float = 0.6931471805599453,
) -> torch.Tensor:
    """Necessity relative to a same-budget random mask (gradients flow through both views).

    contrastive_prob   : relu(margin - (p_rand(y) - p_learned(y)))
    contrastive_logprob: relu(margin_log - (log p_rand(y) - log p_learned(y)))
    A response that lowers the true-class score for *any* mask of this size leaves both terms equal and is not rewarded.
    """
    index = labels[:, None]
    if form == "contrastive_prob":
        learned = F.softmax(learned_logits, dim=-1).gather(1, index).squeeze(1)
        random = F.softmax(random_logits, dim=-1).gather(1, index).squeeze(1)
        return F.relu(float(margin) - (random - learned)).mean()
    if form == "contrastive_logprob":
        learned = F.log_softmax(learned_logits, dim=-1).gather(1, index).squeeze(1)
        random = F.log_softmax(random_logits, dim=-1).gather(1, index).squeeze(1)
        return F.relu(float(margin_log) - (random - learned)).mean()
    raise ValueError(f"Unknown contrastive necessity form: {form}")
