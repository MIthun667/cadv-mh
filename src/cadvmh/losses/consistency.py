from __future__ import annotations

import torch
import torch.nn.functional as F


def teacher_student_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    target = F.softmax(teacher_logits.detach(), dim=-1)
    log_prediction = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(log_prediction, target, reduction="batchmean")
