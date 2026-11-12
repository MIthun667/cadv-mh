from __future__ import annotations

import torch.nn.functional as F


def faithfulness_scores(
    original_logits, counterfactual_logits, evidence_logits, labels, selection, eligible
):
    index = labels[:, None]
    original = F.softmax(original_logits, -1).gather(1, index).squeeze(1)
    counterfactual = F.softmax(counterfactual_logits, -1).gather(1, index).squeeze(1)
    evidence = F.softmax(evidence_logits, -1).gather(1, index).squeeze(1)
    ratio = (selection * eligible.to(selection.dtype)).sum(1) / eligible.sum(
        1
    ).clamp_min(1)
    return {
        "comprehensiveness": original - counterfactual,
        "sufficiency": original - evidence,
        "selection_ratio": ratio,
    }


def rationale_spans(
    mask: list[int], offsets: list[tuple[int, int]], text: str
) -> list[dict[str, object]]:
    spans = []
    start = end = None
    for selected, (left, right) in zip(mask, offsets):
        valid = selected and right > left
        if valid and start is None:
            start, end = left, right
        elif valid:
            end = right
        elif start is not None:
            spans.append({"start": start, "end": end, "text": text[start:end]})
            start = end = None
    if start is not None:
        spans.append({"start": start, "end": end, "text": text[start:end]})
    return spans


def masked_rationale_text(
    text: str, tokens: list[dict[str, object]], *, keep_selected: bool
) -> str:
    pieces: list[str] = []
    cursor = 0
    for token in tokens:
        start, end = int(token["start"]), int(token["end"])
        pieces.append(text[cursor:start])
        keep = bool(token["selected"]) if keep_selected else not bool(token["selected"])
        pieces.append(text[start:end] if keep else "[MASK]")
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)
