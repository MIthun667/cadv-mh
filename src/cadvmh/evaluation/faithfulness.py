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


# --------------------------------------------------------------------------------------------------
# Faithfulness of the deployed prediction mechanism
# --------------------------------------------------------------------------------------------------
import torch  # noqa: E402

from cadvmh.models.masking import budgeted_topk  # noqa: E402


@torch.no_grad()
def final_mechanism_faithfulness(
    model,
    batch: dict,
    mask_token_id: int,
    selection_ratio: float = 0.2,
    fusion_mode: str = "full",
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Faithfulness of the rationale to the FINAL prediction (selector + both views + interaction + final head).

    The rationale R is the model's own deterministic top-budget selection. The entire model is then re-run
    on token ids edited outside the network:
        counterfactual: R -> <mask>          (comprehensiveness = p(orig) - p(counterfactual))
        evidence-only : complement -> <mask> (sufficiency       = p(orig) - p(evidence))
    The edited inputs keep the original special-token mask, so <mask> positions are ordinary eligible tokens
    for the model's own selector, exactly as the deployed model would treat any such input.
    A random rationale with the same budget is evaluated identically as a control. Probabilities are for the
    predicted class (as for the attention/IG baselines) and for the gold class.
    """
    was_training = model.training
    model.eval()
    ids, attention, special, labels = batch["input_ids"], batch["attention_mask"], batch["special_token_mask"], batch["labels"]

    def run(input_ids):
        return model(
            input_ids, attention, special, mask_token_id, stage=3, selection_mode="top_rho",
            selection_ratio=selection_ratio, enable_evidence=fusion_mode == "evidence", return_auxiliary=True,
            fusion_mode=fusion_mode,
        )

    reference = run(ids)
    eligible = reference.valid_token_mask.bool()
    chosen = reference.selection_mask.bool() & eligible
    random_scores = torch.rand(chosen.shape, generator=generator).to(chosen.device)
    random_chosen = budgeted_topk(random_scores, eligible, selection_ratio).bool()

    def probabilities(input_ids):
        return torch.softmax(run(input_ids).final_logits.float(), -1)

    def edited(keep_mask, remove):
        target = (keep_mask if remove else eligible & ~keep_mask)
        return torch.where(target, torch.full_like(ids, mask_token_id), ids)

    original = torch.softmax(reference.final_logits.float(), -1)
    predicted = original.argmax(-1)
    views = {
        "counterfactual": probabilities(edited(chosen, True)),
        "evidence": probabilities(edited(chosen, False)),
        "random_counterfactual": probabilities(edited(random_chosen, True)),
        "random_evidence": probabilities(edited(random_chosen, False)),
    }
    pick = lambda p, cls: p.gather(1, cls[:, None]).squeeze(1)  # noqa: E731
    result = {"selection_ratio": (chosen.sum(1).float() / eligible.sum(1).clamp_min(1)), "correct": predicted.eq(labels).float()}
    for tag, cls in (("predicted", predicted), ("gold", labels)):
        base = pick(original, cls)
        result[f"{tag}_comprehensiveness"] = base - pick(views["counterfactual"], cls)
        result[f"{tag}_sufficiency"] = base - pick(views["evidence"], cls)
        result[f"{tag}_random_comprehensiveness"] = base - pick(views["random_counterfactual"], cls)
        result[f"{tag}_random_sufficiency"] = base - pick(views["random_evidence"], cls)
    model.train(was_training)
    return result
