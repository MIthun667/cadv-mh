"""The repaired selection mechanism cannot satisfy necessity/sufficiency by selecting every token."""
import torch

from cadvmh.losses.objective import CADVMHLoss, LossWeights
from cadvmh.models.masking import budget_counts
from cadvmh.training.stages import configure_stage, set_stage_train_mode

KEYS = ("input_ids", "attention_mask", "special_token_mask")


def _forward(model, batch, mode, ratio=0.2, stage=2):
    return model(**{k: batch[k] for k in KEYS}, mask_token_id=4, stage=stage, temperature=1.0, selection_mode=mode, selection_ratio=ratio)


def _push_selector_to_select_all(model, bias):
    with torch.no_grad():
        model.selector.network[-1].bias.fill_(bias)
        model.selector.network[-1].weight.zero_()


def test_legacy_soft_mode_has_the_select_all_optimum_but_budgeted_st_does_not(tiny_model, tiny_batch):
    configure_stage(tiny_model, 2)
    set_stage_train_mode(tiny_model, 2)
    _push_selector_to_select_all(tiny_model, 30.0)
    eligible = tiny_batch["attention_mask"].bool() & ~tiny_batch["special_token_mask"].bool()
    soft = _forward(tiny_model, tiny_batch, "soft")
    assert (soft.selection_mask[eligible] > 0.99).all()  # collapse is representable under the old mode
    st = _forward(tiny_model, tiny_batch, "budgeted_st")
    count = budget_counts(eligible.sum(1), 0.2)
    assert torch.equal(st.selection_mask.detach().sum(1).long(), count)
    assert (st.selection_mask.detach().sum(1) < eligible.sum(1)).all()


def test_optimizing_necessity_sufficiency_cannot_raise_the_selected_count(tiny_model, tiny_batch):
    configure_stage(tiny_model, 2)
    set_stage_train_mode(tiny_model, 2)
    optimizer = torch.optim.Adam(tiny_model.selector.parameters(), lr=0.5)
    objective = CADVMHLoss()
    eligible = tiny_batch["attention_mask"].bool() & ~tiny_batch["special_token_mask"].bool()
    allowed = budget_counts(eligible.sum(1), 0.2)
    for _ in range(25):
        output = _forward(tiny_model, tiny_batch, "budgeted_st")
        losses = objective(output, tiny_batch["labels"], stage=2)
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        assert torch.equal(output.selection_mask.detach().sum(1).long(), allowed)  # never more than the budget
        # the counterfactual always differs from the original input, the evidence view always hides tokens
        assert (output.selection_mask.detach().sum(1) >= 1).all()
        assert (output.selection_mask.detach().sum(1) <= eligible.sum(1) - 1).all()


def test_sparsity_penalty_is_inert_under_exact_budget_and_active_for_legacy_modes(tiny_model, tiny_batch):
    configure_stage(tiny_model, 2)
    set_stage_train_mode(tiny_model, 2)
    objective = CADVMHLoss(LossWeights(sparsity=1.0))
    st = _forward(tiny_model, tiny_batch, "budgeted_st")
    assert st.budget_enforced
    losses = objective(st, tiny_batch["labels"], stage=2)
    assert float(losses["sparsity"]) == 0.0  # no mean-matching gradient that would saturate the ST surrogate
    _push_selector_to_select_all(tiny_model, 30.0)
    soft = _forward(tiny_model, tiny_batch, "soft")
    assert not soft.budget_enforced and float(objective(soft, tiny_batch["labels"], stage=2)["sparsity"]) > 0.5
    tiny_model.eval()
    assert _forward(tiny_model, tiny_batch, "top_rho").budget_enforced  # validation selection is exact-budget too


def test_config_forbids_sparsity_weight_with_budgeted_selection(tmp_path):
    import pytest

    from cadvmh.utils.config import load_config
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  selection_mode: budgeted_st\nloss:\n  sparsity: 0.05\n")
    with pytest.raises(ValueError, match="sparsity"):
        load_config(path)
