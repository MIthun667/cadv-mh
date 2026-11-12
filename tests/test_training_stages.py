import pytest
import torch
import yaml

from cadvmh.losses.objective import CADVMHLoss, LossWeights
from cadvmh.training.ablations import ABLATED_TERMS, ABLATIONS, evidence_required, spec_to_config
from cadvmh.training.optimizer import build_optimizer
from cadvmh.training.schedules import accumulation_window, encoder_learning_rate, temperature_for
from cadvmh.training.stages import configure_stage, set_stage_train_mode
from cadvmh.training.trainer import Trainer, primary_logits
from cadvmh.utils.config import KNOWN_KEYS, load_config
from conftest import ROOT_DIR


# ---- schedules -------------------------------------------------------------------------------
def test_temperature_anneals_linearly_across_stages_and_reaches_floor():
    values = [temperature_for(2, e, 2, 5, 1.0, 0.3) for e in (1, 2)] + [temperature_for(3, e, 2, 5, 1.0, 0.3) for e in range(1, 6)]
    assert values[0] == 1.0 and values[-1] == pytest.approx(0.3)
    assert all(a > b for a, b in zip(values, values[1:]))
    steps = [a - b for a, b in zip(values, values[1:])]
    assert steps == pytest.approx([steps[0]] * len(steps))  # linear
    assert temperature_for(1, 1, 2, 5, 1.0, 0.3) == 1.0
    assert temperature_for(3, 5, 2, 5, 1.0, 0.3, fixed=True) == 1.0  # A7


def test_accumulation_windows_including_short_tail():
    # 7 micro-batches, accumulation 4 -> windows of 4 and 3
    sizes = [accumulation_window(s, 7, 4) for s in range(1, 8)]
    assert [w for w, _ in sizes] == [4, 4, 4, 4, 3, 3, 3]
    assert [e for _, e in sizes] == [False, False, False, True, False, False, True]
    assert accumulation_window(1, 1, 4) == (1, True)
    with pytest.raises(ValueError):
        accumulation_window(8, 7, 4)


def test_encoder_learning_rate_is_stage_specific():
    assert encoder_learning_rate(1, 2e-5, 1e-5) == 2e-5
    assert encoder_learning_rate(3, 2e-5, 1e-5) == 1e-5


def test_optimizer_groups_follow_stage_freezing_and_learning_rates(tiny_model):
    configure_stage(tiny_model, 2)
    optimizer = build_optimizer(tiny_model, 2e-5, 5e-5, 1e-4, 0.01)
    assert [g["name"] for g in optimizer.param_groups] == ["selector"]
    configure_stage(tiny_model, 3)
    optimizer = build_optimizer(tiny_model, 1e-5, 5e-5, 1e-4, 0.01)
    lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
    assert lrs == {"encoder": 1e-5, "selector": 5e-5, "pooling": 1e-4, "auxiliary_head": 1e-4, "interaction": 1e-4, "final_head": 1e-4}
    trainable = {id(p) for p in tiny_model.parameters() if p.requires_grad}
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == trainable


def test_stage2_freezes_and_evals_frozen_modules(tiny_model):
    configure_stage(tiny_model, 2)
    set_stage_train_mode(tiny_model, 2)
    assert tiny_model.selector.training
    assert not tiny_model.encoder.training and not tiny_model.pooling.training and not tiny_model.auxiliary_head.training
    set_stage_train_mode(tiny_model, 3)
    assert all(m.training for m in (tiny_model.encoder, tiny_model.selector, tiny_model.final_head))


# ---- trainer ---------------------------------------------------------------------------------
def _loader(batch, copies):
    items = [{k: v for k, v in batch.items()} for _ in range(copies)]
    return items


def _trainer(model, **kwargs):
    objective = CADVMHLoss(LossWeights())
    configure_stage(model, kwargs.pop("stage"))
    optimizer = build_optimizer(model, 1e-3, 1e-3, 1e-3, 0.0)
    return Trainer(model, objective, optimizer, "cpu", fp16=False, **kwargs), optimizer


def test_trainer_steps_once_per_window_including_short_tail(tiny_model, tiny_batch):
    trainer, optimizer = _trainer(tiny_model, stage=3, gradient_accumulation=2)
    calls = []
    original_step = optimizer.step
    optimizer.step = lambda *a, **k: (calls.append(1), original_step(*a, **k))[1]
    trainer.train_epoch(_loader(tiny_batch, 3), 3, 1.0, 4)  # windows of 2 and 1
    assert len(calls) == 2


def test_stage2_metrics_use_aux_head_and_report_selector_score(tiny_model, tiny_batch):
    trainer, _ = _trainer(tiny_model, stage=2, gradient_accumulation=1)
    output = tiny_model(**{k: tiny_batch[k] for k in ("input_ids", "attention_mask", "special_token_mask")}, mask_token_id=4, stage=2)
    assert primary_logits(output, 2) is output.original_logits
    assert primary_logits(output, 3) is output.final_logits
    result = trainer.evaluate(_loader(tiny_batch, 2), 2, 4)
    for key in ("validation_selector_score", "validation_comprehensiveness_aux", "validation_sufficiency_aux",
                "validation_random_comprehensiveness_aux", "validation_selection_ratio",
                "validation_selector_saturation", "validation_boundary_tie_fraction"):
        assert key in result
    # score = E[p_ev - p_cf] = comprehensiveness - sufficiency
    assert result["validation_selector_score"] == pytest.approx(
        result["validation_comprehensiveness_aux"] - result["validation_sufficiency_aux"], abs=1e-6)
    assert result["validation_selection_ratio"] <= 0.5


def test_stage2_best_epoch_is_not_a_function_of_the_random_final_head(tiny_model, tiny_batch):
    """Re-initialising the (frozen, untrained) final head must not change the stage-2 selection metric."""
    trainer, _ = _trainer(tiny_model, stage=2, gradient_accumulation=1)
    before = trainer.evaluate(_loader(tiny_batch, 1), 2, 4)["validation_selector_score"]
    for module in (tiny_model.final_head, tiny_model.interaction):
        for p in module.parameters():
            torch.nn.init.normal_(p, std=1.0) if p.dim() > 1 else torch.nn.init.normal_(p)
    assert trainer.evaluate(_loader(tiny_batch, 1), 2, 4)["validation_selector_score"] == pytest.approx(before, abs=1e-6)


def test_evidence_branch_only_when_trained_or_used(tiny_model):
    t, _ = _trainer(tiny_model, stage=2, gradient_accumulation=1)
    assert t._needs_evidence(2) and not t._needs_evidence(1)
    t.objective.weights.sufficiency = 0.0
    assert not t._needs_evidence(2)
    assert evidence_required(0.0, "evidence") and not evidence_required(0.0, "full") and evidence_required(0.5, "mean")


# ---- losses ----------------------------------------------------------------------------------
def test_stage_loss_composition(tiny_model, tiny_batch):
    weights = LossWeights()
    out = tiny_model(**{k: tiny_batch[k] for k in ("input_ids", "attention_mask", "special_token_mask")}, mask_token_id=4, stage=3, selection_mode="top_rho")
    loss = CADVMHLoss(weights)
    labels = tiny_batch["labels"]
    s3 = loss(out, labels, stage=3)
    manual = sum(getattr(weights, k) * s3[k] for k in ("classification", "original", "necessity", "sufficiency", "sparsity", "continuity"))
    assert float(s3["total"]) == pytest.approx(float(manual), rel=1e-5)
    s2 = loss(out, labels, stage=2)
    manual2 = s2["original"] + sum(getattr(weights, k) * s2[k] for k in ("necessity", "sufficiency", "sparsity", "continuity"))
    assert float(s2["total"]) == pytest.approx(float(manual2), rel=1e-5)  # no final-head classification term in stage 2
    assert float(loss(out, labels, stage=1)["total"]) == pytest.approx(float(s3["original"]), rel=1e-5)


def test_necessity_and_sufficiency_give_selector_gradient_only_through_the_masked_views(tiny_model, tiny_batch):
    configure_stage(tiny_model, 2)
    set_stage_train_mode(tiny_model, 2)
    weights = LossWeights(necessity=1.0, sufficiency=0.0, sparsity=0.0, continuity=0.0)
    out = tiny_model(**{k: tiny_batch[k] for k in ("input_ids", "attention_mask", "special_token_mask")}, mask_token_id=4, stage=2, enable_evidence=False)
    CADVMHLoss(weights, necessity_margin=2.0)(out, tiny_batch["labels"], stage=2)["necessity"].backward()  # margin 2 => always active
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tiny_model.selector.parameters())


# ---- ablations & configs ---------------------------------------------------------------------
def test_ablation_specs_match_generated_configs():
    names = {"A0": "a0_original", "A1": "a1_selector", "A2": "a2_counterfactual", "A3": "a3_necessity", "A4": "a4_sufficiency",
             "A5": "a5_interaction", "A6": "a6_continuity", "A7": "a7_consistency", "A8": "a8_complete"}
    for name, spec in ABLATIONS.items():
        on_disk = yaml.safe_load((ROOT_DIR / "configs" / "ablations" / f"{names[name]}.yaml").read_text())["ablation"]
        assert on_disk == spec_to_config(spec), name


def test_ablation_components_are_progressive_and_really_toggled():
    a = ABLATIONS
    assert a["A0"].stages == (1,) and a["A0"].active_losses == () and a["A0"].matched_encoder_epochs
    assert a["A1"].fusion_mode == "evidence" and "necessity" not in a["A1"].active_losses
    assert a["A2"].fusion_mode == "mean" and set(a["A2"].loss_overrides) == {"necessity", "sufficiency", "continuity", "consistency"}
    assert set(a["A3"].active_losses) - set(a["A2"].active_losses) == {"necessity"}
    assert set(a["A4"].active_losses) - set(a["A3"].active_losses) == {"sufficiency"}
    assert a["A5"].fusion_mode == "full" and a["A5"].active_losses == a["A4"].active_losses
    assert set(a["A6"].active_losses) - set(a["A5"].active_losses) == {"continuity"}
    assert set(a["A7"].active_losses) == set(ABLATED_TERMS) and a["A7"].fixed_temperature
    assert set(a["A8"].active_losses) == set(ABLATED_TERMS) and not a["A8"].fixed_temperature and a["A8"].alias_of_canonical_run


def test_configs_load_and_reflect_the_implementation():
    for name in ("dataset_a", "dataset_b"):
        config = load_config(ROOT_DIR / "configs" / f"{name}.yaml")
        assert config["model"]["selection_mode"] == "budgeted_st"
        assert config["project"]["output_root"] == "experiments_v2"
        assert config["loss"]["sparsity"] == 0.0
        assert config["dataset"]["manifest"].endswith("primary_split_v2.csv")
    assert load_config(ROOT_DIR / "configs" / "dataset_b.yaml")["dataset"]["encoding"] == "utf-8"


def test_unknown_config_keys_are_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  comparison_mode: full\n")
    with pytest.raises(ValueError, match="comparison_mode"):
        load_config(bad)
    worse = tmp_path / "worse.yaml"
    worse.write_text("model:\n  selection_mode: soft_hard\n")
    with pytest.raises(ValueError, match="selection_mode"):
        load_config(worse)
    assert "encoding" not in KNOWN_KEYS["data"]  # no global encoding switch
