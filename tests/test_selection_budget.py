import torch

from cadvmh.models.masking import (
    boundary_tie_fraction,
    budget_counts,
    budgeted_straight_through,
    budgeted_topk,
    construct_masked_embeddings,
    deterministic_selection,
)

RATIO = 0.2


def test_budget_counts_formula_and_float_edge():
    n = torch.tensor([0, 1, 2, 5, 10, 15, 100])
    expected = [0, 0, 1, 1, 2, 3, 20]
    assert budget_counts(n, RATIO).tolist() == expected
    assert budget_counts(torch.tensor([3]), 0.2).item() == 1  # min one token on very short rows


def test_topk_exact_count_only_eligible_and_ranked_by_logits():
    logits = torch.tensor([[9.0, 1.0, 5.0, 3.0, 7.0, 2.0, 4.0, 6.0, 8.0, 0.5, -1.0]])
    eligible = torch.ones_like(logits, dtype=torch.bool)
    eligible[0, 0] = False  # special token with the largest logit must never be chosen
    selected = budgeted_topk(logits, eligible, RATIO)
    assert selected.sum() == 2  # ceil(0.2 * 10)
    assert selected[0, 0] == 0
    assert selected[0, 8] == 1 and selected[0, 4] == 1  # logits 8.0 and 7.0 are the top-2 eligible


def test_saturated_probabilities_do_not_create_ties():
    # sigmoid(40) == sigmoid(50) == 1.0 in float32, which made the old top-k index-ordered.
    logits = torch.tensor([[0.0, 40.0, 50.0, 45.0, -3.0, 41.0, 42.0, 43.0, 44.0, 0.1, 0.2]])
    eligible = torch.ones_like(logits, dtype=torch.bool)
    assert torch.sigmoid(logits[0, 1]) == torch.sigmoid(logits[0, 2])
    selected = budgeted_topk(logits, eligible, RATIO)
    assert selected[0, 2] == 1 and selected[0, 3] == 1  # 50 and 45 are the two largest logits
    assert boundary_tie_fraction(logits, eligible, RATIO) == 0.0


def test_exact_ties_resolve_deterministically_and_are_reported():
    logits = torch.zeros(1, 8)
    eligible = torch.ones_like(logits, dtype=torch.bool)
    a, b = budgeted_topk(logits, eligible, RATIO), budgeted_topk(logits, eligible, RATIO)
    assert torch.equal(a, b) and a.sum() == 2
    assert boundary_tie_fraction(logits, eligible, RATIO) == 1.0


def test_never_selects_everything_and_short_rows():
    eligible = torch.tensor([[True] * 6 + [False] * 2, [True, True, False, False, False, False, False, False],
                             [True, False, False, False, False, False, False, False]])
    for value in (-50.0, 0.0, 50.0):
        selected = budgeted_topk(torch.full((3, 8), value), eligible, 1.0)  # even a 100% budget keeps one token
        assert (selected.sum(1) <= eligible.sum(1) - 1).all()
    assert budgeted_topk(torch.zeros(3, 8), eligible, RATIO).sum(1).tolist() == [2.0, 1.0, 0.0]


def test_straight_through_forward_is_hard_and_backward_reaches_logits():
    torch.manual_seed(0)
    logits = torch.randn(4, 12, requires_grad=True)
    eligible = torch.ones(4, 12, dtype=torch.bool)
    eligible[:, 0] = False
    selection, soft = budgeted_straight_through(logits, eligible, 0.5, RATIO)
    assert set(selection.detach().unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(selection.detach().sum(1), budget_counts(eligible.sum(1), RATIO).to(selection.dtype))
    assert (selection.detach()[:, 0] == 0).all() and (soft[:, 0] == 0).all()
    weights = torch.randn_like(selection)
    (selection * weights).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert (logits.grad[:, 0] == 0).all()  # no gradient into ineligible positions


def test_straight_through_gradient_matches_binary_concrete_surrogate():
    torch.manual_seed(1)
    logits = torch.randn(2, 9, requires_grad=True)
    eligible = torch.ones(2, 9, dtype=torch.bool)
    selection, soft = budgeted_straight_through(logits, eligible, 0.7, RATIO)
    upstream = torch.randn_like(selection)
    grad_st, = torch.autograd.grad((selection * upstream).sum(), logits, retain_graph=True)
    grad_soft, = torch.autograd.grad((soft * upstream).sum(), logits)
    assert torch.allclose(grad_st, grad_soft, atol=1e-6)


def test_eval_selection_is_deterministic_top_rho_and_threshold_respects_cap():
    logits = torch.tensor([[0.0, 3.0, 2.0, 1.0, 0.5, -1.0, -2.0, -3.0]])
    eligible = torch.ones_like(logits, dtype=torch.bool)
    assert torch.equal(deterministic_selection(logits, eligible, "top_rho", RATIO), budgeted_topk(logits, eligible, RATIO))
    many = torch.full((1, 6), 10.0)
    chosen = deterministic_selection(many, torch.ones(1, 6, dtype=torch.bool), "threshold", threshold=0.5)
    assert chosen.sum() == 5  # at least one eligible token always stays unmasked


def test_counterfactual_plus_evidence_partition_eligible_tokens():
    torch.manual_seed(2)
    embeddings, mask = torch.randn(1, 6, 4), torch.randn(4)
    eligible = torch.tensor([[False, True, True, True, True, False]])
    selection = budgeted_topk(torch.randn(1, 6), eligible, 0.5)
    cf = construct_masked_embeddings(embeddings, selection, eligible, mask, False)
    ev = construct_masked_embeddings(embeddings, selection, eligible, mask, True)
    masked_in_cf = (cf == mask).all(-1)
    masked_in_ev = (ev == mask).all(-1)
    assert not (masked_in_cf & masked_in_ev).any()  # disjoint
    assert (masked_in_cf | masked_in_ev)[eligible].all()  # together they cover all eligible tokens
    assert torch.equal(cf[:, 0], embeddings[:, 0]) and torch.equal(ev[:, 5], embeddings[:, 5])
