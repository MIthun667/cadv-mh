# Architecture

CADV-MH uses exactly one encoder object for the original, counterfactual, and evidence-only calls. For a hidden width `d`, attention pooling produces the original representation `h_o`. The selector receives `[h_i; h_o; h_i*h_o; |h_i-h_o|]`, so its input width is `4d`, and returns one logit per token.

Eligible tokens exclude padding and tokenizer special tokens. Training (`selection_mode: budgeted_st`) selects exactly `k = min(max(1, ceil(rho n)), n-1)` tokens by hard top-k of Gumbel-perturbed selector logits; the backward pass uses the binary-concrete relaxation `sigmoid((l+g)/tau)` (straight-through). Validation and final evaluation use the same top-k on unperturbed logits, so the budget and mask regime are identical in training and evaluation. Legacy `soft`/`straight_through` modes are retained only to reproduce pre-repair runs. Masking interpolates word embeddings with the encoder's mask-token word embedding; the encoder adds its own positional embeddings. Explicit position IDs preserve equality between unmodified `input_ids` and `inputs_embeds` paths for RoBERTa.

The original, counterfactual, and evidence representations use the same pooling module and shared auxiliary classifier. The main comparison is `[h_o; h_cf; h_o-h_cf; |h_o-h_cf|; h_o*h_cf]` with width `5d`. Projected comparison and additive context-effect interaction are combined, projected to `d`, added residually to `h_o`, normalized, and classified.

Stage 1 executes only the original view. Stage 2 executes all three views with the encoder, pooling and auxiliary head frozen and in eval mode, so only the selector learns; its checkpoint criterion is the validation selector score `mean[p_ev(y)-p_cf(y)]` of the auxiliary head. Stage 3 executes original and counterfactual views, plus evidence-only only when sufficiency is weighted or the fusion mode needs it; its criterion is final-head validation macro-F1. Ordinary final inference does not require the evidence-only view.

No uncertainty or abstention head exists.
