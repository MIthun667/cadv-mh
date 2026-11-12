# Mathematics

For contextual token state `h_i` and pooled state `h_o`, the selector computes:

`q_i = [h_i; h_o; h_i ⊙ h_o; |h_i-h_o|]`, `l_i = MLP(q_i)`.

Soft masks use `m_i = sigmoid((l_i + g_i^1 - g_i^2) / tau)`, where each `g` is Gumbel noise. Special and padding locations are multiplied by zero. Counterfactual embeddings are `(1-m_i)e_i + m_i e_mask`; evidence embeddings reverse those coefficients for eligible tokens.

The comparison representation is `[h_o; h_cf; h_delta; |h_delta|; h_o ⊙ h_cf]`, with `h_delta=h_o-h_cf`. The interaction is:

`h_int = tanh(W_o h_o + W_c h_cf + W_d h_delta + W_m(h_o ⊙ h_cf) + b)`.

The projected comparison and interaction are added, then projected into the residual path.

The implemented total objective is weighted classification plus original-view, necessity, sufficiency, sparsity, continuity, and teacher-student consistency terms. Necessity detaches the original probability reference by default, but retains gradients through the counterfactual path to the mask. Sparsity and continuity ignore padding and special-token boundaries. Effective-number class weights are computed from training counts only and normalized to mean one.

Stage 2 freezes the encoder and shared auxiliary head without using `no_grad`, permitting derivatives with respect to the selector-controlled input embeddings.
