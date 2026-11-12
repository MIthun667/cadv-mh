# Architecture

CADV-MH uses exactly one encoder object for the original, counterfactual, and evidence-only calls. For a hidden width `d`, attention pooling produces the original representation `h_o`. The selector receives `[h_i; h_o; h_i*h_o; |h_i-h_o|]`, so its input width is `4d`, and returns one logit per token.

Eligible tokens exclude padding and tokenizer special tokens. Training supports soft Binary Concrete and straight-through masks. Evaluation supports deterministic top-budget and threshold masks. Masking interpolates word embeddings with the encoder's mask-token word embedding; the encoder adds its own positional embeddings. Explicit position IDs preserve equality between unmodified `input_ids` and `inputs_embeds` paths for RoBERTa.

The original, counterfactual, and evidence representations use the same pooling module and shared auxiliary classifier. The main comparison is `[h_o; h_cf; h_o-h_cf; |h_o-h_cf|; h_o*h_cf]` with width `5d`. Projected comparison and additive context-effect interaction are combined, projected to `d`, added residually to `h_o`, normalized, and classified.

Stage 1 executes only the original view. Stage 2 executes all three views so the frozen encoder and auxiliary head can transmit gradients to the selector inputs. Stage 3 executes original and counterfactual views, plus evidence-only when sufficiency is enabled. Ordinary final inference does not require the evidence-only view.

No uncertainty or abstention head exists.
