# Experiment protocol

## Primary experiments

Dataset A keeps five labels: anxiety, depression, suicidal, neutral, and casual. Dataset B uses the binary `jsfactory/mental_health_reddit_posts` task with OCD and ADHD labels. Models, weights, validation decisions, optimizers, checkpoints, and result directories are never shared. Every experiment uses seed 42.

Exact canonical duplicate groups are constructed before splitting. Conflicting-label groups are reported and excluded from the primary experiment. One deterministic representative per remaining group is retained. A fixed 70/10/20 group-stratified split is then created; actual counts are recorded in immutable manifests. Near-duplicate search is optional and must be reviewed because high lexical similarity does not guarantee semantic equivalence.

Macro-F1 is the selection metric. Secondary metrics include accuracy, balanced accuracy, weighted F1, MCC, per-class measures, one-vs-rest AUPRC, NLL, Brier score, ECE, and confusion matrices. Headline faithfulness re-runs the complete final-prediction mechanism (selector, both views, interaction, final head) on token ids edited outside the network, using the model-predicted class and the same budget rule as the attention/IG baselines, plus a same-budget random-rationale control. Auxiliary-head within-pipeline probabilities are reported separately as mechanism diagnostics.

## Ablations

- A0: original-view encoder classifier, trained for stage1+stage3 epochs so encoder-training budget matches the full model.
- A1: learned selector with evidence-only prediction.
- A2: counterfactual branch with mean fusion.
- A3: A2 plus necessity.
- A4: A3 plus sufficiency.
- A5: full context-effect interaction.
- A6: A5 plus continuity.
- A7: A6 plus conservative consistency with fixed soft-mask temperature.
- A8: complete staged training with temperature annealing (canonical run; trained once into `<root>/<dataset>/cadvmh`).

Leave-one-loss-out commands cover necessity, sufficiency, continuity, and consistency (sparsity is structural). Fusion-mode flags cover full, mean, difference, and evidence paths. Identical manifests and seed 42 are mandatory.

## Cross-dataset experiment

Transfer is separate. Dataset A is mapped to the four-class transfer space only inside transfer runs. A new four-class source model is trained, selected only on source validation data, and evaluated on the target test manifest after overlap exclusion. Target test labels are not used for training or model selection.
