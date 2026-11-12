# Reproducibility

Install the package in editable mode and retain the generated split manifests. Every real run records the complete configuration, hardware, Python and package context, seed, split fingerprint, checkpoint state, history, runtime, and peak GPU allocation.

Use seed 42 for every training, baseline, ablation, and evaluation run. The command-line interfaces reject other training seeds, and evaluation rejects checkpoints whose recorded training seed is not 42. Keep the principal split fixed.

The development environment used Python 3.13.7, PyTorch 2.10.0+cu126, Transformers 5.15.1, scikit-learn 1.8.0, pandas 2.3.3, and an NVIDIA RTX 2060 with 12,288 MiB. The repository directory had no Git metadata, so branch and revision fields are null until version control is initialized.

`roberta-base` was not present in the local Hugging Face cache. The production smoke test therefore remains pending; no large checkpoint was downloaded silently. The tiny random RoBERTa GPU smoke test verifies code paths but does not estimate production-model memory.

Do not overwrite manifests or completed run directories. Do not tune against test metrics. Keep Dataset A and Dataset B experiment directories independent.
