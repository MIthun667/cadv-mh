# Validation report

## Executed checks

- Python compilation of `src`, `scripts`, and `tests`: passed.
- Pytest: 18 passed after two defects were repaired.
- Defect 1: binary AUPRC binarization returned one column; replaced with explicit one-hot encoding.
- Defect 2: RoBERTa `inputs_embeds` generated position IDs differently from padded `input_ids`; explicit equivalent position IDs now flow through every view.
- Dataset A audit and split generation: passed.
- Dataset B audit and split generation: passed.
- Split disjointness: maximum partitions per group was one for both datasets.
- Dataset A and Dataset B training dry runs: passed.
- Classical baseline dry runs: passed.
- Cross-dataset protocol dry runs: passed.
- Tiny random-RoBERTa CUDA Stage 1, Stage 2, and Stage 3 forward/backward/optimizer smoke tests: passed with finite losses and gradients.
- Tiny checkpoint save, restore, and deterministic prediction comparison: passed.

## GPU smoke measurements

The tiny two-layer, width-64 random RoBERTa smoke test used peak allocated memory of 19,928,576 bytes in Stage 1, 20,539,904 bytes in Stage 2, and 21,277,696 bytes in Stage 3. These numbers certify execution only; they do not predict `roberta-base` memory use.

## Remaining validation

- Production `roberta-base` GPU smoke test is pending because the checkpoint is not cached and no large download was silently initiated.
- Near-duplicate candidate search is implemented but not run on the full datasets; exact-group manifests are the current certified primary splits.
- Full training, complete baselines, ablations, and transfer runs were not authorized and were not started.
- Qualitative rationale validity requires human review and must not be described as clinically validated.
