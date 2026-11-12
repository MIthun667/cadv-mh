# Implementation status

| Component | Status | Evidence or limitation |
|---|---|---|
| G0 repository and environment audit | Ready | No Git repository; initial contents and hardware recorded |
| G1 independent data pipelines | Ready for exact-group experiments | Audits and immutable disjoint manifests created; full near-duplicate review pending |
| G2 CADV-MH tensor architecture | Ready | Unit and CUDA smoke paths passed for 4/5-class configurable heads |
| G3 losses, gradients, staging, checkpoints | Ready | Gradient, freezing, loss, and checkpoint tests passed |
| G4 GPU smoke | Conditionally ready | Tiny CUDA tests passed; production RoBERTa smoke awaits checkpoint availability |
| G5 experiment readiness | Partially ready | Core training, classical/original-view baselines, A0-A8 runner, metrics, statistics, and reports exist; all named neural/mechanism baselines are not yet campaign-certified |
| Full CADV-MH training | Not run | Explicitly prohibited without authorization |
| Seed policy | Ready | Training and evaluation are restricted to seed 42 |
| Cross-dataset training | Not run | Protocol dry-run only |

Scientific readiness is not the same as implementation readiness. No accuracy or comparative model claim has been produced.
