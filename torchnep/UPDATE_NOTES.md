# TorchNEP update: 2026-09-15

The GPUMD package moves from 1.0.1 to 1.0.5a1. This is an alpha version.
The update spans 114 commits of package development since the code snapshot
used by the previous integration. Changes are confined to `torchnep/`.

## Included changes

- Validation data, reproducible splits, per-stage early stopping, and resume support.
- Streaming training and prediction, compact neighbor storage, and distributed execution fixes.
- Per-species cutoffs and flexible ZBL, including the compiled autograd correction.
- Optional loss curves, parity plots, and error breakdowns.
- Expanded numerical reference tests and fixtures.

## Compatibility

| Existing usage | Updated behavior |
| --- | --- |
| `early_stop_patience N` | Remains accepted as an alias of `early_stop`. Explicit `early_stop` wins regardless of input order. |
| Early stopping | Uses per-stage monitored loss. A stage-1 plateau advances to stage 2. This replaces the previous true-loss streak rule. |
| Fresh model initialization | `use_gpumd_qscaler=False` by default. Set `True` explicitly for GPUMD-style initialization. |
| Optimizer | `weight_decay=1e-4` selects AdamW by default. Set `weight_decay 0` in `nep.in` for plain Adam. |
| `lambda_1`, `lambda_2`, `pos_noise` | Ignored; use `weight_decay` for regularization. |
| `backend`, `stream_mode` arguments | Removed; backend selection is automatic and data are streamed. Prefer keyword arguments for the training API. |
| Existing `nep.txt` models | Remain readable; numerical compatibility checked on a saved W–Nb–O model. |
| Existing checkpoints | CPU continuation checked from epoch 2 to 3 with optimizer state and all loss rows retained. Use the same explicit training settings when comparing runs. |

Nepstill dependency pins and installed runtime environments are unchanged.
Adopting this package in an existing experiment requires explicitly choosing
the initialization and optimizer settings above.

## Validation

- 228 tests passed across CPU and CUDA, including compiled force paths, with
  two distributed tests initially skipped because `torchrun` was not on PATH.
- Both distributed tests passed after supplying its path: 230 unique tests passed.
- Nepstill adapter tests: 5 passed.
- Existing 88-atom W–Nb–O prediction: maximum differences of 0 eV in energy,
  2.31e-14 eV/angstrom in forces, and 6.94e-17 eV/angstrom^3 in stress.
- A newly exported model passed direct GPUMD energy/force/virial comparison
  with `rtol=1e-5`, `atol=2e-4`.
- Wheel and source distribution built; installed wheel produced a training dashboard.

Hardware coverage: one RTX 3080 and local two-process CPU/gloo execution.
Multi-node GPU and ROCm execution were not tested.
