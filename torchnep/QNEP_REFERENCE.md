# qNEP mode 2 reference and trainer

TorchNEP **1.0.5a1+qnep2** provides an independent, PyTorch qNEP mode-2
reference implementation and a dedicated trainer. The ordinary TorchNEP
`train_nep` APIs remain unchanged.

qNEP accepts only neutral, fully three-dimensional periodic structures. It
predicts total energy, forces, and latent projected charges. Install PyTorch,
NumPy, and ASE in the environment selected for the run; TorchNEP intentionally
has no required runtime dependencies and supports Python 3.9 or newer.

## Routes

The existing reference route is retained for compact experiments and produces
an inference checkpoint directly:

```bash
python -m torchnep.qnep train.xyz --elements H O --checkpoint qnep.pt --steps 100
```

The dedicated route requires an explicit held-out validation file and an output
directory. It adds mini-batching, epoch checkpoints, validation metrics, best
model selection, and resume:

```bash
python -m torchnep.qnep train.xyz --validation valid.xyz --output-dir qnep-run \
  --elements H O --epochs 100 --batch-size 2 --learning-rate 0.001 \
  --precision float64 --snapshot-id curated-snapshot-20260915
```

Use `--resume` with a training checkpoint to continue a dedicated run. The
total epoch target can increase, and the output directory and checkpoint
interval can change:

```bash
python -m torchnep.qnep train.xyz --validation valid.xyz \
  --output-dir continued-run --resume qnep-run/last.training.pt --epochs 200
```

The reference and dedicated options are separate routes. Mixing their
exclusive options is an argument error. qNEP input errors exit nonzero and do
not report a successful run.

## Python API

The dedicated API keeps record construction explicit. `load_records` reads
curated extxyz into CPU-resident records, and `snapshot_id` is opaque
provenance supplied by the dataset workflow.

```python
from pathlib import Path

from torchnep.qnep import (
    QNEPConfig,
    QNEPDatasets,
    QNEPModel,
    QNEPRunConfig,
    train_qnep,
)
from torchnep.qnep.data_io import load_records

model = QNEPModel(QNEPConfig(type_names=("H", "O"))).double()
datasets = QNEPDatasets(
    train=load_records("train.xyz", model),
    validation=load_records("valid.xyz", model),
    snapshot_id="curated-snapshot-20260915",
)
result = train_qnep(
    model,
    datasets,
    QNEPRunConfig(output_dir=Path("qnep-run"), epochs=100, batch_size=2),
)
print(result.best_inference_path)
```

`QNEPRecord` contains a `TrainingSample` plus optional `record_id` and
`group_id`. API callers may construct records themselves. The trainer rejects
identical geometry in both splits and rejects shared supplied record or group
IDs across splits. It does not derive trajectory lineage or dataset splits from
bare extxyz.

## Labels, masks, and weights

The extxyz loader accepts ASE `energy` and `forces` labels. Per-frame metadata
uses `sample_weight` (positive and finite), optional `record_id`, and optional
`group_id`. A per-atom `force_mask` has shape `(N, 3)` and contains booleans or
explicit `0`/`1`. Only present force components are supervised; a masked force
may be represented by `NaN`. Zero is a valid energy, force, mask value, or
weight-independent label and is never interpreted as absent.

Every record needs at least one active energy or force target under the chosen
loss weights. Charge-only records are rejected. Energy and force labels must
be supplied only for neutral, full-3D-PBC structures with finite coordinates,
a finite nonzero-volume cell, and known elements. Extra stress and virial
labels are ignored for energy/force training; stress and NPT use remain
unsupported.

For record `i`, qNEP uses

```text
L_i = wE * ((E_i - Eref_i) / N_i)^2
    + wF * mean_present((F_i - Fref_i)^2)
    + wQ * (sum_j qraw_ij)^2
objective = sum(a_i * L_i) / sum(a_i)
```

Here `a_i` is `sample_weight`; missing energy or force terms contribute zero.
Training draws a seeded uniform permutation without replacement, so each train
record appears once per epoch. For mini-batch `B`, it applies
`n / (len(B) * sum(a)) * sum_B(a_i * L_i)`, including the final short batch.
It does not resample in proportion to weights or normalize each batch by only
its own weights.

Validation has a fixed order and does not update the optimizer or consume the
shuffle stream. Missing metric components are recorded as `null`, never as a
zero measurement. The best and early-stop metric is the weighted validation
energy-plus-force objective; it excludes the raw-charge penalty.

## Artifacts and checkpoint compatibility

Each dedicated run records `run.json` before training and appends epoch rows to
`metrics.jsonl`. Checkpoint boundaries write:

| Path | Meaning |
| --- | --- |
| `last.training.pt` | Authoritative resumable training state at the saved epoch. |
| `latest.qnep.pt` | Inference model at the latest completed epoch. |
| `best.qnep.pt` | Inference model from the strict best validation metric. |

Ties retain the earlier best epoch. The caller's model remains at the latest
weights; load `best.qnep.pt` explicitly when the best model is needed.

Inference uses the backward-compatible
`torchnep-qnep-mode2-reference-v1` payload. It includes model configuration
and weights and remains portable between supported inference devices. Training
state uses the separate `torchnep-qnep-mode2-training-v1` payload, with model
and optimizer state, best state, epoch/step/patience state, metric history,
dataset and settings fingerprints, snapshot ID, shuffle and runtime RNG state,
and runtime provenance. A reference-v1 inference checkpoint cannot resume
training.

Resume requires the same model, labels, masks, effective weights, split and
record order, snapshot ID, loss/sampling settings, precision, resolved device,
and PyTorch version. It restores saved weights and RNG rather than reseeding.
Source records remain immutable CPU float32 or float64 tensors; qNEP converts
them transiently to the configured run precision for model evaluation.
The exact continuation guarantee is limited to CPU float64 on the same runtime
and deterministic settings. Cross-device resume is rejected; inference remains
device-portable. If a checkpoint is interrupted across artifact writes, its
saved history is authoritative and rebuilds the continued metrics sequence.
If saved early-stop patience is already exhausted, resume materializes the
best/latest inference artifacts, metrics, and training checkpoint without an
additional optimizer epoch, even when the requested target equals or exceeds
the completed epoch.

## Numerical scope

- The model uses the qNEP mode-2 reciprocal electrostatic reference with
  neutral projected charges. Neighbour topology is rebuilt for evaluation, and
  force loss differentiates through the full energy.
- `load_gpumd_reference` accepts the supported non-ZBL `nep4_charge2` reference
  layout for numerical energy/force comparisons. qNEP checkpoint export to
  GPUMD text is unsupported.
- ASE exposes energy, forces, and latent charges. Stress and cell derivatives
  are not supported.

Unsupported: charged cells, partial or nonperiodic boundaries, mode 1, ZBL
import, BEC labels, stress/NPT, Extended LES multipoles, GPUMD export, custom
CUDA kernels, distributed training, and automatic teacher/alignment/splitting
workflows. A vacuum cell remains periodic, so use of isolated-molecule or slab
labels requires a separate convergence study.

The numerical tests check software behavior such as finite-difference force
agreement, symmetries, checkpoint round trips, masks, weights, and selected
GPUMD comparisons. They do not establish material accuracy, vacuum convergence,
or molecular-dynamics stability.

## Validation required for releases

CPU CI runs the TorchNEP pytest suite. CUDA-only tests and external GPUMD
comparison tests explicitly skip when their required runtime is absent; those
skips are visible but do not provide GPU/GPUMD coverage. Run the comparison
locally with the exact worktree binary and retain the generated artifacts:

```bash
GPUMD_QNEP_BINARY=/absolute/path/to/gpumd \
  python -m pytest -ra tests/test_qnep_gpumd.py -s
```

The comparison covers the water fixture and a derived non-ZBL Ba/Ti/O model.
The original ZBL BaTiO3 fixture is deliberately rejected. Consult
[`tests/README.md`](tests/README.md) for the focused CPU command and test
coverage.
