# TorchNep tests

Pure pytest suite (`numpy` + `torch`; `ase` only for the ASE test).

```bash
pytest tests/                      # full suite
pytest tests/ -k float32           # one dtype
TEST_DEVICE=cpu pytest tests/      # restrict device (default: cpu + cuda if present)
```

### qNEP mode 2

The focused CPU regression set covers numerical derivatives and symmetries,
curated-record parsing and fingerprints, masked/weighted batches, reference
and training checkpoint boundaries, dedicated training/CLI/resume behavior,
and ASE runtime checks:

```bash
TEST_DEVICE=cpu pytest -ra \
  tests/test_qnep_reference.py tests/test_qnep_numerics.py \
  tests/test_qnep_records.py tests/test_qnep_batching.py \
  tests/test_qnep_training_checkpoint.py tests/test_qnep_trainer.py \
  tests/test_qnep_training_cli.py tests/test_qnep_resume.py
```

The GitHub workflow runs the CPU suite. CUDA-only checks are explicitly
skipped on CPU runners, and a missing `GPUMD_QNEP_BINARY` explicitly skips the
external comparison. Those skips are not GPUMD validation. Before a release,
run the comparison against the exact-worktree binary and retain its output:

```bash
GPUMD_QNEP_BINARY=/absolute/path/to/gpumd \
  python -m pytest -ra tests/test_qnep_gpumd.py -s
```

This exercises the water model and a derived non-ZBL Ba/Ti/O comparison model;
the repository's original ZBL fixture must remain rejected. Numerical and
software tests do not establish material accuracy, vacuum convergence, or MD
stability.

| file | covers |
| --- | --- |
| `test_gpumd_parity.py` | E / F / V / descriptor vs the GPUMD reference (incl. compressed CrCoNi frames where ZBL forces reach ~120 eV/Å); analytical vs autograd; train path vs predict path. |
| `test_descriptors.py` | Angular basis L=1..8; gradient checks; the six higher-body channels (q_222, q_1111, q_112, q_123, q_233, q_134) — GPUMD-polynomial match and rotational invariance; backend auto-resolution and loop/bmm/mulsum numerical equivalence. |
| `test_neighbor.py` | Cell-list vs brute-force neighbor search; tiled / auto-block paths. |
| `test_parsing.py` | Legacy and current `l_max` nep.in / nep.txt parsing. |
| `test_ase_calculator.py` | Optional ASE calculator (energy/forces/stress, ZBL split). |
| `test_b1_and_gpumd_qscaler.py` | Analytical `b1` offset (residual → 0, `nep_best` ≤ `nep_final`); `use_gpumd_qscaler` reproduces GPUMD's `c=1` q_scaler; `gpumd_init_parameters` re-inits coeffs **and** NN weights uniform(−1,1); weight_decay (AdamW) shrinks the weights. |
| `test_run_seed_and_valid.py` | `run_seed` reproducibility; `valid_file` / `valid_ratio` deterministic split, best-model selection on validation loss, `*_test.out`, split preserved across resume; `early_stop` fires on a plateau (validation-loss branch), is off by default, is per-stage (a stage-1 plateau jumps into Stage 2, surviving resume). `export_valid_split` reproduces the internal split (verbatim frames, matches `energy_test.out`). |
| `test_stream_mode.py` | `StreamDataStore` (the training data store): `collate` is bit-exact vs an independently assembled reference (concatenation + offsets + basis straight from the ops functions); metadata/mask consistency incl. missing channels; prefetched `iter_collated` matches direct collate; 2-rank DDP same-seed reproducibility. The DDP case is local-only: set `TORCHNEP_TEST_DDP=1` (skipped in CI — multi-process rendezvous is unreliable on shared runners). |
| `test_compiled_autograd.py` | `CompiledAutogradForce` (make_fx-materialized autograd forces): outputs and parameter gradients (second-order path through the force loss) match eager autograd across batch shapes on one dynamic graph; energy-only calls fall back to eager. CUDA-only — auto-skipped on CPU hosts/CI. |

**Tolerance vs GPUMD:** `rtol=1e-5, atol=2e-4`.

**Re-baking the reference** (only if `nep_CrCoNi.txt` / `CrCoNi.xyz` change):

```bash
GPUMD_NEP=/path/to/GPUMD/src/nep python tests/bake_fixtures.py
```
