# qNEP performance guide

This guide records measurements of the **PyTorch qNEP reference/trainer** on
our i9-12900 / RTX 3080 machine. It is not a benchmark of GPUMD's native CUDA
executable or ordinary TorchNEP. Use these results as a starting point and
measure the actual model, dataset shapes, and hardware before changing a
runtime default.

## Environment and scope

Measured 2026-09-17 at model commit
`36232ab02ec3551b1214f9bfc9b24cf7302d1324`:

- Intel Core i9-12900 under WSL, with 24 logical CPUs visible and allowed by
  process affinity; NVIDIA RTX 3080, 12 GiB.
- PyTorch 2.12.1+cu126, float32 for timing unless noted, TF32 matmul off.
- Default H/O `QNEPConfig`: radial/angular cutoffs 6/4 Angstrom,
  n_max 2/2, basis_size 4/4, l_max (2,0,0), 16 hidden neurons. This is a
  randomly initialized default model, not a production training checkpoint.
- Three-atom compact fixture, the repository's 63-atom water fixture, and
  its (2,2,1) repeat containing 252 atoms. Batch records have the same cell
  and type layout with small distinct coordinate perturbations.
- Nonzero energy/force residuals, two warmups, three measured repetitions,
  and median wall time. CUDA is synchronized before and after each sample.
  Model and initialized Adam states are reset outside each timed repetition.
- Training timings include zero_grad, loss forward/backward, finite checks,
  and Adam update. Initial input/device/label preparation, reset, validation,
  checkpointing, and full-epoch reporting are excluded. The existing forward's
  CPU neighbor reconstruction and transfers remain included.

GPU utilization before the primary run was 1%, 0%, 3%, 2%, 2%; each stage
started at 0%. Post-run desktop activity varied from 0% to 23%. This is a
low-starting-load measurement, not an OS-isolated GPU benchmark. An earlier
run with 98% starting utilization was much slower and should not be used as
an uncontended GPU baseline.

All raw repetitions, configurations, and numerical comparisons are retained
in [baseline-20260917.json](benchmarks/qnep/baseline-20260917.json).

## CPU or GPU for small structures?

One structure's loss/backward plus Adam update; float32, CPU **one PyTorch
intra-op thread**:

| Atoms | CPU (ms) | GPU (ms) |
|---:|---:|---:|
| 3 | 6.08 | 37.81 |
| 63 | 11.53 | 43.87 |
| 252 | 67.32 | 92.10 |

CPU won these three cases. This does not establish a universal crossover
size: atom count, neighbors, reciprocal vectors, species, model dimensions,
precision, and batching all affect cost. For comparable small qNEP jobs,
benchmark CPU before assuming GPU is preferable.

At 63 atoms, float64 took CPU 14.50 ms and GPU 49.53 ms in this run. No
mixed-precision optimization or accuracy trade-off was tested.

## Did the CPU use all cores?

**No all-core configuration was used in the main comparison.** The driver
called `torch.set_num_threads(1)`, controlling PyTorch intra-op parallelism.
The process normally reports 12 intra-op and 12 inter-op threads; inter-op
was left unchanged. CPU affinity was not pinned and NumPy/BLAS pools were
not independently controlled. This is not evidence that exactly one physical
core did all work, or that a larger thread setting fully occupied that many
physical cores. WSL's visible topology should not be treated as a measurement
of physical-core use.

A separate CPU-only sweep tested **1, 4, 12, and all 24 visible logical CPUs**
as intra-op thread counts. Each entry is its own median in ms:

| Intra-op threads | 63 atoms, one structure | 252 atoms, one structure | 63 atoms x 8, tensor batch |
|---:|---:|---:|---:|
| 1 | 13.96 | 67.98 | 62.62 |
| 4 | 13.64 | 61.99 | 53.46 |
| 12 | 13.68 | 78.25 | 56.72 |
| 24 | 23.11 | 787.81 | 272.22 |

Four threads were a useful starting point in this sweep. Differences between
1/4/12 at 63 atoms are small; do not interpret them as a precise optimum.
The 24-thread configuration was slower and particularly unstable. In a fresh
process with 24 tested before 1, the 252-atom medians were 3316.11 vs 67.07 ms;
the eight-structure tensor medians were 5771.11 vs 63.28 ms. These large changes
between passes make 24-thread timing unsuitable as a stable performance
estimate. They support avoiding an automatic all-visible-thread setting for
this workload, without proving a specific scheduling or thread-pool cause.

A supplementary two-thread run, with the same uncontrolled inter-op/BLAS
settings as the sweep, gave **12.98 / 66.48 / 55.19 ms**, respectively, for
the three columns above. This was a later CPU-only run while a separate GPU
job was active; it is not an isolated head-to-head comparison with the earlier
rows. Its raw samples are retained with the CPU job results below.

Start with 1, 2, and 4 threads, then compare larger settings on the real workload.
Changing thread count alone does not prove every core was effectively used.
Intra-op parallelism is separate from multi-process/distributed training.

## Five independent CPU jobs with two threads each

**Yes: five concurrent two-thread jobs improved aggregate throughput here.**
The following CPU-only experiment used five independent model/Adam instances,
each processing eight 63-atom records per update for five updates. Both
conditions perform **25 Adam updates and 200 structure evaluations** in total.
All five processes remain resident in both conditions: sequential execution
activates one at a time; concurrent execution starts all five before waiting.

| Method within each job | Five jobs sequential (s) | Five jobs concurrent (s) | Throughput increase | Structures/s, sequential → concurrent |
|---|---:|---:|---:|---:|
| Current gradient accumulation, batch 8 | 2.450 | 0.612 | 4.00x | 81.6 → 326.6 |
| Experimental tensor batch 8 | 1.488 | 0.418 | 3.56x | 134.4 → 478.7 |

Each entry is the median of three rounds with condition order alternating.
Current-path group times were 2.450/2.499/2.420 s sequential and
0.616/0.612/0.608 s concurrent. Tensor times were 1.666/1.460/1.488 s and
0.418/0.401/0.435 s. This tests concurrency at a fixed thread count; it does
not establish the best process/thread combination or compare with GPU timing.

An individual current-path job's median training-loop time rose from 0.486
to 0.603 s under concurrency, about 24% slower. The tensor job rose from
0.301 to 0.399 s, about 33% slower. Parallel execution helps finish a collection
of independent runs sooner; it does not make each run finish sooner or
combine five jobs into one trained model. It fits independent seeds,
hyperparameter trials, or separate datasets, each with its own output directory.
These workload categories are applications of the result, not additional
measured datasets. The tensor path remains a benchmark-only prototype.

### Thread settings and measurement boundaries

- Each child is a fresh interpreter with PyTorch intra-op **2**, inter-op
  **1**, and OpenMP/BLAS pools **2**. `threadpoolctl` verified one OpenMP and
  two loaded OpenBLAS pools at 2 in every worker after preparation/warmup.
  Setting only `torch.set_num_threads(2)` is insufficient to cap NumPy's
  separately observed default OpenBLAS pool of 24.
- Five times two is a useful compute-thread budget, not a guarantee of exactly
  ten OS threads or ten physical cores in use. Pools and runtime helper
  threads are separate, affinity was not pinned, and all 24 visible logical
  CPUs remained allowed. This follows PyTorch's guidance to limit threads
  within each subprocess to avoid CPU oversubscription. See
  [PyTorch multiprocessing guidance](https://docs.pytorch.org/docs/2.14/notes/multiprocessing.html#cpu-in-multiprocessing).
- Fresh pool startup, imports, fixture/label preparation, Adam initialization,
  and two warmup updates took **3.01 s** for the current path and **2.78 s**
  for the tensor path, outside the table. Both conditions use the same warmed
  pool; this is not a comparison of cold one-worker versus cold five-worker
  startup. Reuse worker processes for many tiny jobs and measure end-to-end
  time when startup matters.
- Model and initialized optimizer states reset before every measured round.
  Group wall time includes command dispatch, training, post-loop parameter
  checks, and collection of all workers' results; per-job time covers the
  training loop. All losses/gradients were finite, parameters changed, and
  each worker's final parameters matched across conditions/repetitions
  exactly (maximum observed error 0). All workers exited successfully.
- Peak resident memory was approximately **669–670 MiB per worker** for
  the current path and **701–705 MiB** for the tensor path, including the
  interpreter, imported libraries, and benchmark copies used for resets.
  Summing RSS overcounts shared libraries; it is not measured incremental
  system memory or a production trainer memory estimate.
- This was a shared-machine CPU run while a separate GPU job was active.
  One-minute load averages before/after were 1.05/2.42 for the current path
  and 2.18/3.02 for the tensor path. The concurrency benchmark hides CUDA
  from its children and performs no GPU workload. No CPU exclusivity or
  optimum scheduling claim is made.

For this machine and tested small workload, **2 threads x 5 independent jobs**
is a supported starting point. More processes, different structures, and full
training including I/O/checkpointing were not measured.

### Reproduce the CPU job comparison

The [CPU job driver](benchmarks/qnep/cpu_jobs.py) requires Linux/WSL, a prepared
Python 3.12+ environment with Torch, NumPy, ASE, and `threadpoolctl`, and the
repository fixture. It does not require a GPU or `nvidia-smi`. It sets scoped
thread environment variables before child imports, then calls
`torch.set_num_threads(2)` and `torch.set_num_interop_threads(1)` in each child.
For other launchers, apply the same environment before importing libraries:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export BLIS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
```

Set the PyTorch counts at process entry, before tensor/autograd work; inter-op
can only be set once before inter-op work starts. See the
[PyTorch inter-op API](https://docs.pytorch.org/docs/2.14/generated/torch.set_num_interop_threads.html).

```bash
QNEP_BENCH_PYTHON=/absolute/path/to/prepared/environment/bin/python
uv run --no-project "$QNEP_BENCH_PYTHON" torchnep/benchmarks/qnep/cpu_jobs.py serial
# Optional: repeat for the experimental tensor path.
uv run --no-project "$QNEP_BENCH_PYTHON" torchnep/benchmarks/qnep/cpu_jobs.py tensor
```

Defaults are `--threads 2 --jobs 5 --updates 5 --repeats 3`. The driver prints
JSONL; the portable observations and exact settings are in
[cpu-jobs-20260917.json](benchmarks/qnep/cpu-jobs-20260917.json).

## Batch size versus tensor batching

Current `backward_weighted_batch` loops over structures and calls backward
for each. `--batch-size` therefore controls gradient accumulation and
optimizer cadence; it does not combine structures into one tensor forward.

Total time for the same eight 63-atom records, float32, CPU one intra-op
thread:

| Method | Batch | Adam updates | CPU (ms) | GPU (ms) |
|---|---:|---:|---:|---:|
| Current accumulation | 1 | 8 | 98.35 | 345.19 |
| Current accumulation | 4 | 2 | 103.26 | 406.59 |
| Experimental tensor batch | 4 | 2 | 67.28 | 139.77 |
| Current accumulation | 8 | 1 | 95.44 | 340.03 |
| Experimental tensor batch | 8 | 1 | 63.09 | 97.77 |

At equal batch 8, the tensor prototype gave 1.51x CPU and 3.48x GPU throughput.
Reversing measurement order gave 1.28x and 3.61x. CPU still completed this
small tensor workload faster than GPU. Increasing only the current batch
size gave little improvement; changing batch size also changes the learning
trajectory, so these timings do not compare time to equal validation error.

The [prototype](benchmarks/qnep/batch_prototype.py) is experimental and is
**not used by `train_qnep`**. It requires equal N, cell, and type layout,
with fixed cells that do not require gradients. Neighbors are rebuilt for
each structure on every call, while descriptors/heads and reciprocal phases
are batched. One shared-cell reciprocal grid is rebuilt per batch. Grid
preparation amortization is part of the benefit; this was not a test of a
persistent cache, another descriptor backend, or general ragged batching.

At batch 8, PyTorch CUDA peak allocated memory increased from 19.06 to
39.12 MiB. This excludes CUDA context, driver reservation, and other programs.
The force graph must remain connected; more batching can cost more memory.

Numerical checks covered all six outputs, nonzero loss, and parameter
gradients through force loss. At batch 8, CPU float64 maximum energy/force/
parameter-gradient errors were 1.78e-15 / 1.67e-16 / 3.89e-16; CUDA float32
errors were 9.54e-7 / 1.04e-7 / 5.96e-8. Energy is in eV, force in eV/Angstrom,
and parameter gradients have parameter-dependent units. Tolerances were
atol=1e-11, rtol=2e-8 for float64 and atol=1e-7, rtol=2e-4 for float32.
This does not verify production convergence, arbitrary weights/missing labels,
all geometries, or exact resume after integrating a new backend.

## Re-run a bounded comparison

Run from the **repository root**, using Python 3.12+ with compatible PyTorch,
NumPy, and ASE already installed. The benchmark is checkout-only and reads
`tests_pytest/fixtures/structures/water-nat63-from-md.xyz`; it is not part of
TorchNEP's wheel. These reproduction commands currently require a working
CUDA runtime and `nvidia-smi`, including the CPU sweep's environment metadata.
Select the interpreter instead of changing an existing CUDA environment:

```bash
QNEP_BENCH_PYTHON=/absolute/path/to/prepared/environment/bin/python
```

Check GPU background activity over several seconds before a GPU comparison.
Record logical CPUs, process affinity, intra-op/inter-op threads, dtype,
versions, model dimensions, and GPU utilization. Defer timing while another
substantial GPU job is active. Do not interrupt other jobs to obtain a baseline.

Choose the relevant command; do not rerun every stage for a documentation-only
change:

```bash
# CPU thread sweep: 1, up to 2, up to 4, up to 12, and all affinity-visible CPUs.
uv run --no-project "$QNEP_BENCH_PYTHON" torchnep/benchmarks/qnep/benchmark.py threads

# E/F/charge and force-loss gradient agreement at batch 8.
uv run --no-project "$QNEP_BENCH_PYTHON" torchnep/benchmarks/qnep/benchmark.py parity8

# Eight-record throughput at batch 1/4/8, then reversed-order batch 8.
uv run --no-project "$QNEP_BENCH_PYTHON" torchnep/benchmarks/qnep/benchmark.py batch
uv run --no-project "$QNEP_BENCH_PYTHON" torchnep/benchmarks/qnep/benchmark.py confirm

# Original bounded size/dtype comparison.
uv run --no-project "$QNEP_BENCH_PYTHON" torchnep/benchmarks/qnep/benchmark.py scale
```

Each command prints JSONL with raw repetitions. Save stdout and stderr in a
new result directory, preserving the dated baseline. If a large difference
could be due to measurement order or scheduling, repeat just the relevant
pair in reverse order. Report variability and equal-work comparisons, not
only the best observed number. The promoted scripts retain the same numerical
operations; the driver adds the CPU-thread sweep and the prototype uses typed
parameter/buffer accessors. Use fresh timings when changing the implementation.

For small investigations, this short protocol and one numerical parity check
are sufficient first steps. Full training, MD, broad atom-count sweeps, and
full regression suites should answer a concrete remaining question rather
than run automatically.
