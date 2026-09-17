# TorchNEP agent guidance

For qNEP work, read [QNEP_REFERENCE.md](QNEP_REFERENCE.md) for the supported
model/trainer contract and [QNEP_PERFORMANCE.md](QNEP_PERFORMANCE.md) before
choosing CPU/GPU defaults or making performance claims.

## Measured qNEP starting points

On the i9-12900 / RTX 3080 / WSL environment, the 2026-09-17 default H/O
qNEP float32 measurements favored CPU for the tested 3-, 63-, and 252-atom
single-structure training steps. Start by comparing CPU at 1, 2, and 4 PyTorch
intra-op threads for similar small workloads. Re-measure the target workload;
these are measurements of three cases, not a universal atom-count cutoff or
results for GPUMD native CUDA, ordinary NEP, or a trained production model.

The main CPU comparison used **1 PyTorch intra-op thread**, not all cores.
A separate 1/4/12/24-thread sweep found 24 slower and highly variable. There
are 24 logical CPUs visible to this WSL process. Thread count does not prove
physical-core occupancy; inter-op threads and NumPy/BLAS pools are separate.

For independent small training jobs, **five concurrent processes with two
threads each** are a measured starting point: at 63 atoms x 8 records, total
throughput was 4.00x sequential execution with the current accumulation path
and 3.56x with the tensor prototype. These are warm-process measurements of
five independent models, not a speedup for one distributed model. Set each
child's PyTorch intra-op and OpenMP/BLAS pools to 2 and inter-op to 1; the
uncapped NumPy OpenBLAS default here was 24. See the guide for startup,
memory, shared-machine conditions, and the CPU-only reproduction command.

## Performance work

- Current qNEP `--batch-size` accumulates gradients through sequential
  structure evaluations. Increasing it alone did not materially improve the
  measured eight-record pass. It also changes optimizer-step count.
- The experimental same-cell tensor batch improved equal-batch-size
  throughput: GPU about 3.5–3.6x and CPU about 1.3–1.5x at batch 8.
  It is **not integrated into the trainer**. It requires the same atom count,
  cell, and type layout; batch-shared reciprocal-grid preparation is part of
  the measured benefit. See the guide for memory cost and exact timings.
- Before GPU timing, observe background utilization over several seconds and
  record it with hardware, versions, dtype, CPU threads, and model dimensions.
  If GPU availability matters and another substantial job is active, defer
  the comparison; do not stop someone else's job.
- Keep checks bounded: a representative fixture, warmup, three synchronized
  repetitions, and a reversed-order comparison when timing variability could
  affect the conclusion. Do not start full suites or long training just for
  documentation edits. Record raw samples and distinguish training-step
  throughput from convergence or full-epoch time.
- A qNEP acceleration must preserve per-structure charge neutralization and
  parameter gradients through force loss. Check E/F/charge and force-loss
  gradients before timing it. Detached geometry/charge caches or ordinary
  NEP's analytical force are not drop-in replacements for this graph.

The runnable benchmark and portable raw measurements live in
[benchmarks/qnep/](benchmarks/qnep/), rather than relying on local `.omo` files.
