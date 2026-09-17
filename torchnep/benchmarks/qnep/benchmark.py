# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "numpy", "ase"]
# ///
# How to run: from the worktree, preserve the existing CUDA environment:
# uv run --no-project "$QNEP_BENCH_PYTHON" \
#   torchnep/benchmarks/qnep/benchmark.py scale|batch|parity|parity8|confirm|threads
from __future__ import annotations

import copy
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import torch
from ase.io import iread

ROOT: Final = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "torchnep"))
from torchnep.qnep.batching import backward_weighted_batch
from torchnep.qnep.model import QNEPConfig, QNEPModel, QNEPResult
from torchnep.qnep.structure import QNEPStructure
from torchnep.qnep.training import TrainingConfig, TrainingSample

CONFIG: Final = QNEPConfig(type_names=("H", "O"))
LOSS_CONFIG: Final = TrainingConfig(energy_weight=1.0, force_weight=1.0, charge_weight=0.01)


def gpu_state() -> str:
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True).strip()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def base_structure(atoms: int) -> QNEPStructure:
    if atoms == 3:
        return QNEPStructure(
            torch.tensor([[1., 1., 1.], [1.8, 1., 1.], [.8, 1.7, 1.1]], dtype=torch.float64),
            torch.diag(torch.tensor([5., 5.5, 6.], dtype=torch.float64)),
            torch.tensor([1, 0, 0], dtype=torch.long),
        )
    water = next(iread(ROOT / "tests_pytest/fixtures/structures/water-nat63-from-md.xyz", index=0))
    if atoms > 63:
        water = water.repeat((2, 2, 1))
    assert len(water) == atoms
    return QNEPStructure(
        torch.tensor(water.positions, dtype=torch.float64),
        torch.tensor(water.cell.array, dtype=torch.float64),
        torch.tensor([0 if symbol == "H" else 1 for symbol in water.get_chemical_symbols()]),
    )


@dataclass(frozen=True, slots=True)
class Case:
    atoms: int
    dtype: torch.dtype
    device: torch.device
    records: int = 1
    threads: int = 1


def make_case(case: Case) -> tuple[QNEPModel, tuple[TrainingSample, ...]]:
    torch.set_num_threads(case.threads)
    torch.manual_seed(20260916)
    model = QNEPModel(CONFIG).double()
    base = base_structure(case.atoms)
    samples: list[TrainingSample] = []
    for index in range(case.records):
        shift = .002 * torch.sin(torch.arange(case.atoms * 3).reshape(case.atoms, 3) + index + 1)
        structure = base._replace(positions=base.positions + shift)
        prediction: QNEPResult = model(structure)
        energy = prediction.total_energy.detach().item() + .02 * case.atoms
        forces = prediction.forces.detach() + .02 * torch.cos(structure.positions)
        moved = QNEPStructure(
            structure.positions.to(device=case.device, dtype=case.dtype),
            structure.cell.to(device=case.device, dtype=case.dtype),
            structure.atom_types.to(device=case.device),
        )
        samples.append(TrainingSample(moved, energy, forces.to(device=case.device, dtype=case.dtype)))
    return model.to(device=case.device, dtype=case.dtype), tuple(samples)


def tensor_loss(model: QNEPModel, samples: tuple[TrainingSample, ...]) -> torch.Tensor:
    from batch_prototype import batch_forward
    prediction = batch_forward(model, tuple(sample.structure for sample in samples), True)
    targets_e = torch.tensor([sample.energy for sample in samples], device=prediction.total_energy.device,
                             dtype=prediction.total_energy.dtype)
    targets_f = torch.stack([sample.forces for sample in samples if sample.forces is not None])
    residual_e = (prediction.total_energy - targets_e) / len(samples[0].structure.positions)
    values = residual_e.square() + (prediction.forces - targets_f).square().mean(dim=(1, 2))
    values = values + .01 * prediction.raw_charges.sum(dim=1).square()
    if not torch.isfinite(values).all():
        raise FloatingPointError("nonfinite tensor-batch losses")
    return values.mean()


def tensor_backward(model: QNEPModel, samples: tuple[TrainingSample, ...]) -> float:
    loss = tensor_loss(model, samples)
    result = loss.detach().item()
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("nonfinite tensor-batch gradient")
    return result


def serial_backward(model: QNEPModel, samples: tuple[TrainingSample, ...]) -> float:
    return backward_weighted_batch(model, samples, LOSS_CONFIG,
                                   global_record_count=8, global_weight=8.0)


@dataclass(frozen=True, slots=True)
class Timing:
    stage: str
    atoms: int
    device: str
    dtype: str
    threads: int
    records: int
    batch: int
    method: str
    samples_ms: list[float]
    median_ms: float
    records_per_s: float
    optimizer_steps: int
    cuda_peak_allocated_mb: float
    gpu_before: str
    gpu_after: str


def measure(case: Case, batch: int, method: str) -> None:
    model, samples = make_case(case)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    seed_model = copy.deepcopy(model.state_dict())
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    seed_optimizer = copy.deepcopy(optimizer.state_dict())
    backward: Callable[[QNEPModel, tuple[TrainingSample, ...]], float] = serial_backward
    if method == "tensor":
        backward = tensor_backward

    def run() -> None:
        if method == "forward":
            model(samples[0].structure)
            return
        for start in range(0, len(samples), batch):
            optimizer.zero_grad(set_to_none=True)
            backward(model, samples[start:start + batch])
            optimizer.step()

    def reset() -> None:
        model.load_state_dict(seed_model)
        optimizer.load_state_dict(copy.deepcopy(seed_optimizer))
        optimizer.zero_grad(set_to_none=True)
        sync(case.device)

    before = gpu_state()
    for _ in range(2):
        reset()
        run()
    sync(case.device)
    if case.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(case.device)
    times: list[float] = []
    for _ in range(3):
        reset()
        start = time.perf_counter()
        run()
        sync(case.device)
        times.append((time.perf_counter() - start) * 1000)
    peak = torch.cuda.max_memory_allocated(case.device) / 2**20 if case.device.type == "cuda" else 0.
    row = Timing("timing", case.atoms, str(case.device), str(case.dtype), case.threads,
                 case.records, batch, method, times, statistics.median(times),
                 case.records * 1000 / statistics.median(times),
                 0 if method == "forward" else (case.records + batch - 1) // batch,
                 peak, before, gpu_state())
    print(json.dumps(asdict(row)), flush=True)


def scale() -> None:
    for device in (torch.device("cpu"), torch.device("cuda")):
        for atoms in (3, 63, 252):
            for method in ("forward", "serial"):
                measure(Case(atoms, torch.float32, device), 1, method)
        for method in ("forward", "serial"):
            measure(Case(63, torch.float64, device), 1, method)
    measure(Case(63, torch.float32, torch.device("cpu"), threads=12), 1, "serial")


def batch() -> None:
    for device in (torch.device("cpu"), torch.device("cuda")):
        for size in (1, 4, 8):
            methods = ("serial",) if size == 1 else ("serial", "tensor")
            for method in methods:
                measure(Case(63, torch.float32, device, records=8), size, method)


def parity(batch_size: int = 4) -> None:
    from batch_prototype import batch_forward
    for dtype, device in ((torch.float64, torch.device("cpu")), (torch.float32, torch.device("cuda"))):
        model, samples = make_case(Case(63, dtype, device, records=batch_size))
        expected: list[QNEPResult] = [model(sample.structure) for sample in samples]
        actual = batch_forward(model, tuple(sample.structure for sample in samples))
        tolerance = 2e-8 if dtype == torch.float64 else 2e-4
        absolute = 1e-11 if dtype == torch.float64 else 1e-7
        errors: list[float] = []
        for field in range(6):
            reference = torch.stack([result[field] for result in expected])
            torch.testing.assert_close(actual[field], reference, atol=absolute, rtol=tolerance)
            errors.append((actual[field] - reference).abs().max().item())
        model.zero_grad(set_to_none=True)
        loss_ref = serial_backward(model, samples)
        gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        model.zero_grad(set_to_none=True)
        loss_batch = tensor_backward(model, samples)
        differences: list[float] = []
        for name, parameter in model.named_parameters():
            if name in gradients:
                assert parameter.grad is not None
                torch.testing.assert_close(parameter.grad, gradients[name], atol=absolute, rtol=tolerance)
                differences.append((parameter.grad - gradients[name]).abs().max().item())
        assert sum(gradient.abs().sum().item() for gradient in gradients.values()) > 0
        assert abs(loss_ref - loss_batch) <= tolerance * max(1., abs(loss_ref))
        print(json.dumps({"stage": "parity", "device": str(device), "dtype": str(dtype),
                          "atoms": 63, "batch": batch_size, "field_max_errors": errors,
                          "gradient_max_error": max(differences), "loss_reference": loss_ref,
                          "loss_tensor": loss_batch, "verdict": "PASS"}), flush=True)


def confirm() -> None:
    for device in (torch.device("cpu"), torch.device("cuda")):
        for method in ("tensor", "serial"):
            measure(Case(63, torch.float32, device, records=8), 8, method)


def threads() -> None:
    visible = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for count in sorted({1, min(2, visible), min(4, visible), min(12, visible), visible}):
        for atoms, records, size, method in ((63, 1, 1, "serial"), (252, 1, 1, "serial"), (63, 8, 8, "tensor")):
            measure(Case(atoms, torch.float32, torch.device("cpu"), records=records, threads=count), size, method)


if __name__ == "__main__":
    commands = {"scale": scale, "batch": batch, "parity": parity,
                "parity8": lambda: parity(8), "confirm": confirm, "threads": threads}
    usage = "Usage: benchmark.py {scale|batch|parity|parity8|confirm|threads}"
    if len(sys.argv) == 2 and sys.argv[1] in {"--help", "-h"}:
        print(usage)
        raise SystemExit(0)
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        print(usage, file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({"stage": "environment", "head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "gpu_state": gpu_state(), "config": CONFIG._asdict(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "logical_cpus": os.cpu_count(), "default_interop_threads": torch.get_num_interop_threads(),
        "source": str(Path(inspect.getfile(QNEPModel)).resolve())}), flush=True)
    commands[sys.argv[1]]()
