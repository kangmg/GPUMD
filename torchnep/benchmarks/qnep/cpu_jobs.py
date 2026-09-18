# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "numpy", "ase", "threadpoolctl"]
# ///
# Run with an existing environment, from the repository root:
# uv run --no-project "$QNEP_BENCH_PYTHON" torchnep/benchmarks/qnep/cpu_jobs.py serial
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import assert_never


class CPUJobsError(RuntimeError):
    pass


class Command(StrEnum):
    RESET = "RESET"
    RUN = "RUN"
    QUIT = "QUIT"


@dataclass(frozen=True, slots=True)
class Settings:
    method: str
    threads: int
    jobs: int
    updates: int
    repeats: int


@dataclass(frozen=True, slots=True)
class Result:
    seconds: float
    loss: float
    parameter_change_linf: float
    repeat_error_linf: float
    peak_rss_kib: float


@dataclass(frozen=True, slots=True)
class Round:
    condition: str
    repetition: int
    group_seconds: float
    records_per_second: float
    workers: list[Result]


def worker(settings: Settings) -> None:
    import torch
    from threadpoolctl import threadpool_info

    from benchmark import Case, make_case, serial_backward, tensor_backward

    torch.set_num_threads(settings.threads)
    torch.set_num_interop_threads(1)
    model, samples = make_case(Case(63, torch.float32, torch.device("cpu"), 8, settings.threads))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    seed_model = copy.deepcopy(model.state_dict())
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    seed_optimizer = copy.deepcopy(optimizer.state_dict())
    seed_parameters = torch.cat([p.detach().flatten() for p in model.parameters()])
    reference: torch.Tensor | None = None
    reference_loss: float | None = None
    backward = serial_backward if settings.method == "serial" else tensor_backward

    def reset() -> None:
        model.load_state_dict(seed_model)
        optimizer.load_state_dict(copy.deepcopy(seed_optimizer))
        optimizer.zero_grad(set_to_none=True)

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        loss = backward(model, samples)
        optimizer.step()
        return loss

    for _ in range(2):
        reset()
        step()
    pools: list[str] = []
    for pool in threadpool_info():
        count = pool["num_threads"]
        if count != settings.threads:
            raise CPUJobsError(f"Uncapped thread pool: {pool['internal_api']}={count}")
        pools.append(f"{pool['internal_api']}={count}")
    print("\t".join(["READY", str(torch.__version__), str(torch.get_num_threads()),
                     str(torch.get_num_interop_threads()), *pools]), flush=True)
    for command in sys.stdin:
        message = Command(command.strip())
        match message:
            case Command.RESET:
                reset()
                print("RESET", flush=True)
            case Command.RUN:
                start = time.perf_counter()
                loss = 0.0
                for _ in range(settings.updates):
                    loss = step()
                seconds = time.perf_counter() - start
                final = torch.cat([p.detach().flatten() for p in model.parameters()])
                change = (final - seed_parameters).abs().max().item()
                if not math.isfinite(loss) or not torch.isfinite(final).all() or change <= 0:
                    raise FloatingPointError("Expected finite loss and a real parameter update")
                error = 0.0
                if reference is None:
                    reference, reference_loss = final.clone(), loss
                else:
                    torch.testing.assert_close(final, reference, atol=1e-7, rtol=1e-5)
                    if reference_loss is None or not math.isclose(loss, reference_loss, abs_tol=1e-7, rel_tol=1e-5):
                        raise AssertionError("Loss changed between equal-work conditions")
                    error = (final - reference).abs().max().item()
                values = (seconds, loss, change, error, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                print("\t".join(str(value) for value in values), flush=True)
            case Command.QUIT:
                return
            case _:
                assert_never(message)


def send(child: subprocess.Popen[str], command: str) -> None:
    assert child.stdin is not None
    child.stdin.write(command + "\n")
    child.stdin.flush()


def receive(child: subprocess.Popen[str]) -> str:
    assert child.stdout is not None
    line = child.stdout.readline().strip()
    if not line:
        raise CPUJobsError(f"Worker {child.pid} stopped before responding")
    return line


def result(child: subprocess.Popen[str]) -> Result:
    elapsed, loss, change, error, rss = (float(value) for value in receive(child).split("\t"))
    return Result(elapsed, loss, change, error, rss)


def stop(child: subprocess.Popen[str]) -> None:
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    if child.stdin is not None:
        child.stdin.close()
    if child.stdout is not None:
        child.stdout.close()


def benchmark(settings: Settings) -> None:
    caps = {key: str(settings.threads) for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    )}
    caps["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, str(Path(__file__).resolve()), settings.method, "--worker",
               "--threads", str(settings.threads), "--updates", str(settings.updates)]
    children: list[subprocess.Popen[str]] = []
    rows: list[Round] = []
    load_before = os.getloadavg()
    start = time.perf_counter()
    with ExitStack() as stack:
        for _ in range(settings.jobs):
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, env=os.environ | caps)
            stack.callback(stop, child)
            children.append(child)
        ready = [receive(child) for child in children]
        if any(not line.startswith("READY\t") for line in ready):
            raise CPUJobsError("Worker did not complete warmup")
        startup = time.perf_counter() - start
        print(json.dumps({"stage": "ready", "pool_startup_seconds": startup,
                          "workers": ready}), flush=True)
        for repetition in range(settings.repeats):
            conditions = ("sequential", "concurrent") if repetition % 2 == 0 else ("concurrent", "sequential")
            for condition in conditions:
                for child in children:
                    send(child, "RESET")
                if any(receive(child) != "RESET" for child in children):
                    raise CPUJobsError("Worker reset failed")
                start = time.perf_counter()
                outcomes: list[Result] = []
                if condition == "sequential":
                    for child in children:
                        send(child, "RUN")
                        outcomes.append(result(child))
                else:
                    for child in children:
                        send(child, "RUN")
                    outcomes = [result(child) for child in children]
                elapsed = time.perf_counter() - start
                row = Round(condition, repetition, elapsed, settings.jobs * settings.updates * 8 / elapsed, outcomes)
                rows.append(row)
                print(json.dumps({"stage": "round", **asdict(row)}), flush=True)
        for child in children:
            send(child, "QUIT")
        for child in children:
            if child.wait(timeout=10) != 0:
                raise CPUJobsError("Worker failed at shutdown")
    medians = {condition: statistics.median(row.group_seconds for row in rows if row.condition == condition)
               for condition in ("sequential", "concurrent")}
    print(json.dumps({"stage": "summary", "settings": asdict(settings), "atoms": 63,
                      "records_per_update": 8, "device": "cpu", "dtype": "float32", "warmups": 2,
                      "logical_cpus": os.cpu_count(), "affinity_cpus": len(os.sched_getaffinity(0)),
                      "affinity_pinned": False, "environment_caps": caps, "interop_threads": 1,
                      "pool_startup_seconds": startup, "load_before": load_before, "load_after": os.getloadavg(),
                      "median_group_seconds": medians, "speedup": medians["sequential"] / medians["concurrent"],
                      "workers_exited_zero": True}), flush=True)


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare equal-work CPU jobs in a warmed process pool (Linux/WSL).")
    parser.add_argument("method", choices=("serial", "tensor"))
    parser.add_argument("--threads", type=positive, default=2)
    parser.add_argument("--jobs", type=positive, default=5)
    parser.add_argument("--updates", type=positive, default=5)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    settings = Settings(args.method, args.threads, args.jobs, args.updates, args.repeats)
    if args.worker:
        worker(settings)
    else:
        benchmark(settings)
