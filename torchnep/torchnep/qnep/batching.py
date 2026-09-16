from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .errors import QNEPError
from .model import QNEPModel
from .training import TrainingConfig, TrainingSample, per_sample_loss


def make_shuffle_generator(seed: int) -> torch.Generator:
    """Create the CPU generator whose state can be stored for training resume."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def shuffled_batches(
    record_count: int, batch_size: int, generator: torch.Generator
) -> tuple[tuple[int, ...], ...]:
    """Return one uniform permutation split into batches, including a short tail."""
    if record_count < 1:
        raise QNEPError("record_count must be positive")
    if batch_size < 1:
        raise QNEPError("batch_size must be positive")
    permutation = torch.randperm(record_count, generator=generator).tolist()
    return tuple(
        tuple(permutation[start : start + batch_size])
        for start in range(0, record_count, batch_size)
    )


def _validate_batch_scale(
    samples: Sequence[TrainingSample], global_record_count: int, global_weight: float
) -> None:
    if not samples:
        raise QNEPError("training batch requires at least one sample")
    if global_record_count < len(samples):
        raise QNEPError("global_record_count cannot be smaller than the batch")
    if not math.isfinite(global_weight) or global_weight <= 0:
        raise QNEPError("global_weight must be positive and finite")


def backward_weighted_batch(
    model: QNEPModel,
    samples: Sequence[TrainingSample],
    config: TrainingConfig,
    *,
    global_record_count: int,
    global_weight: float,
) -> float:
    """Accumulate the globally normalized mini-batch gradient without retained graphs."""
    _validate_batch_scale(samples, global_record_count, global_weight)
    coefficient = global_record_count / (len(samples) * global_weight)
    batch_loss = 0.0
    for sample in samples:
        loss = per_sample_loss(model, sample, config)
        contribution = coefficient * sample.weight * loss
        if not torch.isfinite(contribution):
            raise FloatingPointError("qNEP training loss became nonfinite")
        batch_loss += contribution.detach().item()
        contribution.backward()
        del contribution, loss
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("qNEP training gradient became nonfinite")
    return batch_loss
