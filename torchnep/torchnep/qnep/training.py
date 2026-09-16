from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, NamedTuple

import torch

from .errors import QNEPError
from .model import QNEPModel
from .structure import QNEPStructure


class TrainingSample(NamedTuple):
    structure: QNEPStructure
    energy: float | None = None
    forces: torch.Tensor | None = None
    force_mask: torch.Tensor | None = None
    weight: float = 1.0

    def validate(self) -> None:
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise QNEPError("sample weight must be positive and finite")
        if self.energy is not None and not math.isfinite(self.energy):
            raise QNEPError("energy must be finite or absent")
        if self.force_mask is not None:
            if (
                self.forces is None
                or self.force_mask.shape != self.structure.positions.shape
            ):
                raise QNEPError("force_mask requires forces and must have shape (N, 3)")
            if self.force_mask.dtype != torch.bool:
                raise QNEPError("force_mask must be boolean")
        if self.forces is not None:
            if self.forces.shape != self.structure.positions.shape:
                raise QNEPError("forces must have shape (N, 3)")
            present = (
                self.forces if self.force_mask is None else self.forces[self.force_mask]
            )
            if not torch.isfinite(present).all():
                raise QNEPError("present forces must be finite")


class TrainingConfig(NamedTuple):
    steps: int = 100
    learning_rate: float = 0.001
    energy_weight: float = 1.0
    force_weight: float = 1.0
    charge_weight: float = 0.01


DEFAULT_TRAINING_CONFIG: Final = TrainingConfig()


def _validate_loss_weights(config: TrainingConfig) -> None:
    for value in (config.energy_weight, config.force_weight, config.charge_weight):
        if not math.isfinite(value) or value < 0:
            raise QNEPError("loss weights must be finite and nonnegative")


def per_sample_loss(
    model: QNEPModel,
    sample: TrainingSample,
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """Return one unweighted structure loss with its coordinate derivative graph."""
    _validate_loss_weights(config)
    sample.validate()
    prediction = model(sample.structure, create_graph=create_graph)
    loss = config.charge_weight * prediction.raw_charges.sum().square()
    if sample.energy is not None:
        residual = (prediction.total_energy - sample.energy) / len(
            sample.structure.positions
        )
        loss = loss + config.energy_weight * residual.square()
    if sample.forces is not None:
        forces = sample.forces.to(prediction.forces)
        if sample.force_mask is None:
            difference = prediction.forces - forces
        else:
            mask = sample.force_mask.to(prediction.forces.device)
            difference = prediction.forces[mask] - forces[mask]
        if difference.numel():
            loss = loss + config.force_weight * difference.square().mean()
    return loss


def supervised_loss(
    model: QNEPModel,
    samples: Sequence[TrainingSample],
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
) -> torch.Tensor:
    """Weighted structure mean of E/N, present force components, and raw sum(Q) errors."""
    if not samples:
        raise QNEPError("training requires at least one sample")
    _validate_loss_weights(config)
    total = model.nep.b1 * 0.0
    denominator = 0.0
    for sample in samples:
        total = total + sample.weight * per_sample_loss(model, sample, config)
        denominator += sample.weight
    return total / denominator


def fit(
    model: QNEPModel,
    samples: Sequence[TrainingSample],
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
) -> tuple[float, ...]:
    if (
        config.steps < 1
        or not math.isfinite(config.learning_rate)
        or config.learning_rate <= 0
    ):
        raise QNEPError("steps and learning_rate must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses = []
    model.train()
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = supervised_loss(model, samples, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("qNEP training loss became nonfinite")
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
    return tuple(losses)
