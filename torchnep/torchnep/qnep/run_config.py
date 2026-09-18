from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from .errors import QNEPError

if TYPE_CHECKING:
    from .records import QNEPRecord

Precision = Literal["float32", "float64"]
StopReason = Literal["completed", "early_stopping"]


@dataclass(frozen=True)
class QNEPDatasets:
    train: Sequence[QNEPRecord]
    validation: Sequence[QNEPRecord]
    snapshot_id: str | None = None

    def __post_init__(self) -> None:
        train = tuple(self.train)
        validation = tuple(self.validation)
        if not train or not validation:
            raise QNEPError("train and validation datasets must both be nonempty")
        if self.snapshot_id is not None and not isinstance(self.snapshot_id, str):
            raise QNEPError("snapshot_id must be a string or absent")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "validation", validation)


@dataclass(frozen=True)
class QNEPRunConfig:
    output_dir: Path
    epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 0.001
    energy_weight: float = 1.0
    force_weight: float = 1.0
    charge_weight: float = 0.01
    seed: int = 0
    device: str = "cpu"
    precision: Precision = "float64"
    early_stop_patience: int | None = None
    checkpoint_every: int = 1
    resume_from: Path | None = None

    def __post_init__(self) -> None:
        output_dir = Path(self.output_dir)
        resume_from = None if self.resume_from is None else Path(self.resume_from)
        if self.epochs < 1:
            raise QNEPError("epochs must be positive")
        if self.batch_size < 1:
            raise QNEPError("batch_size must be positive")
        if self.checkpoint_every < 1:
            raise QNEPError("checkpoint_every must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise QNEPError("learning_rate must be positive and finite")
        weights = (self.energy_weight, self.force_weight, self.charge_weight)
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise QNEPError("loss weights must be finite and nonnegative")
        if self.energy_weight == 0 and self.force_weight == 0:
            raise QNEPError("energy_weight and force_weight cannot both be zero")
        if self.precision not in ("float32", "float64"):
            raise QNEPError("precision must be float32 or float64")
        if self.early_stop_patience is not None and self.early_stop_patience < 1:
            raise QNEPError("early_stop_patience must be positive when provided")
        device = self._resolve_device(self.device)
        if output_dir.exists() and not output_dir.is_dir():
            raise QNEPError("output_dir must be a directory path")
        if output_dir.is_dir() and any(output_dir.iterdir()) and resume_from is None:
            raise QNEPError("nonempty output_dir requires explicit resume_from")
        if resume_from is not None and not resume_from.is_file():
            raise QNEPError("resume_from must name an existing training checkpoint")
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "resume_from", resume_from)
        object.__setattr__(self, "device", device)

    @staticmethod
    def _resolve_device(value: str) -> str:
        try:
            device = torch.device(value)
        except RuntimeError as error:
            raise QNEPError(f"invalid device {value!r}") from error
        if device.type == "cpu":
            return "cpu"
        if device.type != "cuda":
            raise QNEPError("device must be an available CPU or CUDA device")
        if not torch.cuda.is_available():
            raise QNEPError("CUDA device requested but CUDA is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise QNEPError(f"CUDA device index {index} is unavailable")
        return f"cuda:{index}"


@dataclass(frozen=True)
class QNEPTrainingResult:
    completed_epoch: int
    optimizer_step: int
    best_epoch: int
    stop_reason: StopReason
    best_inference_path: Path
    latest_inference_path: Path
    training_checkpoint_path: Path

    @property
    def nep_path(self) -> Path:
        """Canonical native inference model selected by validation."""
        return self.best_inference_path.parent / "nep.txt"

    @property
    def best_nep_path(self) -> Path:
        """Native model from the best validation epoch."""
        return self.best_inference_path.parent / "nep_best.txt"

    @property
    def latest_nep_path(self) -> Path:
        """Native model from the latest saved epoch."""
        return self.latest_inference_path.parent / "nep_last.txt"
