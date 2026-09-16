from __future__ import annotations

from typing import TYPE_CHECKING

from .checkpoint import load_checkpoint, save_checkpoint
from .data_io import load_records
from .gpumd_import import load_gpumd_reference
from .model import QNEPConfig, QNEPModel, QNEPResult
from .records import QNEPRecord
from .run_config import QNEPDatasets, QNEPRunConfig, QNEPTrainingResult
from .structure import QNEPStructure
from .trainer import train_qnep
from .training import TrainingConfig, TrainingSample, fit, supervised_loss
from .training_checkpoint import read_training_checkpoint

if TYPE_CHECKING:
    from .calculator import QNEPCalculator

__all__ = [
    "QNEPCalculator",
    "QNEPConfig",
    "QNEPDatasets",
    "QNEPModel",
    "QNEPRecord",
    "QNEPResult",
    "QNEPRunConfig",
    "QNEPStructure",
    "QNEPTrainingResult",
    "TrainingConfig",
    "TrainingSample",
    "fit",
    "load_checkpoint",
    "load_gpumd_reference",
    "load_records",
    "read_training_checkpoint",
    "save_checkpoint",
    "supervised_loss",
    "train_qnep",
]


def __getattr__(name: str) -> type[QNEPCalculator]:
    if name == "QNEPCalculator":
        from .calculator import QNEPCalculator

        return QNEPCalculator
    raise AttributeError(name)
