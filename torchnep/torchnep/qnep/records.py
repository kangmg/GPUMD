from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch

from .errors import QNEPError
from .training import TrainingSample

if TYPE_CHECKING:
    from .model import QNEPModel


@dataclass(frozen=True)
class QNEPRecord:
    sample: TrainingSample
    record_id: str | None = None
    group_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample, TrainingSample):
            raise QNEPError("sample must be a TrainingSample")
        self.sample.validate()
        for name, value in (("record_id", self.record_id), ("group_id", self.group_id)):
            if value is not None and (not isinstance(value, str) or not value):
                raise QNEPError(f"{name} must be a nonempty string or absent")


@dataclass(frozen=True)
class QNEPDatasetIdentities:
    train: str
    validation: str
    settings: str


class _HashSink(Protocol):
    def update(self, value: bytes, /) -> None: ...


class _DatasetsView(Protocol):
    @property
    def train(self) -> Sequence[QNEPRecord]: ...

    @property
    def validation(self) -> Sequence[QNEPRecord]: ...


class _RunConfigView(Protocol):
    @property
    def batch_size(self) -> int: ...

    @property
    def learning_rate(self) -> float: ...

    @property
    def energy_weight(self) -> float: ...

    @property
    def force_weight(self) -> float: ...

    @property
    def charge_weight(self) -> float: ...

    @property
    def seed(self) -> int: ...

    @property
    def device(self) -> str: ...

    @property
    def precision(self) -> str: ...


def _add_bytes(hasher: _HashSink, value: bytes) -> None:
    hasher.update(struct.pack("<Q", len(value)))
    hasher.update(value)


def _add_text(hasher: _HashSink, value: str | None) -> None:
    if value is None:
        hasher.update(b"\x00")
        return
    hasher.update(b"\x01")
    _add_bytes(hasher, value.encode("utf-8"))


def _float_bytes(value: torch.Tensor) -> bytes:
    array = value.detach().cpu().to(torch.float64).numpy()
    array = np.array(array, dtype="<f8", order="C", copy=True)
    array[array == 0] = 0.0
    return array.tobytes()


def geometry_fingerprint(record: QNEPRecord, type_names: tuple[str, ...]) -> str:
    hasher = hashlib.sha256(b"torchnep-qnep-geometry-v1")
    structure = record.sample.structure
    for name in type_names:
        _add_text(hasher, name)
    _add_bytes(hasher, structure.atom_types.detach().cpu().to(torch.int64).numpy().astype("<i8").tobytes())
    _add_bytes(hasher, _float_bytes(structure.positions))
    _add_bytes(hasher, _float_bytes(structure.cell))
    hasher.update(bytes(structure.pbc))
    hasher.update(struct.pack("<d", float(structure.total_charge)))
    return hasher.hexdigest()


def record_fingerprint(record: QNEPRecord, type_names: tuple[str, ...]) -> str:
    hasher = hashlib.sha256(b"torchnep-qnep-record-v1")
    _add_text(hasher, geometry_fingerprint(record, type_names))
    sample = record.sample
    if sample.energy is None:
        hasher.update(b"\x00")
    else:
        hasher.update(b"\x01")
        hasher.update(struct.pack("<d", sample.energy))
    if sample.forces is None:
        hasher.update(b"\x00")
    else:
        hasher.update(b"\x01")
        mask = (
            torch.ones_like(sample.forces, dtype=torch.bool)
            if sample.force_mask is None
            else sample.force_mask
        )
        hasher.update(b"\x00" if sample.force_mask is None else b"\x01")
        _add_bytes(hasher, mask.detach().cpu().numpy().tobytes())
        canonical = torch.where(mask, sample.forces, torch.zeros_like(sample.forces))
        _add_bytes(hasher, _float_bytes(canonical))
    hasher.update(struct.pack("<d", sample.weight))
    _add_text(hasher, record.record_id)
    _add_text(hasher, record.group_id)
    return hasher.hexdigest()


def _split_fingerprint(
    split: str, records: tuple[QNEPRecord, ...], type_names: tuple[str, ...]
) -> str:
    hasher = hashlib.sha256(b"torchnep-qnep-split-v1")
    _add_text(hasher, split)
    hasher.update(struct.pack("<Q", len(records)))
    for record in records:
        _add_text(hasher, record_fingerprint(record, type_names))
    return hasher.hexdigest()


def _settings_fingerprint(config: _RunConfigView) -> str:
    settings = {
        "batch_size": config.batch_size,
        "charge_weight": config.charge_weight,
        "device": config.device,
        "energy_weight": config.energy_weight,
        "force_weight": config.force_weight,
        "learning_rate": config.learning_rate,
        "precision": config.precision,
        "seed": config.seed,
    }
    payload = json.dumps(settings, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_record(
    record: QNEPRecord, model: QNEPModel, config: _RunConfigView
) -> None:
    sample = record.sample
    structure = sample.structure
    sample.validate()
    structure.validate(len(model.config.type_names))
    if structure.positions.device.type != "cpu":
        raise QNEPError("QNEPRecord structures must be CPU-resident")
    if sample.forces is not None and (
        sample.forces.device.type != "cpu"
        or sample.forces.dtype != structure.positions.dtype
    ):
        raise QNEPError("forces must be CPU-resident and match the structure dtype")
    if sample.force_mask is not None and sample.force_mask.device.type != "cpu":
        raise QNEPError("force_mask must be CPU-resident")
    energy_active = sample.energy is not None and config.energy_weight > 0
    force_active = sample.forces is not None and config.force_weight > 0
    if force_active and sample.force_mask is not None:
        force_active = bool(sample.force_mask.any())
    if not energy_active and not force_active:
        raise QNEPError("each record requires an active energy or force target")


def validate_datasets(
    datasets: _DatasetsView, model: QNEPModel, config: _RunConfigView
) -> QNEPDatasetIdentities:
    train = tuple(datasets.train)
    validation = tuple(datasets.validation)
    for record in train + validation:
        _validate_record(record, model, config)
    train_geometry = {
        geometry_fingerprint(record, model.config.type_names) for record in train
    }
    validation_geometry = {
        geometry_fingerprint(record, model.config.type_names) for record in validation
    }
    if train_geometry & validation_geometry:
        raise QNEPError("identical geometry appears in both train and validation splits")
    train_record_ids = {record.record_id for record in train if record.record_id is not None}
    validation_record_ids = {
        record.record_id for record in validation if record.record_id is not None
    }
    if train_record_ids & validation_record_ids:
        raise QNEPError("record_id appears in both train and validation splits")
    train_group_ids = {record.group_id for record in train if record.group_id is not None}
    validation_group_ids = {
        record.group_id for record in validation if record.group_id is not None
    }
    if train_group_ids & validation_group_ids:
        raise QNEPError("group_id appears in both train and validation splits")
    return QNEPDatasetIdentities(
        train=_split_fingerprint("train", train, model.config.type_names),
        validation=_split_fingerprint(
            "validation", validation, model.config.type_names
        ),
        settings=_settings_fingerprint(config),
    )
