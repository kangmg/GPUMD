from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import torch

from .. import __version__
from .checkpoint import save_checkpoint
from .errors import QNEPError
from .model import QNEPConfig, QNEPModel
from .records import QNEPDatasetIdentities, QNEPRecord
from .run_config import QNEPRunConfig
from .structure import QNEPStructure
from .training import TrainingConfig, TrainingSample
from .training_checkpoint import resolved_run_config
from .training_checkpoint_schema import MetricScalar

_OWNED_FILES = {
    "best.qnep.pt",
    "last.training.pt",
    "latest.qnep.pt",
    "nep.txt",
    "nep_best.txt",
    "nep_last.txt",
    "metrics.jsonl",
    "run.json",
}


class SplitMetrics(NamedTuple):
    objective: float
    supervised_objective: float
    raw_charge_penalty: float
    energy_rmse: float | None
    force_rmse: float | None
    record_count: int
    atom_count: int
    energy_record_count: int
    force_record_count: int
    force_component_count: int


EpochMetricsRow = Mapping[str, MetricScalar]


def sample_to_model(sample: TrainingSample, model: QNEPModel) -> TrainingSample:
    reference = model.nep.b1
    structure = sample.structure
    moved = QNEPStructure(
        positions=structure.positions.to(reference),
        cell=structure.cell.to(reference),
        atom_types=structure.atom_types.to(device=reference.device),
        pbc=structure.pbc,
        total_charge=structure.total_charge,
    )
    forces = None if sample.forces is None else sample.forces.to(reference)
    mask = (
        None
        if sample.force_mask is None
        else sample.force_mask.to(device=reference.device)
    )
    return TrainingSample(moved, sample.energy, forces, mask, sample.weight)


def evaluate_metrics(
    model: QNEPModel,
    records: Sequence[QNEPRecord],
    config: TrainingConfig,
) -> SplitMetrics:
    total_weight = sum(record.sample.weight for record in records)
    energy_square = 0.0
    force_square = 0.0
    raw_charge_square = 0.0
    energy_weight = 0.0
    force_weight = 0.0
    atom_count = 0
    energy_count = 0
    force_count = 0
    component_count = 0
    model.eval()
    for record in records:
        sample = sample_to_model(record.sample, model)
        prediction = model(sample.structure, create_graph=False)
        weight = sample.weight
        atom_count += len(sample.structure.positions)
        raw_charge_square += weight * prediction.raw_charges.detach().sum().square().item()
        if sample.energy is not None:
            residual = (
                prediction.total_energy.detach().item() - sample.energy
            ) / len(sample.structure.positions)
            energy_square += weight * residual * residual
            energy_weight += weight
            energy_count += 1
        if sample.forces is not None:
            predicted_forces = prediction.forces.detach()
            if sample.force_mask is None:
                difference = predicted_forces - sample.forces
            else:
                difference = (
                    predicted_forces[sample.force_mask] - sample.forces[sample.force_mask]
                )
            if difference.numel() > 0:
                force_square += weight * difference.square().mean().item()
                force_weight += weight
                force_count += 1
                component_count += difference.numel()
        del prediction
    supervised = (
        config.energy_weight * energy_square + config.force_weight * force_square
    ) / total_weight
    raw_penalty = raw_charge_square / total_weight
    return SplitMetrics(
        objective=supervised + config.charge_weight * raw_penalty,
        supervised_objective=supervised,
        raw_charge_penalty=raw_penalty,
        energy_rmse=None if energy_weight == 0 else math.sqrt(energy_square / energy_weight),
        force_rmse=None if force_weight == 0 else math.sqrt(force_square / force_weight),
        record_count=len(records),
        atom_count=atom_count,
        energy_record_count=energy_count,
        force_record_count=force_count,
        force_component_count=component_count,
    )


def epoch_metrics_row(
    epoch: int,
    optimizer_step: int,
    train: SplitMetrics,
    validation: SplitMetrics,
) -> EpochMetricsRow:
    row: dict[str, MetricScalar] = {"epoch": epoch, "optimizer_step": optimizer_step}
    for prefix, metrics in (("train", train), ("validation", validation)):
        for field, value in metrics._asdict().items():
            row[f"{prefix}_{field}"] = value
    return row


def model_content_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"torchnep-qnep-model-state-v1")
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_save_inference(model: QNEPModel, path: Path) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        save_checkpoint(model, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_best_inference(model: QNEPModel, path: Path) -> None:
    atomic_save_inference(model, path)
    native_path = path.parent / "nep_best.txt"
    model.export_nep(native_path)
    atomic_write_text(path.parent / "nep.txt", native_path.read_text(encoding="utf-8"))


def save_best_inference_state(
    config: QNEPConfig, state: Mapping[str, torch.Tensor], path: Path
) -> None:
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        saved_model = QNEPModel(config).to(dtype=state["nep.b1"].dtype)
        saved_model.load_state_dict(state, strict=True)
        save_best_inference(saved_model, path)


def validate_output_dir(config: QNEPRunConfig) -> None:
    output = config.output_dir
    if output.exists() and not output.is_dir():
        raise QNEPError("output_dir must be a directory path")
    entries = set() if not output.exists() else {path.name for path in output.iterdir()}
    if config.resume_from is None:
        if entries:
            raise QNEPError("nonempty output_dir requires explicit resume_from")
        return
    if entries - _OWNED_FILES:
        raise QNEPError("resume output_dir contains files not owned by this qNEP run")
    expected = (output / "last.training.pt").resolve()
    if entries and config.resume_from.resolve() != expected:
        raise QNEPError("nonempty output_dir can resume only its last.training.pt")


def run_manifest(
    config: QNEPRunConfig,
    identities: QNEPDatasetIdentities,
    snapshot_id: str | None,
    initial_model_hash: str,
) -> str:
    payload = {
        "producer_version": __version__,
        "initial_model_hash": initial_model_hash,
        "snapshot_id": snapshot_id,
        "train_fingerprint": identities.train,
        "validation_fingerprint": identities.validation,
        "settings_fingerprint": identities.settings,
        "config": resolved_run_config(config),
    }
    return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"


def metrics_history_text(history: Sequence[Mapping[str, MetricScalar]]) -> str:
    return "".join(
        json.dumps(dict(row), allow_nan=False, sort_keys=True) + "\n" for row in history
    )
