from __future__ import annotations

import os
import pickle
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

from .errors import QNEPError
from .model import QNEPModel
from .records import QNEPDatasetIdentities
from .run_config import QNEPRunConfig
from .training_checkpoint_schema import (
    CONFIG_FIELDS,
    FORMAT,
    AdamOptimizerState,
    MetricScalar,
    RuntimeMetadata,
    TrainingCheckpointState,
    parse_adam_state,
    parse_payload,
)

_MUTABLE_RESUME_FIELDS = {"output_dir", "epochs", "checkpoint_every", "resume_from"}


def resolved_run_config(config: QNEPRunConfig) -> dict[str, MetricScalar]:
    result: dict[str, MetricScalar] = {}
    for name in CONFIG_FIELDS:
        value = getattr(config, name)
        result[name] = str(value) if isinstance(value, Path) else value
    return result


def adam_optimizer_state(optimizer: torch.optim.Optimizer) -> AdamOptimizerState:
    return parse_adam_state(optimizer.state_dict())


def save_training_checkpoint(state: TrainingCheckpointState, path: str | Path) -> None:
    destination = Path(path)
    payload = _state_payload(state)
    parse_payload(payload)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def read_training_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> TrainingCheckpointState:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError) as error:
        raise QNEPError(f"invalid qNEP training checkpoint: {error}") from error
    return parse_payload(payload)


def validate_resume(
    state: TrainingCheckpointState,
    config: QNEPRunConfig,
    identities: QNEPDatasetIdentities,
    snapshot_id: str | None,
) -> None:
    current = resolved_run_config(config)
    for name in CONFIG_FIELDS:
        if (
            name not in _MUTABLE_RESUME_FIELDS
            and current[name] != state.resolved_config[name]
        ):
            raise QNEPError(f"resume config mismatch: {name}")
    saved_epochs = state.resolved_config["epochs"]
    if isinstance(saved_epochs, bool) or not isinstance(saved_epochs, int):
        raise QNEPError("malformed training checkpoint: resolved_config epochs")
    if config.epochs < saved_epochs:
        raise QNEPError("resume epochs cannot decrease")
    if identities.train != state.train_fingerprint:
        raise QNEPError("resume training data fingerprint mismatch")
    if identities.validation != state.validation_fingerprint:
        raise QNEPError("resume validation data fingerprint mismatch")
    if identities.settings != state.settings_fingerprint:
        raise QNEPError("resume loss/sampling settings fingerprint mismatch")
    if snapshot_id != state.snapshot_id:
        raise QNEPError("resume snapshot_id mismatch")
    runtime = RuntimeMetadata.current(torch.device(config.device), config.precision)
    for name in ("torch_version", "device", "precision"):
        if getattr(runtime, name) != getattr(state.runtime, name):
            label = "resolved device" if name == "device" else name.replace("_", " ")
            raise QNEPError(f"resume {label} mismatch")


def restore_training_state(
    state: TrainingCheckpointState,
    model: QNEPModel,
    optimizer: torch.optim.Optimizer,
    shuffle_generator: torch.Generator,
) -> None:
    if model.config != state.model_config:
        raise QNEPError("resume model config mismatch")
    current = model.state_dict()
    if current.keys() != state.model_state.keys():
        raise QNEPError("resume model state keys mismatch")
    for name, tensor in current.items():
        saved = state.model_state[name]
        if tensor.shape != saved.shape or tensor.dtype != saved.dtype:
            raise QNEPError(f"resume model state mismatch: {name}")
    _validate_restore_inputs(optimizer, shuffle_generator, state)
    model.load_state_dict(state.model_state, strict=True)
    optimizer.load_state_dict(dict(state.optimizer_state))
    torch.set_rng_state(state.cpu_rng_state)
    if state.device_rng_states:
        if not torch.cuda.is_available():
            raise QNEPError("resume CUDA RNG state requires CUDA")
        torch.cuda.set_rng_state_all(list(state.device_rng_states))
    shuffle_generator.set_state(state.shuffle_rng_state)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {_cpu(key): _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _state_payload(state: TrainingCheckpointState):
    return _cpu(
        {
            "format": FORMAT,
            "producer_version": state.producer_version,
            "model_config": state.model_config._asdict(),
            "model_state": dict(state.model_state),
            "optimizer_state": state.optimizer_state,
            "completed_epoch": state.completed_epoch,
            "optimizer_step": state.optimizer_step,
            "best_metric": state.best_metric,
            "best_epoch": state.best_epoch,
            "best_model_state": dict(state.best_model_state),
            "patience_count": state.patience_count,
            "train_fingerprint": state.train_fingerprint,
            "validation_fingerprint": state.validation_fingerprint,
            "settings_fingerprint": state.settings_fingerprint,
            "initial_model_hash": state.initial_model_hash,
            "snapshot_id": state.snapshot_id,
            "metrics_history": [dict(row) for row in state.metrics_history],
            "resolved_config": dict(state.resolved_config),
            "cpu_rng_state": state.cpu_rng_state,
            "device_rng_states": list(state.device_rng_states),
            "shuffle_rng_state": state.shuffle_rng_state,
            "runtime": asdict(state.runtime),
        }
    )


def _validate_optimizer_groups(
    optimizer: torch.optim.Optimizer, saved: AdamOptimizerState
) -> None:
    current_groups = optimizer.state_dict()["param_groups"]
    saved_groups = saved["param_groups"]
    if len(saved_groups) != len(current_groups):
        raise QNEPError("resume optimizer parameter groups mismatch")
    for current, candidate in zip(current_groups, saved_groups):
        parameters = candidate.get("params")
        if not isinstance(parameters, Sequence) or len(parameters) != len(
            current["params"]
        ):
            raise QNEPError("resume optimizer parameters mismatch")


def _validate_restore_inputs(
    optimizer: torch.optim.Optimizer,
    shuffle_generator: torch.Generator,
    state: TrainingCheckpointState,
) -> None:
    _validate_optimizer_groups(optimizer, state.optimizer_state)
    try:
        deepcopy(optimizer).load_state_dict(dict(state.optimizer_state))
        torch.Generator(device="cpu").set_state(state.cpu_rng_state)
        torch.Generator(device=shuffle_generator.device).set_state(
            state.shuffle_rng_state
        )
    except (KeyError, RuntimeError, ValueError) as error:
        raise QNEPError(f"invalid resumable state: {error}") from error
    if state.device_rng_states and not torch.cuda.is_available():
        raise QNEPError("resume CUDA RNG state requires CUDA")
    if len(state.device_rng_states) > torch.cuda.device_count():
        raise QNEPError("resume CUDA RNG state count exceeds available devices")
    for index, rng_state in enumerate(state.device_rng_states):
        try:
            torch.Generator(device=f"cuda:{index}").set_state(rng_state)
        except RuntimeError as error:
            raise QNEPError(f"invalid CUDA RNG state {index}") from error
