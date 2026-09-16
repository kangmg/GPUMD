from __future__ import annotations

import math
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict, Union

import torch

from .errors import QNEPError
from .model import QNEPConfig

FORMAT = "torchnep-qnep-mode2-training-v1"
CONFIG_FIELDS = (
    "output_dir",
    "epochs",
    "batch_size",
    "learning_rate",
    "energy_weight",
    "force_weight",
    "charge_weight",
    "seed",
    "device",
    "precision",
    "early_stop_patience",
    "checkpoint_every",
    "resume_from",
)
MetricScalar = Union[int, float, str, bool, None]
AdamGroupValue = Union[int, float, bool, str, None, list[int], tuple[float, float]]


class AdamOptimizerState(TypedDict):
    state: dict[int, dict[str, torch.Tensor]]
    param_groups: list[dict[str, AdamGroupValue]]


@dataclass(frozen=True)
class RuntimeMetadata:
    torch_version: str
    python_version: str
    platform: str
    cuda_version: str | None
    device: str
    precision: str
    default_dtype: str
    deterministic_algorithms: bool

    @classmethod
    def current(cls, device: torch.device, precision: str) -> RuntimeMetadata:
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return cls(
            str(torch.__version__),
            platform.python_version(),
            platform.platform(),
            torch.version.cuda,
            str(device),
            precision,
            str(torch.get_default_dtype()),
            torch.are_deterministic_algorithms_enabled(),
        )


@dataclass(frozen=True)
class TrainingCheckpointState:
    producer_version: str
    model_config: QNEPConfig
    model_state: Mapping[str, torch.Tensor]
    optimizer_state: AdamOptimizerState
    completed_epoch: int
    optimizer_step: int
    best_metric: float
    best_epoch: int
    best_model_state: Mapping[str, torch.Tensor]
    patience_count: int
    train_fingerprint: str
    validation_fingerprint: str
    settings_fingerprint: str
    initial_model_hash: str
    snapshot_id: str | None
    metrics_history: tuple[Mapping[str, MetricScalar], ...]
    resolved_config: Mapping[str, MetricScalar]
    cpu_rng_state: torch.Tensor
    device_rng_states: tuple[torch.Tensor, ...]
    shuffle_rng_state: torch.Tensor
    runtime: RuntimeMetadata


def parse_payload(payload) -> TrainingCheckpointState:
    checkpoint_format = _need(payload, "format")
    if checkpoint_format == "torchnep-qnep-mode2-reference-v1":
        raise QNEPError("inference checkpoint cannot be used to resume training")
    if checkpoint_format != FORMAT:
        raise QNEPError("unsupported qNEP training checkpoint format")
    producer = _string(payload, "producer_version")
    try:
        model_config = QNEPConfig(
            **_mapping(_need(payload, "model_config"), "model_config")
        )
    except (TypeError, ValueError) as error:
        raise QNEPError("malformed training checkpoint: model_config") from error
    model_state = _tensor_map(_need(payload, "model_state"), "model_state")
    best_state = _tensor_map(_need(payload, "best_model_state"), "best_model_state")
    if model_state.keys() != best_state.keys():
        raise QNEPError("malformed training checkpoint: best model keys differ")
    for name, tensor in model_state.items():
        best_tensor = best_state[name]
        if tensor.shape != best_tensor.shape or tensor.dtype != best_tensor.dtype:
            raise QNEPError("malformed training checkpoint: best model shape differs")
    snapshot_id = _need(payload, "snapshot_id")
    if snapshot_id is not None and not isinstance(snapshot_id, str):
        raise QNEPError("malformed training checkpoint: snapshot_id")
    state = TrainingCheckpointState(
        producer,
        model_config,
        model_state,
        parse_adam_state(_need(payload, "optimizer_state")),
        _integer(payload, "completed_epoch"),
        _integer(payload, "optimizer_step"),
        _finite(payload, "best_metric"),
        _integer(payload, "best_epoch"),
        best_state,
        _integer(payload, "patience_count"),
        _fingerprint(payload, "train_fingerprint"),
        _fingerprint(payload, "validation_fingerprint"),
        _fingerprint(payload, "settings_fingerprint"),
        _fingerprint(payload, "initial_model_hash"),
        snapshot_id,
        _metrics(_need(payload, "metrics_history")),
        _config(_need(payload, "resolved_config")),
        _rng(payload, "cpu_rng_state"),
        _device_rng(_need(payload, "device_rng_states")),
        _rng(payload, "shuffle_rng_state"),
        _runtime(_need(payload, "runtime")),
    )
    _validate_semantics(state)
    return state


def _validate_semantics(state: TrainingCheckpointState) -> None:
    saved_epochs = state.resolved_config["epochs"]
    if (
        isinstance(saved_epochs, bool)
        or not isinstance(saved_epochs, int)
        or saved_epochs < 1
        or state.completed_epoch > saved_epochs
    ):
        raise QNEPError("malformed training checkpoint: completed_epoch")
    if len(state.metrics_history) != state.completed_epoch:
        raise QNEPError("malformed training checkpoint: metrics_history length")
    for expected_epoch, row in enumerate(state.metrics_history, start=1):
        epoch = row.get("epoch")
        if isinstance(epoch, bool) or epoch != expected_epoch:
            raise QNEPError("malformed training checkpoint: metrics_history epoch")
    if state.completed_epoch == 0:
        if state.best_epoch != 0 or state.patience_count != 0:
            raise QNEPError("malformed training checkpoint: initial best state")
        if state.optimizer_step != 0:
            raise QNEPError("malformed training checkpoint: optimizer_step")
        return
    if not 1 <= state.best_epoch <= state.completed_epoch:
        raise QNEPError("malformed training checkpoint: best_epoch")
    if state.patience_count != state.completed_epoch - state.best_epoch:
        raise QNEPError("malformed training checkpoint: patience_count")
    if state.optimizer_step < state.completed_epoch:
        raise QNEPError("malformed training checkpoint: optimizer_step")


def parse_adam_state(value) -> AdamOptimizerState:
    raw = _mapping(value, "optimizer_state")
    raw_states = _mapping(raw.get("state"), "optimizer_state.state")
    states: dict[int, dict[str, torch.Tensor]] = {}
    for key, item in raw_states.items():
        if isinstance(key, bool) or not isinstance(key, int):
            raise QNEPError("malformed training checkpoint: optimizer state key")
        states[key] = _tensor_map(item, "optimizer parameter state", allow_empty=True)
    raw_groups = raw.get("param_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise QNEPError("malformed training checkpoint: optimizer param_groups")
    groups: list[dict[str, AdamGroupValue]] = []
    for item in raw_groups:
        group = _mapping(item, "optimizer param_group")
        parsed: dict[str, AdamGroupValue] = {}
        for key, group_value in group.items():
            if not isinstance(key, str) or not _adam_group_value(group_value):
                raise QNEPError("malformed training checkpoint: optimizer param_group")
            parsed[key] = group_value
        parameters = parsed.get("params")
        if not isinstance(parameters, list) or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in parameters
        ):
            raise QNEPError("malformed training checkpoint: optimizer params")
        groups.append(parsed)
    return {"state": states, "param_groups": groups}


def _adam_group_value(value) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
    if isinstance(value, tuple):
        return len(value) == 2 and all(isinstance(item, float) for item in value)
    return False


def _need(payload, name: str):
    if not isinstance(payload, dict):
        raise QNEPError("malformed training checkpoint: payload must be a mapping")
    if name not in payload:
        raise QNEPError(f"malformed training checkpoint: missing {name}")
    return payload[name]


def _mapping(value, field: str):
    if not isinstance(value, dict):
        raise QNEPError(f"malformed training checkpoint: {field}")
    return value


def _string(payload, name: str) -> str:
    value = _need(payload, name)
    if not isinstance(value, str):
        raise QNEPError(f"malformed training checkpoint: {name}")
    return value


def _integer(payload, name: str) -> int:
    value = _need(payload, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QNEPError(f"malformed training checkpoint: {name}")
    return value


def _finite(payload, name: str) -> float:
    value = _need(payload, name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise QNEPError(f"malformed training checkpoint: {name}")
    return float(value)


def _fingerprint(payload, name: str) -> str:
    value = _string(payload, name)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise QNEPError(f"malformed training checkpoint: {name}")
    return value


def _tensor_map(
    value, field: str, allow_empty: bool = False
) -> dict[str, torch.Tensor]:
    result = _mapping(value, field)
    valid = all(
        isinstance(key, str) and isinstance(item, torch.Tensor)
        for key, item in result.items()
    )
    if (not result and not allow_empty) or not valid:
        raise QNEPError(f"malformed training checkpoint: {field}")
    if any(
        (item.is_floating_point() or item.is_complex())
        and not bool(torch.isfinite(item).all())
        for item in result.values()
    ):
        raise QNEPError(f"malformed training checkpoint: nonfinite {field}")
    return dict(result)


def _rng(payload, name: str) -> torch.Tensor:
    value = _need(payload, name)
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
    ):
        raise QNEPError(f"malformed training checkpoint: {name}")
    return value


def _device_rng(value) -> tuple[torch.Tensor, ...]:
    if not isinstance(value, (list, tuple)):
        raise QNEPError("malformed training checkpoint: device_rng_states")
    states = tuple(value)
    if any(
        not isinstance(item, torch.Tensor)
        or item.dtype != torch.uint8
        or item.ndim != 1
        for item in states
    ):
        raise QNEPError("malformed training checkpoint: device_rng_states")
    return states


def _metrics(value) -> tuple[dict[str, MetricScalar], ...]:
    if not isinstance(value, (list, tuple)):
        raise QNEPError("malformed training checkpoint: metrics_history")
    rows: list[dict[str, MetricScalar]] = []
    for item in value:
        row = _mapping(item, "metrics_history")
        if not all(
            isinstance(key, str) and _metric(row_value)
            for key, row_value in row.items()
        ):
            raise QNEPError("malformed training checkpoint: metrics_history")
        rows.append(dict(row))
    return tuple(rows)


def _metric(value) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _config(value) -> dict[str, MetricScalar]:
    result = _mapping(value, "resolved_config")
    if set(result) != set(CONFIG_FIELDS) or not all(
        _metric(item) for item in result.values()
    ):
        raise QNEPError("malformed training checkpoint: resolved_config")
    integer_fields = ("epochs", "batch_size", "seed", "checkpoint_every")
    if any(
        isinstance(result[name], bool) or not isinstance(result[name], int)
        for name in integer_fields
    ):
        raise QNEPError("malformed training checkpoint: resolved_config integer")
    if not isinstance(result["output_dir"], str):
        raise QNEPError("malformed training checkpoint: resolved_config output_dir")
    if result["resume_from"] is not None and not isinstance(result["resume_from"], str):
        raise QNEPError("malformed training checkpoint: resolved_config resume_from")
    return dict(result)


def _runtime(value) -> RuntimeMetadata:
    try:
        runtime = RuntimeMetadata(**_mapping(value, "runtime"))
    except TypeError as error:
        raise QNEPError("malformed training checkpoint: runtime") from error
    strings = (
        runtime.torch_version,
        runtime.python_version,
        runtime.platform,
        runtime.device,
        runtime.precision,
        runtime.default_dtype,
    )
    if not all(isinstance(item, str) for item in strings):
        raise QNEPError("malformed training checkpoint: runtime")
    if runtime.cuda_version is not None and not isinstance(runtime.cuda_version, str):
        raise QNEPError("malformed training checkpoint: runtime")
    if not isinstance(runtime.deterministic_algorithms, bool):
        raise QNEPError("malformed training checkpoint: runtime")
    return runtime
