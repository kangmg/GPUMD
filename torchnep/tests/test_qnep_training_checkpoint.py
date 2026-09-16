from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch


def _state(tmp_path: Path):
    from torchnep.qnep.model import QNEPConfig, QNEPModel
    from torchnep.qnep.run_config import QNEPRunConfig
    from torchnep.qnep.training_checkpoint import (
        RuntimeMetadata,
        TrainingCheckpointState,
        adam_optimizer_state,
        resolved_run_config,
    )

    torch.manual_seed(29)
    model = QNEPModel(QNEPConfig(type_names=("H",), neuron=3)).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    sum(
        (parameter.square().sum() for parameter in model.parameters()), torch.zeros(())
    ).backward()
    optimizer.step()
    shuffle = torch.Generator(device="cpu").manual_seed(71)
    config = QNEPRunConfig(
        output_dir=tmp_path / "run",
        epochs=5,
        batch_size=2,
        learning_rate=0.002,
        seed=29,
        checkpoint_every=2,
    )
    state = TrainingCheckpointState(
        producer_version="test-version",
        model_config=model.config,
        model_state=model.state_dict(),
        optimizer_state=adam_optimizer_state(optimizer),
        completed_epoch=2,
        optimizer_step=4,
        best_metric=0.125,
        best_epoch=1,
        best_model_state=model.state_dict(),
        patience_count=1,
        train_fingerprint="a" * 64,
        validation_fingerprint="b" * 64,
        settings_fingerprint="c" * 64,
        initial_model_hash="d" * 64,
        snapshot_id="snapshot-7",
        metrics_history=(
            {"epoch": 1, "validation_supervised_objective": 0.125},
            {"epoch": 2, "validation_supervised_objective": 0.25},
        ),
        resolved_config=resolved_run_config(config),
        cpu_rng_state=torch.get_rng_state(),
        device_rng_states=(),
        shuffle_rng_state=shuffle.get_state(),
        runtime=RuntimeMetadata.current(torch.device("cpu"), "float64"),
    )
    return state, model, optimizer, shuffle, config


def test_training_checkpoint_roundtrip_restores_state_and_rng(tmp_path: Path) -> None:
    from torchnep.qnep.model import QNEPModel
    from torchnep.qnep.training_checkpoint import (
        read_training_checkpoint,
        restore_training_state,
        save_training_checkpoint,
    )

    state, _, _, _, _ = _state(tmp_path)
    path = tmp_path / "last.training.pt"
    save_training_checkpoint(state, path)

    loaded = read_training_checkpoint(path)
    model = QNEPModel(loaded.model_config).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=9.0)
    shuffle = torch.Generator(device="cpu").manual_seed(999)
    restore_training_state(loaded, model, optimizer, shuffle)

    assert loaded.completed_epoch == 2
    assert loaded.metrics_history == state.metrics_history
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.002)
    for key, expected in state.model_state.items():
        torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
    expected_global = torch.rand(
        4, generator=torch.Generator().set_state(state.cpu_rng_state)
    )
    expected_shuffle = torch.rand(
        4, generator=torch.Generator().set_state(state.shuffle_rng_state)
    )
    torch.testing.assert_close(torch.rand(4), expected_global, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.rand(4, generator=shuffle), expected_shuffle, rtol=0, atol=0
    )
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert all(tensor.device.type == "cpu" for tensor in raw["model_state"].values())


def test_resume_validation_rejects_changed_data_settings_and_runtime(
    tmp_path: Path,
) -> None:
    from torchnep.qnep.records import QNEPDatasetIdentities
    from torchnep.qnep.training_checkpoint import validate_resume

    state, _, _, _, config = _state(tmp_path)
    identities = QNEPDatasetIdentities("a" * 64, "b" * 64, "c" * 64)
    resume_path = tmp_path / "last.training.pt"
    resume_path.touch()
    allowed = replace(
        config,
        output_dir=tmp_path / "continued",
        epochs=8,
        checkpoint_every=1,
        resume_from=resume_path,
    )
    validate_resume(state, allowed, identities, "snapshot-7")

    with pytest.raises(ValueError, match="batch_size"):
        validate_resume(state, replace(allowed, batch_size=3), identities, "snapshot-7")
    with pytest.raises(ValueError, match="training data"):
        validate_resume(
            state,
            allowed,
            replace(identities, train="d" * 64),
            "snapshot-7",
        )
    with pytest.raises(ValueError, match="loss/sampling settings"):
        validate_resume(
            state,
            allowed,
            replace(identities, settings="e" * 64),
            "snapshot-7",
        )
    with pytest.raises(ValueError, match="torch version"):
        validate_resume(
            replace(state, runtime=replace(state.runtime, torch_version="0.0")),
            allowed,
            identities,
            "snapshot-7",
        )
    with pytest.raises(ValueError, match="resolved device"):
        validate_resume(
            replace(state, runtime=replace(state.runtime, device="cuda:0")),
            allowed,
            identities,
            "snapshot-7",
        )
    with pytest.raises(ValueError, match="precision"):
        validate_resume(
            replace(state, runtime=replace(state.runtime, precision="float32")),
            allowed,
            identities,
            "snapshot-7",
        )


def test_reader_rejects_inference_format_and_malformed_payload(tmp_path: Path) -> None:
    from torchnep.qnep.training_checkpoint import read_training_checkpoint

    inference = tmp_path / "inference.pt"
    torch.save({"format": "torchnep-qnep-mode2-reference-v1"}, inference)
    with pytest.raises(ValueError, match="inference checkpoint.*cannot.*resume"):
        read_training_checkpoint(inference)

    malformed = tmp_path / "malformed.pt"
    torch.save({"format": "torchnep-qnep-mode2-training-v1"}, malformed)
    with pytest.raises(ValueError, match="missing.*producer_version"):
        read_training_checkpoint(malformed)

    truncated = tmp_path / "truncated.pt"
    truncated.write_bytes(b"PK\x03\x04truncated")
    with pytest.raises(ValueError, match="invalid qNEP training checkpoint"):
        read_training_checkpoint(truncated)


def test_reader_rejects_impossible_counter_and_history_relationships(
    tmp_path: Path,
) -> None:
    from torchnep.qnep.training_checkpoint import (
        read_training_checkpoint,
        save_training_checkpoint,
    )

    state, _, _, _, _ = _state(tmp_path)
    valid_path = tmp_path / "valid.training.pt"
    save_training_checkpoint(state, valid_path)
    valid = torch.load(valid_path, map_location="cpu", weights_only=True)
    malformed = (
        ("completed_epoch", 999, "completed_epoch"),
        ("metrics_history", valid["metrics_history"][:1], "metrics_history"),
        (
            "metrics_history",
            [{**valid["metrics_history"][0], "epoch": 2}, valid["metrics_history"][1]],
            "metrics_history",
        ),
        ("best_epoch", 3, "best_epoch"),
        ("patience_count", 0, "patience_count"),
        ("optimizer_step", 1, "optimizer_step"),
    )
    for index, (field, value, message) in enumerate(malformed):
        payload = deepcopy(valid)
        payload[field] = value
        path = tmp_path / f"impossible-{index}.training.pt"
        torch.save(payload, path)
        with pytest.raises(ValueError, match=message):
            read_training_checkpoint(path)


@pytest.mark.parametrize(
    ("tensor_map", "nonfinite"),
    (
        ("model_state", float("nan")),
        ("model_state", float("inf")),
        ("best_model_state", float("nan")),
        ("best_model_state", float("inf")),
        ("optimizer_state", float("nan")),
        ("optimizer_state", float("inf")),
    ),
)
def test_reader_rejects_nonfinite_checkpoint_tensor_before_mutating_caller_state(
    tmp_path: Path, tensor_map: str, nonfinite: float
) -> None:
    from torchnep.qnep.model import QNEPModel
    from torchnep.qnep.training_checkpoint import (
        read_training_checkpoint,
        restore_training_state,
        save_training_checkpoint,
    )

    state, _, _, _, _ = _state(tmp_path)
    valid_path = tmp_path / "valid.training.pt"
    save_training_checkpoint(state, valid_path)
    payload = torch.load(valid_path, map_location="cpu", weights_only=True)
    if tensor_map == "optimizer_state":
        tensor = next(
            parameter_state["exp_avg"]
            for parameter_state in payload["optimizer_state"]["state"].values()
        )
    else:
        tensor = next(iter(payload[tensor_map].values()))
    tensor = tensor.clone()
    tensor.reshape(-1)[0] = nonfinite
    if tensor_map == "optimizer_state":
        next(
            parameter_state
            for parameter_state in payload["optimizer_state"]["state"].values()
            if "exp_avg" in parameter_state
        )["exp_avg"] = tensor
    else:
        next_key = next(iter(payload[tensor_map]))
        payload[tensor_map][next_key] = tensor
    malformed_path = tmp_path / f"{tensor_map}-{nonfinite}.training.pt"
    torch.save(payload, malformed_path)

    model = QNEPModel(state.model_config).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    shuffle = torch.Generator().manual_seed(1)
    before_model = {name: value.clone() for name, value in model.state_dict().items()}
    before_cpu_rng = torch.get_rng_state().clone()
    before_shuffle_rng = shuffle.get_state().clone()

    with pytest.raises(ValueError, match="nonfinite"):
        restore_training_state(
            read_training_checkpoint(malformed_path), model, optimizer, shuffle
        )

    for name, expected in before_model.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), before_cpu_rng, rtol=0, atol=0)
    torch.testing.assert_close(shuffle.get_state(), before_shuffle_rng, rtol=0, atol=0)


def test_restore_rejects_malformed_optimizer_before_model_mutation(
    tmp_path: Path,
) -> None:
    from torchnep.qnep.model import QNEPModel
    from torchnep.qnep.training_checkpoint import restore_training_state
    from torchnep.qnep.training_checkpoint_schema import AdamOptimizerState

    state, _, _, _, _ = _state(tmp_path)
    malformed_optimizer: AdamOptimizerState = {
        "state": state.optimizer_state["state"],
        "param_groups": [dict(state.optimizer_state["param_groups"][0])],
    }
    malformed_optimizer["param_groups"][0]["params"] = []
    malformed_state = replace(state, optimizer_state=malformed_optimizer)
    model = QNEPModel(state.model_config).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    with pytest.raises(ValueError, match="optimizer parameters mismatch"):
        restore_training_state(
            malformed_state,
            model,
            optimizer,
            torch.Generator().manual_seed(1),
        )

    for name, expected in before.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)


def test_atomic_failure_preserves_preceding_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torchnep.qnep import training_checkpoint

    state, _, _, _, _ = _state(tmp_path)
    path = tmp_path / "last.training.pt"
    training_checkpoint.save_training_checkpoint(state, path)
    preceding = path.read_bytes()

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        training_checkpoint.save_training_checkpoint(
            replace(
                state,
                completed_epoch=3,
                optimizer_step=6,
                patience_count=2,
                metrics_history=state.metrics_history
                + ({"epoch": 3, "validation_supervised_objective": 0.5},),
            ),
            path,
        )

    assert path.read_bytes() == preceding
    assert list(tmp_path.glob(".last.training.pt.*.tmp")) == []


def test_legacy_literal_reference_v1_payload_remains_loadable(tmp_path: Path) -> None:
    from torchnep.qnep.checkpoint import load_checkpoint
    from torchnep.qnep.model import QNEPConfig, QNEPModel

    torch.manual_seed(43)
    model = QNEPModel(QNEPConfig(type_names=("H",), neuron=3)).float()
    path = tmp_path / "literal-reference-v1.pt"
    torch.save(
        {
            "format": "torchnep-qnep-mode2-reference-v1",
            "producer_version": "baseline-test",
            "config": model.config._asdict(),
            "state_dict": model.state_dict(),
        },
        path,
    )

    loaded = load_checkpoint(path)
    assert loaded.nep.b1.dtype == torch.float32
    for key, expected in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], expected, rtol=0, atol=0)
