from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from torchnep.qnep.checkpoint import load_checkpoint
from torchnep.qnep.errors import QNEPError
from torchnep.qnep.model import QNEPConfig, QNEPModel
from torchnep.qnep.records import QNEPRecord, record_fingerprint
from torchnep.qnep.run_config import QNEPDatasets, QNEPRunConfig
from torchnep.qnep.structure import QNEPStructure
from torchnep.qnep.trainer import train_qnep
from torchnep.qnep.training import TrainingSample
from torchnep.qnep.training_checkpoint import read_training_checkpoint


def _model(seed: int) -> QNEPModel:
    torch.manual_seed(seed)
    return QNEPModel(
        QNEPConfig(
            type_names=("H", "O"),
            cutoff_radial=2.5,
            cutoff_angular=2.5,
            n_max_radial=0,
            n_max_angular=0,
            basis_size_radial=1,
            basis_size_angular=1,
            l_max=(1, 0, 0),
            neuron=3,
        )
    ).double()


def _structure(
    displacement: float, dtype: torch.dtype = torch.float64
) -> QNEPStructure:
    return QNEPStructure(
        positions=torch.tensor(
            [[1.0, 1.0, 1.0], [1.8, 1.0 + displacement, 1.0], [0.8, 1.7, 1.1]],
            dtype=dtype,
        ),
        cell=torch.diag(torch.tensor([5.0, 5.5, 6.0], dtype=dtype)),
        atom_types=torch.tensor([1, 0, 0]),
    )


def _record(
    teacher: QNEPModel,
    displacement: float,
    weight: float,
    masked: bool,
    record_id: str,
) -> QNEPRecord:
    structure = _structure(displacement, teacher.nep.b1.dtype)
    prediction = teacher(structure)
    mask = None
    if masked:
        mask = torch.tensor(
            [[True, False, True], [True, True, False], [False, True, True]]
        )
    return QNEPRecord(
        TrainingSample(
            structure=structure,
            energy=prediction.total_energy.detach().item(),
            forces=prediction.forces.detach(),
            force_mask=mask,
            weight=weight,
        ),
        record_id=record_id,
    )


def _datasets() -> QNEPDatasets:
    teacher = _model(101)
    return QNEPDatasets(
        train=(
            _record(teacher, 0.0, 1.0, False, "train-0"),
            _record(teacher, 0.1, 3.0, True, "train-1"),
            _record(teacher, 0.2, 7.0, False, "train-2"),
        ),
        validation=(_record(teacher, 0.3, 2.0, True, "validation-0"),),
        snapshot_id="resume-fixture-v1",
    )


def _config(output_dir: Path, **changes: float | Path | None) -> QNEPRunConfig:
    values: dict[str, float | Path | None] = {
        "output_dir": output_dir,
        "epochs": 4,
        "batch_size": 2,
        "learning_rate": 0.003,
        "energy_weight": 1.0,
        "force_weight": 0.2,
        "charge_weight": 0.0,
        "seed": 17,
        "checkpoint_every": 1,
    }
    values.update(changes)
    return QNEPRunConfig(**values)


def _assert_tree_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_tree_equal(actual[key], value)
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for value, expected_value in zip(actual, expected):
            _assert_tree_equal(value, expected_value)
        return
    assert actual == expected


def _metrics_epochs(path: Path) -> list[int]:
    return [
        json.loads(line)["epoch"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_cpu_float64_resume_matches_continuous_state_and_recovers_metrics(
    tmp_path: Path,
) -> None:
    datasets = _datasets()
    continuous = _model(7)
    continuous_result = train_qnep(continuous, datasets, _config(tmp_path / "continuous"))
    continuous_state = read_training_checkpoint(continuous_result.training_checkpoint_path)

    interrupted = _model(7)
    interrupted_result = train_qnep(
        interrupted, datasets, _config(tmp_path / "interrupted", epochs=2)
    )
    interrupted_metrics = interrupted_result.training_checkpoint_path.parent / "metrics.jsonl"
    with interrupted_metrics.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write('{"epoch":999,"interrupted":true}\n')

    snapshot_result = train_qnep(
        _model(43),
        datasets,
        _config(
            tmp_path / "fresh-snapshot",
            epochs=2,
            resume_from=interrupted_result.training_checkpoint_path,
        ),
    )
    assert snapshot_result.training_checkpoint_path.is_file()
    assert snapshot_result.completed_epoch == 2

    resumed = _model(31)
    resumed_result = train_qnep(
        resumed,
        datasets,
        _config(
            tmp_path / "fresh-resume",
            resume_from=interrupted_result.training_checkpoint_path,
        ),
    )
    resumed_state = read_training_checkpoint(resumed_result.training_checkpoint_path)

    in_place = _model(53)
    in_place_result = train_qnep(
        in_place,
        datasets,
        _config(
            interrupted_result.training_checkpoint_path.parent,
            resume_from=interrupted_result.training_checkpoint_path,
        ),
    )

    assert resumed_result.training_checkpoint_path.parent == tmp_path / "fresh-resume"
    assert resumed_result.completed_epoch == continuous_result.completed_epoch == 4
    assert resumed_result.optimizer_step == continuous_result.optimizer_step == 8
    assert resumed_result.best_epoch == continuous_result.best_epoch
    for name, expected in continuous.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)
    _assert_tree_equal(resumed_state.optimizer_state, continuous_state.optimizer_state)
    _assert_tree_equal(resumed_state.model_state, continuous_state.model_state)
    _assert_tree_equal(resumed_state.best_model_state, continuous_state.best_model_state)
    assert resumed_state.metrics_history == continuous_state.metrics_history
    torch.testing.assert_close(resumed_state.cpu_rng_state, continuous_state.cpu_rng_state)
    torch.testing.assert_close(resumed_state.shuffle_rng_state, continuous_state.shuffle_rng_state)
    resumed_generator = torch.Generator().set_state(resumed_state.shuffle_rng_state)
    continuous_generator = torch.Generator().set_state(continuous_state.shuffle_rng_state)
    torch.testing.assert_close(
        torch.rand(5, generator=resumed_generator),
        torch.rand(5, generator=continuous_generator),
        rtol=0,
        atol=0,
    )
    assert _metrics_epochs(in_place_result.training_checkpoint_path.parent / "metrics.jsonl") == [1, 2, 3, 4]
    continuous_best = load_checkpoint(continuous_result.best_inference_path)
    resumed_best = load_checkpoint(resumed_result.best_inference_path)
    structure = datasets.validation[0].sample.structure
    torch.testing.assert_close(
        resumed_best(structure).total_energy, continuous_best(structure).total_energy, rtol=0, atol=0
    )
    torch.testing.assert_close(
        resumed_best(structure).forces, continuous_best(structure).forces, rtol=0, atol=0
    )


def test_changed_training_label_and_weight_reject_resume_without_overwrite(
    tmp_path: Path,
) -> None:
    datasets = _datasets()
    initial = _model(7)
    original = train_qnep(initial, datasets, _config(tmp_path / "original", epochs=2))
    checkpoint_bytes = original.training_checkpoint_path.read_bytes()
    metrics_bytes = (tmp_path / "original" / "metrics.jsonl").read_bytes()
    changed_sample = datasets.train[0].sample._replace(
        energy=datasets.train[0].sample.energy + 1.0,
        weight=5.0,
    )
    changed_datasets = QNEPDatasets(
        train=(QNEPRecord(changed_sample, record_id="train-0"),) + datasets.train[1:],
        validation=datasets.validation,
        snapshot_id=datasets.snapshot_id,
    )
    resume_dir = tmp_path / "rejected"

    with pytest.raises(QNEPError, match="training data fingerprint"):
        train_qnep(
            _model(31),
            changed_datasets,
            _config(resume_dir, resume_from=original.training_checkpoint_path),
        )

    assert original.training_checkpoint_path.read_bytes() == checkpoint_bytes
    assert (tmp_path / "original" / "metrics.jsonl").read_bytes() == metrics_bytes
    assert not resume_dir.exists()


def test_default_float64_run_resumes_float32_source_records_without_mutation(
    tmp_path: Path,
) -> None:
    # Given
    teacher = _model(101).float()
    datasets = QNEPDatasets(
        train=(
            _record(teacher, 0.0, 1.0, False, "train-0"),
            _record(teacher, 0.1, 3.0, True, "train-1"),
        ),
        validation=(_record(teacher, 0.2, 2.0, False, "validation-0"),),
        snapshot_id="float32-source-records-v1",
    )
    model = _model(7).float()
    source_tensors = tuple(
        (
            record.sample.structure.positions.clone(),
            record.sample.structure.cell.clone(),
            None if record.sample.forces is None else record.sample.forces.clone(),
        )
        for record in datasets.train + datasets.validation
    )
    fingerprints = tuple(
        record_fingerprint(record, model.config.type_names)
        for record in datasets.train + datasets.validation
    )

    # When
    initial = train_qnep(model, datasets, _config(tmp_path / "initial", epochs=1))
    resumed = train_qnep(
        model,
        datasets,
        _config(
            tmp_path / "resumed",
            epochs=2,
            resume_from=initial.training_checkpoint_path,
        ),
    )

    # Then
    assert model.nep.b1.dtype == torch.float64
    assert resumed.completed_epoch == 2
    assert resumed.optimizer_step == 2
    for record, tensors, fingerprint in zip(
        datasets.train + datasets.validation, source_tensors, fingerprints
    ):
        positions, cell, forces = tensors
        assert record.sample.structure.positions.dtype == torch.float32
        assert record.sample.structure.cell.dtype == torch.float32
        torch.testing.assert_close(record.sample.structure.positions, positions, rtol=0, atol=0)
        torch.testing.assert_close(record.sample.structure.cell, cell, rtol=0, atol=0)
        if forces is not None:
            assert record.sample.forces is not None
            torch.testing.assert_close(record.sample.forces, forces, rtol=0, atol=0)
        assert record_fingerprint(record, model.config.type_names) == fingerprint


def test_resume_exhausted_patience_materializes_state_without_optimizer_steps(
    tmp_path: Path,
) -> None:
    # Given
    datasets = _datasets()
    stopped = train_qnep(
        _model(101),
        datasets,
        _config(
            tmp_path / "stopped",
            epochs=2,
            batch_size=2,
            learning_rate=1e-12,
            early_stop_patience=1,
        ),
    )
    stopped_state = read_training_checkpoint(stopped.training_checkpoint_path)

    # When
    same_target = train_qnep(
        _model(31),
        datasets,
        _config(
            tmp_path / "same-target",
            epochs=2,
            batch_size=2,
            learning_rate=1e-12,
            early_stop_patience=1,
            resume_from=stopped.training_checkpoint_path,
        ),
    )
    larger_target = train_qnep(
        _model(43),
        datasets,
        _config(
            tmp_path / "larger-target",
            epochs=3,
            batch_size=2,
            learning_rate=1e-12,
            early_stop_patience=1,
            resume_from=stopped.training_checkpoint_path,
        ),
    )

    # Then
    assert stopped.stop_reason == "early_stopping"
    for result in (same_target, larger_target):
        state = read_training_checkpoint(result.training_checkpoint_path)
        assert result.completed_epoch == stopped.completed_epoch == 2
        assert result.optimizer_step == stopped.optimizer_step == 4
        assert result.best_epoch == stopped.best_epoch == 1
        assert result.stop_reason == "early_stopping"
        assert result.best_inference_path.is_file()
        assert result.latest_inference_path.is_file()
        assert result.training_checkpoint_path.is_file()
        assert _metrics_epochs(result.training_checkpoint_path.parent / "metrics.jsonl") == [1, 2]
        _assert_tree_equal(state.model_state, stopped_state.model_state)
        _assert_tree_equal(state.optimizer_state, stopped_state.optimizer_state)
        _assert_tree_equal(state.best_model_state, stopped_state.best_model_state)
        assert state.metrics_history == stopped_state.metrics_history
        torch.testing.assert_close(state.cpu_rng_state, stopped_state.cpu_rng_state)
        torch.testing.assert_close(state.shuffle_rng_state, stopped_state.shuffle_rng_state)
