from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torchnep.qnep.batching import backward_weighted_batch
from torchnep.qnep.checkpoint import load_checkpoint
from torchnep.qnep.gpumd_import import load_gpumd_reference
from torchnep.qnep.metrics import evaluate_metrics, sample_to_model
from torchnep.qnep.model import QNEPConfig, QNEPModel
from torchnep.qnep.records import QNEPRecord
from torchnep.qnep.run_config import QNEPDatasets, QNEPRunConfig
from torchnep.qnep.structure import QNEPStructure
from torchnep.qnep.trainer import train_qnep
from torchnep.qnep.training import TrainingConfig, TrainingSample


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


def _structure(shift: float) -> QNEPStructure:
    return QNEPStructure(
        positions=torch.tensor(
            [
                [1.0 + shift, 1.0, 1.0],
                [1.8 + shift, 1.0, 1.0],
                [0.8 + shift, 1.7, 1.1],
            ],
            dtype=torch.float64,
        ),
        cell=torch.diag(torch.tensor([5.0, 5.5, 6.0], dtype=torch.float64)),
        atom_types=torch.tensor([1, 0, 0]),
    )


def _teacher_record(teacher: QNEPModel, shift: float, weight: float) -> QNEPRecord:
    structure = _structure(shift)
    prediction = teacher(structure)
    return QNEPRecord(
        TrainingSample(
            structure=structure,
            energy=prediction.total_energy.detach().item(),
            forces=prediction.forces.detach(),
            weight=weight,
        ),
        record_id=f"row-{shift}",
    )


def _datasets() -> QNEPDatasets:
    teacher = _model(101)
    return QNEPDatasets(
        train=(
            _teacher_record(teacher, 0.0, 1.0),
            _teacher_record(teacher, 0.2, 3.0),
        ),
        validation=(_teacher_record(teacher, 0.4, 2.0),),
        snapshot_id="tiny-oh2-v1",
    )


def _config(output_dir: Path, **changes: float | Path | None) -> QNEPRunConfig:
    config = QNEPRunConfig(
        output_dir=output_dir,
        epochs=2,
        batch_size=1,
        learning_rate=0.003,
        energy_weight=1.0,
        force_weight=0.2,
        charge_weight=0.0,
        seed=17,
    )
    return replace(config, **changes)


def test_train_qnep_runs_batches_validation_and_writes_canonical_artifacts(
    tmp_path: Path,
) -> None:
    # Given
    model = _model(7)
    datasets = _datasets()
    original_positions = datasets.train[0].sample.structure.positions.clone()

    # When
    result = train_qnep(model, datasets, _config(tmp_path / "run"))

    # Then
    assert result.completed_epoch == 2
    assert result.optimizer_step == 4
    assert result.stop_reason == "completed"
    assert result.best_inference_path.is_file()
    assert result.latest_inference_path.is_file()
    assert result.training_checkpoint_path.is_file()
    assert result.nep_path == tmp_path / "run" / "nep.txt"
    assert result.best_nep_path.name == "nep_best.txt"
    assert result.latest_nep_path.name == "nep_last.txt"
    assert result.nep_path.read_bytes() == result.best_nep_path.read_bytes()
    assert result.latest_nep_path.is_file()
    torch.testing.assert_close(
        datasets.train[0].sample.structure.positions, original_positions, rtol=0, atol=0
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [row["epoch"] for row in rows] == [1, 2]
    assert all(row["validation_record_count"] == 1 for row in rows)
    assert all(row["validation_force_component_count"] == 9 for row in rows)
    run = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert run["snapshot_id"] == "tiny-oh2-v1"
    assert len(run["initial_model_hash"]) == 64


def test_validation_metrics_use_global_record_weight_denominators() -> None:
    # Given
    model = _model(31)
    structures = (_structure(0.0), _structure(0.3))
    predictions = tuple(model(structure) for structure in structures)
    records = (
        QNEPRecord(
            TrainingSample(
                structures[0],
                energy=predictions[0].total_energy.detach().item() + 3.0,
                weight=1.0,
            )
        ),
        QNEPRecord(
            TrainingSample(
                structures[1],
                energy=predictions[1].total_energy.detach().item() + 6.0,
                weight=3.0,
            )
        ),
    )
    config = TrainingConfig(energy_weight=2.0, force_weight=0.0, charge_weight=0.0)

    # When
    metrics = evaluate_metrics(model, records, config)

    # Then
    assert metrics.energy_rmse == pytest.approx((3.25) ** 0.5)
    assert metrics.supervised_objective == pytest.approx(6.5)
    assert metrics.force_rmse is None
    assert metrics.energy_record_count == 2


def test_one_epoch_full_batch_matches_reference_adam(tmp_path: Path) -> None:
    # Given
    datasets = _datasets()
    trained = _model(53)
    reference = _model(53)
    config = _config(tmp_path / "run", epochs=1, batch_size=2)
    loss_config = TrainingConfig(
        learning_rate=config.learning_rate,
        energy_weight=config.energy_weight,
        force_weight=config.force_weight,
        charge_weight=config.charge_weight,
    )
    optimizer = torch.optim.Adam(reference.parameters(), lr=config.learning_rate)
    samples = tuple(sample_to_model(record.sample, reference) for record in datasets.train)

    # When
    train_qnep(trained, datasets, config)
    optimizer.zero_grad(set_to_none=True)
    backward_weighted_batch(
        reference,
        samples,
        loss_config,
        global_record_count=2,
        global_weight=4.0,
    )
    optimizer.step()

    # Then
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(trained.state_dict()[name], expected, rtol=0, atol=0)


def test_strict_best_ties_enable_early_stop_and_rejects_output_overwrite(
    tmp_path: Path,
) -> None:
    # Given
    datasets = _datasets()
    model = _model(101)
    run_dir = tmp_path / "early"
    config = _config(
        run_dir,
        epochs=2,
        batch_size=2,
        learning_rate=1e-12,
        early_stop_patience=1,
    )

    # When
    result = train_qnep(model, datasets, config)

    # Then
    assert result.stop_reason == "early_stopping"
    assert result.completed_epoch == 2
    assert result.best_epoch == 1
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    marker = occupied / "keep.txt"
    marker.write_text("owned by caller", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty output_dir"):
        _config(occupied)
    assert marker.read_text(encoding="utf-8") == "owned by caller"
    best = load_checkpoint(result.best_inference_path)
    latest = load_checkpoint(result.latest_inference_path)
    assert best.config == latest.config


def test_best_and_latest_are_distinct_after_validation_regresses(tmp_path: Path) -> None:
    # Given
    datasets = _datasets()
    model = _model(101)
    config = _config(
        tmp_path / "run",
        epochs=2,
        batch_size=2,
        learning_rate=0.001,
        charge_weight=1.0,
        early_stop_patience=1,
    )

    # When
    result = train_qnep(model, datasets, config)

    # Then
    best = load_checkpoint(result.best_inference_path)
    latest = load_checkpoint(result.latest_inference_path)
    assert result.best_epoch < result.completed_epoch
    assert any(
        not torch.equal(best.state_dict()[name], latest.state_dict()[name])
        for name in best.state_dict()
    )
    assert result.nep_path.read_bytes() == result.best_nep_path.read_bytes()
    assert result.nep_path.read_bytes() != result.latest_nep_path.read_bytes()
    for native_path, checkpoint in (
        (result.nep_path, best),
        (result.latest_nep_path, latest),
    ):
        exported = load_gpumd_reference(native_path)
        for name, expected in checkpoint.state_dict().items():
            torch.testing.assert_close(exported.state_dict()[name], expected, rtol=0, atol=0)
