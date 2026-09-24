from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from torchnep.qnep import QNEPConfig, QNEPModel, QNEPStructure


def _run_cli(directory: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    return subprocess.run(
        [sys.executable, "-m", "torchnep.qnep", *arguments],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _tiny_model(seed: int) -> QNEPModel:
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


def _write_labeled_splits(directory: Path) -> tuple[Path, Path]:
    from test_qnep_reference import system

    teacher = _tiny_model(19)
    _, prototype, _ = system()
    frames = []
    for index, (shift, labels) in enumerate(
        ((0.0, (True, False)), (0.13, (False, True)), (0.27, (True, True)))
    ):
        atoms = prototype.copy()
        atoms.positions[0, 0] += shift
        prediction = teacher(QNEPStructure.from_atoms(atoms, teacher))
        results: dict[str, float | object] = {}
        if labels[0]:
            results["energy"] = prediction.total_energy.detach().item() + 0.02
        if labels[1]:
            results["forces"] = prediction.forces.detach().numpy()
        atoms.calc = SinglePointCalculator(atoms, **results)
        atoms.info["sample_weight"] = index + 1
        frames.append(atoms)
    train_path = directory / "train.xyz"
    validation_path = directory / "validation.xyz"
    write(train_path, frames[:2], format="extxyz")
    write(validation_path, frames[2:], format="extxyz")
    return train_path, validation_path


def _new_run_arguments(train_path: Path, validation_path: Path, output_dir: Path) -> list[str]:
    return [
        str(train_path),
        "--validation",
        str(validation_path),
        "--output-dir",
        str(output_dir),
        "--elements",
        "H",
        "O",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--seed",
        "19",
        "--cutoff-radial",
        "2.5",
        "--cutoff-angular",
        "2.5",
        "--n-max-radial",
        "0",
        "--n-max-angular",
        "0",
        "--basis-size-radial",
        "1",
        "--basis-size-angular",
        "1",
        "--l-max",
        "1",
        "0",
        "0",
        "--neuron",
        "3",
        "--charge-weight",
        "0",
    ]


def test_training_cli_trains_mixed_labels_then_resumes(tmp_path: Path) -> None:
    # Given
    train_path, validation_path = _write_labeled_splits(tmp_path)
    output_dir = tmp_path / "run"

    # When
    fresh = _run_cli(
        tmp_path, *_new_run_arguments(train_path, validation_path, output_dir)
    )
    resumed = _run_cli(
        tmp_path,
        str(train_path),
        "--validation",
        str(validation_path),
        "--output-dir",
        str(output_dir),
        "--resume",
        str(output_dir / "last.training.pt"),
        "--epochs",
        "2",
    )

    # Then
    assert fresh.returncode == 0, fresh.stderr
    assert "completed_epoch=1" in fresh.stdout
    assert resumed.returncode == 0, resumed.stderr
    assert "completed_epoch=2" in resumed.stdout
    assert (output_dir / "best.qnep.pt").is_file()
    assert (output_dir / "latest.qnep.pt").is_file()
    assert (output_dir / "last.training.pt").is_file()
    assert f"nep={output_dir / 'nep.txt'}" in resumed.stdout
    assert (output_dir / "nep.txt").read_bytes() == (output_dir / "nep_best.txt").read_bytes()
    assert (output_dir / "nep_last.txt").is_file()


def test_training_cli_rejects_mixed_legacy_and_training_routes(tmp_path: Path) -> None:
    # Given
    arguments = (
        "train.xyz",
        "--validation",
        "validation.xyz",
        "--output-dir",
        "run",
        "--elements",
        "H",
        "--checkpoint",
        "legacy.qnep.pt",
    )

    # When
    mixed = _run_cli(tmp_path, *arguments)
    missing_output = _run_cli(
        tmp_path,
        "train.xyz",
        "--validation",
        "validation.xyz",
        "--resume",
        "run/last.training.pt",
    )

    # Then
    assert mixed.returncode == 2
    assert "cannot be combined" in mixed.stderr
    assert missing_output.returncode == 2
    assert "requires --output-dir" in missing_output.stderr


def test_training_cli_rejects_conflicting_explicit_resume_architecture(
    tmp_path: Path,
) -> None:
    # Given
    train_path, validation_path = _write_labeled_splits(tmp_path)
    output_dir = tmp_path / "run"
    fresh = _run_cli(
        tmp_path, *_new_run_arguments(train_path, validation_path, output_dir)
    )
    assert fresh.returncode == 0, fresh.stderr

    # When
    result = _run_cli(
        tmp_path,
        str(train_path),
        "--validation",
        str(validation_path),
        "--output-dir",
        str(output_dir),
        "--resume",
        str(output_dir / "last.training.pt"),
        "--neuron",
        "7",
    )

    # Then
    assert result.returncode == 1
    assert "architecture" in result.stderr


def test_training_cli_reports_runtime_qnep_errors_without_success_output(
    tmp_path: Path,
) -> None:
    # Given
    arguments = (
        "missing.xyz",
        "--validation",
        "validation.xyz",
        "--output-dir",
        "run",
        "--elements",
        "H",
    )

    # When
    result = _run_cli(tmp_path, *arguments)

    # Then
    assert result.returncode == 1
    assert "dataset file does not exist" in result.stderr
    assert result.stdout == ""
