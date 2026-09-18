from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_qnep_resume import _assert_tree_equal
from test_qnep_trainer import _config, _datasets, _model
from test_qnep_training_cli import _run_cli, _write_labeled_splits
from torchnep.qnep.checkpoint import load_checkpoint
from torchnep.qnep.model import QNEPModel
from torchnep.qnep.trainer import train_qnep
from torchnep.qnep.training_checkpoint import read_training_checkpoint


def test_native_publication_preserves_exact_two_epoch_resume(tmp_path: Path) -> None:
    datasets = _datasets()
    continuous = train_qnep(_model(7), datasets, _config(tmp_path / "continuous"))
    config = _config(tmp_path / "first")
    first = train_qnep(_model(7), datasets, replace(config, epochs=1))

    resumed = train_qnep(
        _model(31),
        datasets,
        replace(config, resume_from=first.training_checkpoint_path),
    )

    expected = read_training_checkpoint(continuous.training_checkpoint_path)
    actual = read_training_checkpoint(resumed.training_checkpoint_path)
    for field in (
        "model_state",
        "optimizer_state",
        "best_model_state",
        "cpu_rng_state",
        "shuffle_rng_state",
    ):
        _assert_tree_equal(getattr(actual, field), getattr(expected, field))
    assert actual.metrics_history == expected.metrics_history
    assert resumed.nep_path.read_bytes() == continuous.nep_path.read_bytes()
    assert (
        resumed.latest_nep_path.read_bytes() == continuous.latest_nep_path.read_bytes()
    )


def test_legacy_checkpoint_resume_publishes_native_without_an_optimizer_step(
    tmp_path: Path,
) -> None:
    datasets = _datasets()
    initial = train_qnep(_model(7), datasets, _config(tmp_path / "initial", epochs=1))
    payload = torch.load(initial.training_checkpoint_path, weights_only=True)
    payload["producer_version"] = "1.0.5a1+qnep2"
    payload["model_config"].pop("sqrt_epsilon_inf", None)
    legacy_path = tmp_path / "legacy.training.pt"
    torch.save(payload, legacy_path)

    resumed = train_qnep(
        _model(31),
        datasets,
        _config(tmp_path / "resumed", epochs=1, resume_from=legacy_path),
    )

    assert resumed.completed_epoch == initial.completed_epoch == 1
    assert resumed.optimizer_step == initial.optimizer_step
    assert resumed.nep_path.read_bytes() == initial.nep_path.read_bytes()
    assert resumed.best_nep_path.read_bytes() == initial.best_nep_path.read_bytes()
    assert resumed.latest_nep_path.read_bytes() == initial.latest_nep_path.read_bytes()
    old_state = read_training_checkpoint(legacy_path)
    new_state = read_training_checkpoint(resumed.training_checkpoint_path)
    torch.testing.assert_close(
        new_state.cpu_rng_state, old_state.cpu_rng_state, rtol=0, atol=0
    )
    torch.testing.assert_close(
        new_state.shuffle_rng_state, old_state.shuffle_rng_state, rtol=0, atol=0
    )
    for name, expected in old_state.model_state.items():
        torch.testing.assert_close(
            new_state.model_state[name], expected, rtol=0, atol=0
        )


def test_trainer_rejects_unexportable_config_before_changing_state_or_output(
    tmp_path: Path,
) -> None:
    model = QNEPModel(_model(7).config._replace(reciprocal_cutoff_factor=0.5)).double()
    datasets = _datasets()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    output = tmp_path / "rejected"

    with pytest.raises(ValueError, match="reciprocal_cutoff_factor"):
        train_qnep(model, datasets, _config(output, epochs=1))

    assert not output.exists()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for name, expected in state.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)


def test_reference_cli_exports_explicit_native_path_and_preserves_checkpoint(
    tmp_path: Path,
) -> None:
    train, _ = _write_labeled_splits(tmp_path)
    result = _run_cli(
        tmp_path,
        str(train),
        "--elements",
        "H",
        "O",
        "--steps",
        "1",
        "--checkpoint",
        "internal.pt",
        "--output",
        "final.txt",
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
    )

    assert result.returncode == 0, result.stderr
    assert "nep=final.txt" in result.stdout
    assert (tmp_path / "final.txt").read_text().startswith("nep4_charge2 2 H O\n")
    assert not (tmp_path / "nep.txt").exists()
    checkpoint = load_checkpoint(tmp_path / "internal.pt")
    assert checkpoint.config.neuron == 3
    expected = tmp_path / "expected.txt"
    checkpoint.export_nep(expected)
    assert expected.read_bytes() == (tmp_path / "final.txt").read_bytes()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--reciprocal-cutoff-factor", "0.5"), "reciprocal_cutoff_factor"),
        (("--output", "internal.pt"), "different files"),
    ],
)
def test_reference_cli_rejects_native_export_errors_before_reading_dataset(
    tmp_path: Path,
    extra: tuple[str, str],
    message: str,
) -> None:
    result = _run_cli(
        tmp_path,
        "absent.xyz",
        "--elements",
        "H",
        "O",
        "--steps",
        "1",
        "--checkpoint",
        "internal.pt",
        *extra,
    )

    assert result.returncode == 1
    assert message in result.stderr
    assert result.stdout == ""
    assert not (tmp_path / "internal.pt").exists()
    assert not (tmp_path / "nep.txt").exists()
