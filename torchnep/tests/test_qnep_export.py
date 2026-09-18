from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ase import Atoms
from torchnep.qnep import (
    QNEPConfig,
    QNEPModel,
    QNEPStructure,
    load_checkpoint,
    load_gpumd_reference,
    save_checkpoint,
)
from torchnep.qnep.errors import QNEPError


@pytest.mark.parametrize("l_max", [(2, 2, 1), (0, 0, 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_roundtrip_preserves_predictions_and_weights(
    tmp_path: Path, l_max: tuple[int, int, int], dtype: torch.dtype
) -> None:
    # Given distinct weights, nontrivial scaling and both element networks.
    torch.manual_seed(43)
    model = QNEPModel(
        QNEPConfig(
            type_names=("O", "H"),
            cutoff_radial=3.5,
            cutoff_angular=3.0,
            n_max_radial=1,
            n_max_angular=1,
            basis_size_radial=2,
            basis_size_angular=2,
            neuron=3,
            l_max=l_max,
        )
    ).to(dtype=dtype)
    with torch.no_grad():
        model.nep.b1.fill_(0.314159265358979)
        model.nep.q_scaler.copy_(torch.linspace(0.7, 1.7, model.nep.dim))
        for network in model.nep.fitting_nets:
            network.b0.uniform_(-0.3, 0.3)
        for weights in model.charge_weights:
            weights.mul_(20)
    atoms = Atoms(
        "OH2",
        positions=[[1, 1, 1], [1.93, 1, 1], [0.78, 1.90, 1.13]],
        cell=[7, 8, 9],
        pbc=True,
    )
    expected = model(QNEPStructure.from_atoms(atoms, model))
    output = tmp_path / "nep.txt"
    before_rng = torch.get_rng_state()

    # When exporting and loading through the native text importer.
    model.export_nep(output)
    after_export_rng = torch.get_rng_state()
    restored = load_gpumd_reference(output).to(dtype=dtype)
    actual = restored(QNEPStructure.from_atoms(atoms, restored))

    # Then the complete physical model survives without RNG consumption.
    assert output.read_text().splitlines()[0] == "nep4_charge2 2 O H"
    assert restored.config == model.config
    torch.testing.assert_close(before_rng, after_export_rng, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


def test_import_checkpoint_export_preserves_native_bec_metadata(tmp_path: Path) -> None:
    # Given an independently authored native model with its BEC scale.
    source = (
        Path(__file__).resolve().parents[2]
        / "tests_pytest/fixtures/models/qnep_mode2_water.txt"
    )
    original = [float(line) for line in source.read_text().splitlines()[6:]]
    model = load_gpumd_reference(source)
    output = tmp_path / "nep.txt"

    # When the imported model is checkpointed and exported again.
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(model, checkpoint)
    load_checkpoint(checkpoint).export_nep(output)

    # Then every native numeric parameter, including BEC metadata, is retained.
    exported = [float(line) for line in output.read_text().splitlines()[6:]]
    assert exported == original


@pytest.mark.parametrize(
    "config",
    [
        QNEPConfig(("H",), reciprocal_cutoff_factor=1.1),
        QNEPConfig(("H",), cutoff_radial=3, cutoff_angular=4),
        QNEPConfig(("X",)),
        QNEPConfig(("H",), neuron=121),
        QNEPConfig(("H",), n_max_radial=13),
        QNEPConfig(("H",), n_max_angular=9),
        QNEPConfig(("H",), basis_size_radial=17),
        QNEPConfig(("H",), basis_size_angular=13),
        QNEPConfig(("H",), sqrt_epsilon_inf=float("nan")),
    ],
)
def test_unsupported_export_preserves_existing_file(
    tmp_path: Path, config: QNEPConfig
) -> None:
    # Given an existing artifact and a nonrepresentable model.
    destination = tmp_path / "nep.txt"
    destination.write_text("previous artifact\n")
    model = QNEPModel(config)

    # When native export rejects the model.
    with pytest.raises(QNEPError):
        model.export_nep(destination)

    # Then the existing artifact is untouched and no temporary file remains.
    assert destination.read_text() == "previous artifact\n"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 1e40])
def test_nonfinite_native_parameter_preserves_existing_file(
    tmp_path: Path, invalid: float
) -> None:
    # Given a scaler that cannot be represented by native float parameters.
    model = QNEPModel(QNEPConfig(("H",))).double()
    model.nep.q_scaler[0] = invalid
    destination = tmp_path / "nep.txt"
    destination.write_text("previous artifact\n")

    # When export validates native numeric values.
    with pytest.raises(QNEPError):
        model.export_nep(destination)

    # Then failed validation never truncates the destination.
    assert destination.read_text() == "previous artifact\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_failed_atomic_replacement_removes_temporary_file(tmp_path: Path) -> None:
    # Given a destination directory that cannot be replaced by a model file.
    destination = tmp_path / "nep.txt"
    destination.mkdir()
    model = QNEPModel(QNEPConfig(("H",)))

    # When the filesystem rejects the atomic replacement.
    with pytest.raises(IsADirectoryError):
        model.export_nep(destination)

    # Then the destination survives and staging files are cleaned up.
    assert destination.is_dir()
    assert list(tmp_path.iterdir()) == [destination]
