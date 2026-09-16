import numpy as np
import pytest
import torch
from ase import Atoms


def system():
    from torchnep.qnep import QNEPConfig, QNEPModel, QNEPStructure

    torch.manual_seed(29)
    model = QNEPModel(
        QNEPConfig(
            type_names=("H", "O"), cutoff_radial=3.5, cutoff_angular=3.0, neuron=6
        )
    ).double()
    atoms = Atoms(
        "OH2",
        positions=[[1.0, 1.0, 1.0], [1.93, 1.0, 1.0], [0.78, 1.90, 1.13]],
        cell=[7, 8, 9],
        pbc=True,
    )
    return model, atoms, QNEPStructure.from_atoms(atoms, model)


def test_total_force_matches_energy_difference():
    from torchnep.qnep import QNEPStructure

    model, atoms, structure = system()
    predicted = model(structure)
    numeric = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[i, j] += 1e-5
            minus.positions[i, j] -= 1e-5
            ep = model(QNEPStructure.from_atoms(plus, model)).total_energy.item()
            em = model(QNEPStructure.from_atoms(minus, model)).total_energy.item()
            numeric[i, j] = -(ep - em) / 2e-5
    np.testing.assert_allclose(
        predicted.forces.detach().numpy(), numeric, atol=2e-8, rtol=2e-5
    )


def test_force_loss_trains_charge_head_and_shared_descriptor():
    model, _, structure = system()
    result = model(structure, create_graph=True)
    loss = result.forces.square().mean()
    loss.backward()
    assert sum(p.grad.abs().sum().item() for p in model.charge_weights) > 1e-12
    assert model.nep.c_param_2.grad.abs().sum().item() > 1e-12
    assert abs(result.charges.sum().item()) < 1e-14


def test_rotation_translation_and_periodic_wrapping():
    from torchnep.qnep import QNEPStructure

    model, atoms, structure = system()
    reference = model(structure)
    rotation, _ = np.linalg.qr(
        np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 3.0, 1.0]])
    )
    atoms.positions = atoms.positions @ rotation + [2.0, -3.0, 5.0]
    atoms.set_cell(atoms.cell.array @ rotation)
    atoms.positions[1] += atoms.cell[0] - atoms.cell[2]
    transformed = model(QNEPStructure.from_atoms(atoms, model))
    assert transformed.total_energy.item() == pytest.approx(
        reference.total_energy.item(), abs=1e-10
    )
    np.testing.assert_allclose(
        transformed.forces.detach().numpy(),
        reference.forces.detach().numpy() @ rotation,
        atol=1e-10,
    )


def test_checkpoint_and_ase_roundtrip(tmp_path):
    from torchnep.qnep import QNEPCalculator, load_checkpoint, save_checkpoint

    model, atoms, structure = system()
    reference = model(structure)
    path = tmp_path / "qnep.pt"
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    atoms.calc = QNEPCalculator(loaded)
    assert atoms.get_potential_energy() == pytest.approx(
        reference.total_energy.item(), abs=1e-12
    )
    np.testing.assert_allclose(
        atoms.get_forces(), reference.forces.detach().numpy(), atol=1e-12
    )


@pytest.mark.parametrize(
    "pbc,charge", [(False, 0.0), ([True, True, False], 0.0), (True, 1.0)]
)
def test_unsupported_boundary_conditions_rejected(pbc, charge):
    from torchnep.qnep import QNEPStructure

    model, atoms, _ = system()
    atoms.pbc = pbc
    atoms.info["charge"] = charge
    with pytest.raises(ValueError):
        model(QNEPStructure.from_atoms(atoms, model))


def test_missing_energy_and_force_components_do_not_enter_loss():
    from torchnep.qnep import TrainingSample, TrainingConfig, supervised_loss

    model, _, structure = system()
    prediction = model(structure)
    forces = torch.full_like(prediction.forces, float("nan"))
    mask = torch.zeros_like(forces, dtype=torch.bool)
    mask[0, 0] = True
    forces[0, 0] = prediction.forces[0, 0].detach() + 2.0
    config = TrainingConfig(charge_weight=0.0)
    sample = TrainingSample(structure, forces=forces, force_mask=mask)
    loss = supervised_loss(model, [sample], config)
    assert loss.item() == pytest.approx(4.0)
    loss.backward()
    assert torch.isfinite(model.nep.c_param_2.grad).all()


def test_sample_weights_have_defined_structure_mean():
    from torchnep.qnep import TrainingSample, TrainingConfig, supervised_loss

    model, _, structure = system()
    energy = model(structure).total_energy.item()
    samples = [
        TrainingSample(structure, energy=energy + 3, weight=1),
        TrainingSample(structure, energy=energy + 6, weight=3),
    ]
    loss = supervised_loss(model, samples, TrainingConfig(charge_weight=0.0))
    assert loss.item() == pytest.approx(3.25)


def test_small_training_reduces_energy_force_loss():
    from torchnep.qnep import TrainingSample, TrainingConfig, fit

    model, _, structure = system()
    target = model(structure)
    sample = TrainingSample(
        structure,
        energy=target.total_energy.item() + 0.1,
        forces=target.forces.detach() * 1.2,
    )
    losses = fit(model, [sample], TrainingConfig(steps=25, learning_rate=0.005))
    assert losses[-1] < losses[0] * 0.1


def test_inference_inside_no_grad_returns_forces():
    model, _, structure = system()
    reference = model(structure)
    with torch.no_grad():
        result = model(structure)
    torch.testing.assert_close(result.forces, reference.forces)


@pytest.mark.parametrize(
    "metadata", [{"total_charge": 1}, {"charge": 0, "total_charge": 1}]
)
def test_total_charge_metadata_cannot_be_silently_ignored(metadata):
    from torchnep.qnep import QNEPStructure

    model, atoms, _ = system()
    atoms.info.update(metadata)
    with pytest.raises(ValueError):
        model(QNEPStructure.from_atoms(atoms, model))


def test_inconsistent_tensor_precision_is_rejected():
    model, _, structure = system()
    with pytest.raises(ValueError):
        model(structure._replace(cell=structure.cell.float()))


def test_ase_charge_metadata_change_invalidates_cached_energy():
    from torchnep.qnep import QNEPCalculator

    model, atoms, _ = system()
    atoms.calc = QNEPCalculator(model)
    atoms.get_potential_energy()
    atoms.info["total_charge"] = 1
    with pytest.raises(ValueError):
        atoms.get_potential_energy()


@pytest.mark.parametrize("l_max", [(2, 1, 0), (2, 2, 2), (1, 2, 0), (0, 0, 1)])
def test_unsupported_angular_config_is_rejected(l_max):
    from torchnep.qnep import QNEPConfig, QNEPModel

    with pytest.raises(ValueError):
        QNEPModel(QNEPConfig(type_names=("H",), l_max=l_max))
