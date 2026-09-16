import copy
import itertools

import pytest
import torch
from ase import Atoms

from torchnep.qnep import QNEPConfig, QNEPModel, QNEPStructure
from torchnep.qnep.batching import (
    backward_weighted_batch,
    make_shuffle_generator,
    shuffled_batches,
)
from torchnep.qnep.errors import QNEPError
from torchnep.qnep.training import (
    TrainingConfig,
    TrainingSample,
    per_sample_loss,
    supervised_loss,
)


def _model_and_samples():
    torch.manual_seed(29)
    model = QNEPModel(
        QNEPConfig(
            type_names=("H", "O"),
            cutoff_radial=3.5,
            cutoff_angular=3.0,
            neuron=6,
        )
    ).double()
    atoms = Atoms(
        "OH2",
        positions=[[1.0, 1.0, 1.0], [1.93, 1.0, 1.0], [0.78, 1.90, 1.13]],
        cell=[7, 8, 9],
        pbc=True,
    )
    structure = QNEPStructure.from_atoms(atoms, model)
    reference = model(structure)
    samples = tuple(
        TrainingSample(
            structure,
            energy=reference.total_energy.detach().item() + energy_offset,
            forces=reference.forces.detach() + force_offset,
            force_mask=force_mask,
            weight=weight,
        )
        for energy_offset, force_offset, force_mask, weight in (
            (0.2, 0.1, None, 1.0),
            (-0.4, -0.2, torch.tensor([[True, True, False], [True, False, True], [False, True, True]]), 3.0),
            (0.6, 0.3, None, 7.0),
            (-0.8, -0.4, torch.tensor([[True, False, True], [True, True, False], [False, True, False]]), 11.0),
        )
    )
    return model, samples


def _parameter_gradients(model):
    return tuple(parameter.grad.detach().clone() for parameter in model.parameters())


def test_weighted_full_batch_backward_matches_reference_loss_and_gradients():
    base_model, samples = _model_and_samples()
    config = TrainingConfig(energy_weight=1.3, force_weight=0.7, charge_weight=0.05)
    reference_model = copy.deepcopy(base_model)
    batched_model = copy.deepcopy(base_model)

    reference_loss = supervised_loss(reference_model, samples, config)
    reference_loss.backward()

    batch_loss = backward_weighted_batch(
        batched_model,
        samples,
        config,
        global_record_count=len(samples),
        global_weight=sum(sample.weight for sample in samples),
    )

    assert batch_loss == pytest.approx(reference_loss.detach().item(), abs=1e-10)
    for actual, expected in zip(
        _parameter_gradients(batched_model), _parameter_gradients(reference_model)
    ):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)


def test_equal_size_subset_gradients_are_unbiased_and_short_batch_uses_its_actual_size():
    base_model, samples = _model_and_samples()
    config = TrainingConfig(energy_weight=1.3, force_weight=0.7, charge_weight=0.05)
    global_weight = sum(sample.weight for sample in samples)

    full_model = copy.deepcopy(base_model)
    backward_weighted_batch(
        full_model,
        samples,
        config,
        global_record_count=len(samples),
        global_weight=global_weight,
    )
    full_gradients = _parameter_gradients(full_model)

    subset_gradients = []
    for indices in itertools.combinations(range(len(samples)), 2):
        subset_model = copy.deepcopy(base_model)
        backward_weighted_batch(
            subset_model,
            tuple(samples[index] for index in indices),
            config,
            global_record_count=len(samples),
            global_weight=global_weight,
        )
        subset_gradients.append(_parameter_gradients(subset_model))
    for parameter_index, expected in enumerate(full_gradients):
        mean_gradient = torch.stack(
            [gradient[parameter_index] for gradient in subset_gradients]
        ).mean(dim=0)
        torch.testing.assert_close(mean_gradient, expected, atol=1e-10, rtol=1e-8)

    generator = make_shuffle_generator(13)
    batches = shuffled_batches(len(samples), 3, generator)
    assert tuple(index for batch in batches for index in batch) == tuple(
        torch.randperm(
            len(samples), generator=make_shuffle_generator(13)
        ).tolist()
    )
    assert len(batches) == 2
    assert len(batches[-1]) == 1

    short_model = copy.deepcopy(base_model)
    last_sample = samples[batches[-1][0]]
    expected = (
        len(samples)
        / global_weight
        * last_sample.weight
        * per_sample_loss(short_model, last_sample, config).detach().item()
    )
    actual = backward_weighted_batch(
        short_model,
        (last_sample,),
        config,
        global_record_count=len(samples),
        global_weight=global_weight,
    )
    assert actual == pytest.approx(expected, abs=1e-10)


def test_batching_rejects_empty_or_invalid_scale_and_nonfinite_loss():
    model, samples = _model_and_samples()
    config = TrainingConfig()

    with pytest.raises(QNEPError):
        shuffled_batches(0, 1, make_shuffle_generator(0))
    with pytest.raises(QNEPError):
        backward_weighted_batch(
            model,
            (),
            config,
            global_record_count=4,
            global_weight=22.0,
        )
    with pytest.raises(QNEPError):
        backward_weighted_batch(
            model,
            samples[:1],
            config,
            global_record_count=4,
            global_weight=float("nan"),
        )

    nonfinite_model = copy.deepcopy(model)
    with torch.no_grad():
        nonfinite_model.nep.b1.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        backward_weighted_batch(
            nonfinite_model,
            samples[:1],
            config,
            global_record_count=4,
            global_weight=22.0,
        )
