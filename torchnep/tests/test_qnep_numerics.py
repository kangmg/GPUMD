import itertools
import math

import numpy as np
import pytest
import torch

from torchnep.constants import K_C_SP
from torchnep.qnep import QNEPStructure
from torchnep.qnep.electrostatics import reciprocal_energy


def test_reciprocal_energy_matches_independent_full_space_sum():
    positions = np.array(
        [[0.2, 0.4, 0.7], [1.1, 0.3, 1.4], [0.6, 1.5, 0.2]], dtype=np.float64
    )
    cell = np.array(
        [[4.1, 0.0, 0.0], [0.7, 4.8, 0.0], [0.2, 0.5, 5.3]], dtype=np.float64
    )
    charges = np.array([0.37, -0.21, -0.16], dtype=np.float64)
    radial_cutoff = 3.2
    alpha = math.pi / radial_cutoff
    k_cutoff = 2.0 * math.pi * alpha
    reciprocal = 2.0 * math.pi * np.linalg.inv(cell).T
    limits = np.ceil(
        k_cutoff * np.linalg.norm(cell, axis=1) / (2.0 * math.pi)
    ).astype(int)

    full_sum = 0.0
    for indices in itertools.product(
        range(-limits[0], limits[0] + 1),
        range(-limits[1], limits[1] + 1),
        range(-limits[2], limits[2] + 1),
    ):
        if indices == (0, 0, 0):
            continue
        wavevector = np.asarray(indices) @ reciprocal
        k2 = float(wavevector @ wavevector)
        if k2 >= k_cutoff**2:
            continue
        phases = positions @ wavevector
        structure_factor = np.sum(charges * np.exp(1j * phases))
        full_sum += (
            abs(structure_factor) ** 2 * math.exp(-k2 / (4.0 * alpha**2)) / k2
        )
    expected = 2.0 * math.pi * K_C_SP * full_sum / abs(np.linalg.det(cell))

    structure = QNEPStructure(
        positions=torch.from_numpy(positions),
        cell=torch.from_numpy(cell),
        atom_types=torch.zeros(3, dtype=torch.long),
    )
    actual = reciprocal_energy(
        structure, torch.from_numpy(charges), radial_cutoff=radial_cutoff
    )
    assert actual.item() == pytest.approx(expected, rel=2e-13, abs=1e-14)


def test_reciprocal_grid_limit_rejects_oversized_reference_sum():
    structure = QNEPStructure(
        positions=torch.zeros((1, 3), dtype=torch.float64),
        cell=torch.eye(3, dtype=torch.float64) * 1_000.0,
        atom_types=torch.zeros(1, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="reciprocal grid exceeds reference limit"):
        reciprocal_energy(
            structure,
            torch.zeros(1, dtype=torch.float64),
            radial_cutoff=1.0,
        )
