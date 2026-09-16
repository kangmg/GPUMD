from __future__ import annotations

import math

import torch

from ..constants import K_C_SP
from .errors import QNEPError
from .structure import QNEPStructure


def reciprocal_energy(
    structure: QNEPStructure,
    charges: torch.Tensor,
    radial_cutoff: float,
    cutoff_factor: float = 1.0,
) -> torch.Tensor:
    """GPUMD mode 2 half-space sum, without real-space or self subtraction.

    The default sphere is |k| < 2*pi*alpha, alpha=pi/radial_cutoff.
    cutoff_factor changes only numerical reciprocal truncation for convergence tests.
    """
    if not math.isfinite(cutoff_factor) or cutoff_factor <= 0:
        raise QNEPError("reciprocal cutoff factor must be positive and finite")
    alpha = math.pi / radial_cutoff
    k_cutoff = 2 * math.pi * alpha * cutoff_factor
    cell = structure.cell
    reciprocal = 2 * math.pi * torch.linalg.inv(cell).T
    limits = (
        torch.ceil(
            k_cutoff * torch.linalg.vector_norm(cell.detach(), dim=1) / (2 * math.pi)
        )
        .to(torch.long)
        .tolist()
    )
    candidates = (limits[0] + 1) * (2 * limits[1] + 1) * (2 * limits[2] + 1)
    if candidates > 2_000_000:
        raise QNEPError(
            "reciprocal grid exceeds reference limit; reduce cell size or cutoff factor"
        )
    grids = [
        torch.arange(0, limits[0] + 1, device=cell.device),
        torch.arange(-limits[1], limits[1] + 1, device=cell.device),
        torch.arange(-limits[2], limits[2] + 1, device=cell.device),
    ]
    indices = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).reshape(-1, 3)
    half = (
        (indices[:, 0] > 0)
        | ((indices[:, 0] == 0) & (indices[:, 1] > 0))
        | ((indices[:, 0] == 0) & (indices[:, 1] == 0) & (indices[:, 2] > 0))
    )
    vectors = indices[half].to(cell.dtype) @ reciprocal
    squares = vectors.square().sum(-1)
    keep = squares.detach() < k_cutoff**2
    vectors, squares = vectors[keep], squares[keep]
    energy = charges.sum() * 0.0 + structure.positions.sum() * 0.0
    for wavevectors, k2 in zip(vectors.split(256), squares.split(256)):
        phases = structure.positions @ wavevectors.T
        real = charges @ torch.cos(phases)
        imag = charges @ torch.sin(phases)
        energy = (
            energy
            + (
                (real.square() + imag.square()) * torch.exp(-k2 / (4 * alpha**2)) / k2
            ).sum()
        )
    return energy * (4 * math.pi * K_C_SP / torch.linalg.det(cell).abs())
