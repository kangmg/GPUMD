#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.0", "numpy>=1.20"]
# ///
# ─── How to run ───
# Import batch_forward from the benchmark driver with torchnep on PYTHONPATH.
# CPU smoke test, using the repository's existing environment without installs:
# PYTHONPATH=./torchnep CUDA_VISIBLE_DEVICES='' uv run --no-project \
#   "$QNEP_BENCH_PYTHON" \
#   torchnep/benchmarks/qnep/batch_prototype.py
# ──────────────────
"""Experimental homogeneous, fixed-cell qNEP tensor batching.

Every input must have identical N, cell values, and atom-type layout, and cells
must not require gradients. Coordinate, charge, and force-loss parameter graphs
stay connected. Energies are [B], charges [B, N], and forces [B, N, 3].
Neighbors are rebuilt separately for every input on every call. The descriptor
backend remains ``loop``. A fresh shared-cell reciprocal grid is built once per
batch: this amortization is explicitly part of the measured treatment. The sum
uses the reference half-space ordering and 256-vector chunks, without PPPM.
"""

from __future__ import annotations

import math

import torch

from torchnep.constants import K_C_SP
from torchnep.qnep.errors import QNEPError
from torchnep.qnep.model import QNEPConfig, QNEPModel, QNEPResult
from torchnep.qnep.structure import QNEPStructure, connected_neighbors


def batch_forward(
    model: QNEPModel,
    structures: tuple[QNEPStructure, ...],
    create_graph: bool = False,
) -> QNEPResult:
    if not structures:
        raise QNEPError("batch must contain at least one structure")
    first = structures[0]
    for structure in structures:
        structure.validate(model.nep.num_types)
        if (
            structure.positions.dtype != model.nep.b1.dtype
            or structure.positions.device != model.nep.b1.device
        ):
            raise QNEPError("structure and model must share dtype and device")
        if (
            structure.positions.shape != first.positions.shape
            or not torch.equal(structure.cell, first.cell)
            or not torch.equal(structure.atom_types, first.atom_types)
        ):
            raise QNEPError("batch requires identical N, cell, and atom-type layout")
        if structure.cell.requires_grad:
            raise QNEPError("batch prototype requires fixed cells without gradients")
    with torch.enable_grad():
        positions = torch.stack([structure.positions for structure in structures])
        if not positions.requires_grad:
            positions = positions.detach().requires_grad_(True)
        batch_size, atom_count = positions.shape[:2]
        pair_lists = tuple(
            connected_neighbors(
                structure._replace(positions=positions[index]),
                max(model.config.cutoff_radial, model.config.cutoff_angular),
            )
            for index, structure in enumerate(structures)
        )
        vectors = torch.cat([pairs.vectors for pairs in pair_lists])
        centers = torch.cat(
            [pairs.centers + index * atom_count for index, pairs in enumerate(pair_lists)]
        )
        neighbors = torch.cat(
            [pairs.neighbors + index * atom_count for index, pairs in enumerate(pair_lists)]
        )
        atom_types = first.atom_types.repeat(batch_size)
        distances = vectors.detach().norm(dim=-1)
        radial = distances < model.config.cutoff_radial
        angular = distances < model.config.cutoff_angular
        descriptors = model.nep.compute_descriptors(
            vectors[radial], vectors[angular], centers[radial], neighbors[radial],
            centers[angular], neighbors[angular], atom_types, batch_size * atom_count,
            backend="loop",
        ) * model.nep.get_buffer("q_scaler")
        energies = torch.zeros_like(positions[:, :, 0]).reshape(-1)
        raw_charges = torch.zeros_like(energies)
        for species, network in enumerate(model.nep.fitting_nets):
            mask = atom_types == species
            hidden = torch.tanh(descriptors[mask] @ network.get_parameter("w0") - network.get_parameter("b0"))
            energies = energies.index_put((mask,), hidden @ network.get_parameter("w1"))
            raw_charges = raw_charges.index_put(
                (mask,), hidden @ model.charge_weights[species]
            )
        short_range = (energies.reshape(batch_size, atom_count) - model.nep.b1).sum(-1)
        raw_charges = raw_charges.reshape(batch_size, atom_count)
        charges = raw_charges - raw_charges.mean(dim=1, keepdim=True)
        alpha = math.pi / model.config.cutoff_radial
        factor = model.config.reciprocal_cutoff_factor
        if not math.isfinite(factor) or factor <= 0:
            raise QNEPError("reciprocal cutoff factor must be positive and finite")
        k_cutoff = 2 * math.pi * alpha * factor
        cell = first.cell
        reciprocal = 2 * math.pi * torch.linalg.inv(cell).T
        limits: list[int] = (
            torch.ceil(
                k_cutoff * torch.linalg.vector_norm(cell.detach(), dim=1)
                / (2 * math.pi)
            ).to(torch.long).tolist()
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
        wavevectors = indices[half].to(cell.dtype) @ reciprocal
        squares = wavevectors.square().sum(-1)
        keep = squares.detach() < k_cutoff**2
        wavevectors, squares = wavevectors[keep], squares[keep]
        electrostatic = charges.sum(-1) * 0.0 + positions.sum(dim=(1, 2)) * 0.0
        for chunk, k2 in zip(wavevectors.split(256), squares.split(256), strict=True):
            phases = positions @ chunk.T
            real = torch.bmm(charges[:, None, :], torch.cos(phases)).squeeze(1)
            imag = torch.bmm(charges[:, None, :], torch.sin(phases)).squeeze(1)
            electrostatic = electrostatic + (
                (real.square() + imag.square()) * torch.exp(-k2 / (4 * alpha**2)) / k2
            ).sum(-1)
        electrostatic = electrostatic * (4 * math.pi * K_C_SP / torch.linalg.det(cell).abs())
        total = short_range + electrostatic + positions.sum(dim=(1, 2)) * 0.0
        forces = -torch.autograd.grad(
            total.sum(), positions, create_graph=create_graph, retain_graph=True
        )[0]
    return QNEPResult(total, short_range, electrostatic, raw_charges, charges, forces)


def _cpu_smoke_test() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(29)
    model = QNEPModel(
        QNEPConfig(
            type_names=("H", "O"), cutoff_radial=3.5, cutoff_angular=3.0, neuron=6
        )
    ).double().cpu()
    cell = torch.tensor(
        [[7.0, 0.0, 0.0], [0.3, 8.0, 0.0], [0.2, 0.4, 9.0]],
        dtype=torch.float64,
    )
    atom_types = torch.tensor([1, 0, 0], dtype=torch.long)
    base = torch.tensor(
        [[1.0, 1.0, 1.0], [1.93, 1.0, 1.0], [0.78, 1.90, 1.13]],
        dtype=torch.float64,
    )
    displacement = torch.tensor(
        [[0.01, 0.02, -0.01], [0.02, -0.01, 0.03], [-0.03, 0.01, 0.02]],
        dtype=torch.float64,
    )
    structures = tuple(
        QNEPStructure(positions.requires_grad_(), cell, atom_types)
        for positions in (base.clone(), base + displacement)
    )
    reference = tuple(model(structure, create_graph=True) for structure in structures)
    actual = batch_forward(model, structures, create_graph=True)
    for values, expected_values in zip(actual, zip(*reference, strict=True), strict=True):
        expected = torch.stack(expected_values)
        torch.testing.assert_close(values, expected, atol=2e-12, rtol=2e-10)
    coordinate_grads = torch.autograd.grad(
        actual.total_energy.sum(),
        tuple(structure.positions for structure in structures),
        retain_graph=True,
    )
    torch.testing.assert_close(torch.stack(coordinate_grads), -actual.forces)
    actual_loss = actual.forces.square().mean()
    expected_loss = torch.stack([result.forces.square().mean() for result in reference]).mean()
    if not torch.isfinite(actual_loss) or not torch.isfinite(expected_loss):
        raise QNEPError("CPU smoke test produced a nonfinite force loss")
    parameters = tuple(model.parameters())
    actual_grads = torch.autograd.grad(actual_loss, parameters, allow_unused=True)
    expected_grads = torch.autograd.grad(expected_loss, parameters, allow_unused=True)
    maximum_error = 0.0
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        if actual_grad is None or expected_grad is None:
            if actual_grad is not expected_grad:
                raise QNEPError("CPU smoke test gradient connectivity differs")
            continue
        if not torch.isfinite(actual_grad).all() or not torch.isfinite(expected_grad).all():
            raise QNEPError("CPU smoke test produced a nonfinite parameter gradient")
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-12, rtol=2e-9)
        maximum_error = max(maximum_error, (actual_grad - expected_grad).abs().max().item())
    print(f"CPU B2 float64 output and coordinate parity passed; force-loss gradient max abs error={maximum_error:.3e}")


if __name__ == "__main__":
    _cpu_smoke_test()
