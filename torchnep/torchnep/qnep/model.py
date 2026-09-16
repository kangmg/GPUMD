from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn

from ..model import NEPModel
from .electrostatics import reciprocal_energy
from .errors import QNEPError
from .structure import QNEPStructure, connected_neighbors


class QNEPConfig(NamedTuple):
    type_names: tuple[str, ...]
    cutoff_radial: float = 6.0
    cutoff_angular: float = 4.0
    n_max_radial: int = 2
    n_max_angular: int = 2
    basis_size_radial: int = 4
    basis_size_angular: int = 4
    l_max: tuple[int, int, int] = (2, 0, 0)
    neuron: int = 16
    reciprocal_cutoff_factor: float = 1.0


class QNEPResult(NamedTuple):
    total_energy: torch.Tensor
    short_range_energy: torch.Tensor
    electrostatic_energy: torch.Tensor
    raw_charges: torch.Tensor
    charges: torch.Tensor
    forces: torch.Tensor


class QNEPModel(nn.Module):
    """Trainable neutral mode 2 reference using shared NEP hidden representations."""

    def __init__(self, config: QNEPConfig):
        super().__init__()
        if not config.type_names or len(set(config.type_names)) != len(
            config.type_names
        ):
            raise QNEPError("type_names must contain distinct elements")
        for cutoff in (
            config.cutoff_radial,
            config.cutoff_angular,
            config.reciprocal_cutoff_factor,
        ):
            if not math.isfinite(cutoff) or cutoff <= 0:
                raise QNEPError("cutoffs must be positive and finite")
        if (
            config.neuron < 1
            or min(
                config.n_max_radial,
                config.n_max_angular,
                config.basis_size_radial,
                config.basis_size_angular,
            )
            < 0
        ):
            raise QNEPError("invalid descriptor or hidden layer size")
        if len(config.l_max) != 3 or not 0 <= config.l_max[0] <= 4:
            raise QNEPError(
                "reference l_max requires three entries and angular order 0..4"
            )
        if config.l_max[1] not in (0, 2) or config.l_max[2] not in (0, 1):
            raise QNEPError("supported higher-body l_max entries are 0/2 and 0/1")
        if (config.l_max[1] and config.l_max[0] < 2) or (
            config.l_max[2] and config.l_max[0] < 1
        ):
            raise QNEPError(
                "higher-body invariants require the corresponding angular order"
            )
        self.config = config
        nep_config = config._asdict()
        nep_config["num_types"] = len(config.type_names)
        self.nep = NEPModel(nep_config)
        self.charge_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(config.neuron)) for _ in config.type_names]
        )
        for weight in self.charge_weights:
            nn.init.normal_(weight, std=0.01)

    def forward(
        self, structure: QNEPStructure, create_graph: bool = False
    ) -> QNEPResult:
        structure.validate(self.nep.num_types)
        if (
            structure.positions.dtype != self.nep.b1.dtype
            or structure.positions.device != self.nep.b1.device
        ):
            raise QNEPError("structure and model must share dtype and device")
        with torch.enable_grad():
            positions = structure.positions
            if not positions.requires_grad:
                positions = positions.detach().requires_grad_(True)
            connected = structure._replace(positions=positions)
            pairs = connected_neighbors(
                connected, max(self.config.cutoff_radial, self.config.cutoff_angular)
            )
            distances = pairs.vectors.detach().norm(dim=-1)
            radial = distances < self.config.cutoff_radial
            angular = distances < self.config.cutoff_angular
            descriptors = (
                self.nep.compute_descriptors(
                    pairs.vectors[radial],
                    pairs.vectors[angular],
                    pairs.centers[radial],
                    pairs.neighbors[radial],
                    pairs.centers[angular],
                    pairs.neighbors[angular],
                    structure.atom_types,
                    len(positions),
                    backend="loop",
                )
                * self.nep.q_scaler
            )
            energies = torch.zeros_like(positions[:, 0])
            raw_charges = torch.zeros_like(energies)
            for species, network in enumerate(self.nep.fitting_nets):
                mask = structure.atom_types == species
                hidden = torch.tanh(descriptors[mask] @ network.w0 - network.b0)
                energies = energies.index_put((mask,), hidden @ network.w1)
                raw_charges = raw_charges.index_put(
                    (mask,), hidden @ self.charge_weights[species]
                )
            short_range = (energies - self.nep.b1).sum()
            charges = raw_charges - raw_charges.mean()
            electrostatic = reciprocal_energy(
                connected,
                charges,
                self.config.cutoff_radial,
                self.config.reciprocal_cutoff_factor,
            )
            total = short_range + electrostatic + positions.sum() * 0.0
            forces = -torch.autograd.grad(
                total, positions, create_graph=create_graph, retain_graph=True
            )[0]
        return QNEPResult(
            total, short_range, electrostatic, raw_charges, charges, forces
        )

    def export_nep(self, path: str) -> None:
        raise NotImplementedError(
            "qNEP reference checkpoints cannot be exported as GPUMD nep.txt"
        )
