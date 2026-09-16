from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from ..data import build_neighbor_list_np_ex
from .errors import QNEPError

if TYPE_CHECKING:
    from ase import Atoms
    from .model import QNEPModel


class QNEPStructure(NamedTuple):
    positions: torch.Tensor
    cell: torch.Tensor
    atom_types: torch.Tensor
    pbc: tuple[bool, bool, bool] = (True, True, True)
    total_charge: float = 0.0

    @classmethod
    def from_atoms(cls, atoms: Atoms, model: QNEPModel) -> QNEPStructure:
        reference = model.nep.b1
        types = [
            model.config.type_names.index(symbol)
            for symbol in atoms.get_chemical_symbols()
        ]
        total_charge = atoms.info.get("total_charge")
        charge_alias = atoms.info.get("charge")
        if (
            total_charge is not None
            and charge_alias is not None
            and float(total_charge) != float(charge_alias)
        ):
            raise QNEPError("charge and total_charge metadata disagree")
        charge = float(
            total_charge
            if total_charge is not None
            else charge_alias
            if charge_alias is not None
            else atoms.get_initial_charges().sum()
        )
        return cls(
            torch.tensor(
                atoms.positions, dtype=reference.dtype, device=reference.device
            ),
            torch.tensor(
                atoms.cell.array, dtype=reference.dtype, device=reference.device
            ),
            torch.tensor(types, dtype=torch.long, device=reference.device),
            tuple(bool(value) for value in atoms.pbc),
            charge,
        )

    def validate(self, num_types: int) -> None:
        if self.positions.dtype not in (torch.float32, torch.float64):
            raise QNEPError("positions must use float32 or float64")
        if self.cell.dtype != self.positions.dtype:
            raise QNEPError("positions and cell must have the same dtype")
        if (
            self.cell.device != self.positions.device
            or self.atom_types.device != self.positions.device
        ):
            raise QNEPError("positions, cell and atom_types must share a device")
        if (
            self.positions.ndim != 2
            or self.positions.shape[1] != 3
            or len(self.positions) == 0
        ):
            raise QNEPError("positions must have shape (N, 3) with N > 0")
        if self.cell.shape != (3, 3) or not torch.isfinite(self.cell).all():
            raise QNEPError("cell must be a finite 3 by 3 matrix")
        if abs(torch.linalg.det(self.cell).item()) < 1e-10:
            raise QNEPError("cell must have nonzero volume")
        if not all(self.pbc) or len(self.pbc) != 3 or self.total_charge != 0.0:
            raise QNEPError(
                "qNEP reference supports only neutral, fully 3D periodic structures"
            )
        if not torch.isfinite(self.positions).all():
            raise QNEPError("positions must be finite")
        if (
            self.atom_types.shape != (len(self.positions),)
            or self.atom_types.dtype != torch.long
        ):
            raise QNEPError("atom_types must be an int64 vector of length N")
        if (self.atom_types < 0).any() or (self.atom_types >= num_types).any():
            raise QNEPError("atom_types contains an unsupported species index")


class Neighbors(NamedTuple):
    centers: torch.Tensor
    neighbors: torch.Tensor
    vectors: torch.Tensor


def connected_neighbors(structure: QNEPStructure, cutoff: float) -> Neighbors:
    positions = structure.positions.detach().cpu().numpy()
    cell = structure.cell.detach().cpu().numpy()
    i, j, vectors, _, _ = build_neighbor_list_np_ex(positions, cell, cutoff)
    # Recover integer images relative to the original, possibly unwrapped positions.
    shifts = np.rint((vectors - (positions[j] - positions[i])) @ np.linalg.inv(cell))
    device = structure.positions.device
    centers = torch.tensor(i, device=device)
    neighbors = torch.tensor(j, device=device)
    images = torch.tensor(shifts, device=device, dtype=structure.positions.dtype)
    connected = (
        structure.positions[neighbors]
        - structure.positions[centers]
        + images @ structure.cell
    )
    return Neighbors(centers, neighbors, connected)
