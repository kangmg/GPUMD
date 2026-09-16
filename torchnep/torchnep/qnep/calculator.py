from __future__ import annotations

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from .model import QNEPModel
from .structure import QNEPStructure


class QNEPCalculator(Calculator):
    implemented_properties = ["energy", "forces", "charges"]

    def __init__(self, model: QNEPModel):
        super().__init__()
        self.model = model

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> list[str]:
        changes = super().check_state(atoms, tol=tol)
        if self.atoms is not None and any(
            atoms.info.get(key) != self.atoms.info.get(key)
            for key in ("total_charge", "charge")
        ):
            changes.append("total_charge")
        return changes

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: tuple[str, ...] = ("energy", "forces"),
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        result = self.model(QNEPStructure.from_atoms(self.atoms, self.model))
        self.results = {
            "energy": result.total_energy.detach().item(),
            "forces": result.forces.detach().cpu().numpy(),
            "charges": result.charges.detach().cpu().numpy(),
        }
