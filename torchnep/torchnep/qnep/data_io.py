from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import iread

from .errors import QNEPError
from .model import QNEPModel
from .records import QNEPRecord
from .structure import QNEPStructure
from .training import TrainingSample


def _mask_tensor(value: np.ndarray, shape: torch.Size) -> torch.Tensor:
    array = np.asarray(value)
    if array.shape != tuple(shape):
        raise QNEPError("force_mask must have shape (N, 3)")
    if np.issubdtype(array.dtype, np.bool_):
        return torch.as_tensor(array, dtype=torch.bool)
    if not np.issubdtype(array.dtype, np.number) or not np.isin(array, (0, 1)).all():
        raise QNEPError("force_mask values must be explicit booleans or 0/1")
    return torch.as_tensor(array, dtype=torch.bool)


def _optional_id(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise QNEPError(f"{name} must be a nonempty string or absent")
    return value


def _record_from_atoms(atoms: Atoms, model: QNEPModel) -> QNEPRecord:
    labels = {} if atoms.calc is None else atoms.calc.results
    symbols = atoms.get_chemical_symbols()
    unsupported = next(
        (symbol for symbol in symbols if symbol not in model.config.type_names), None
    )
    if unsupported is not None:
        raise QNEPError(f"unsupported element {unsupported!r}")
    atom_types = [model.config.type_names.index(symbol) for symbol in symbols]
    dtype = model.nep.b1.dtype
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
    structure = QNEPStructure(
        positions=torch.as_tensor(atoms.positions, dtype=dtype).clone(),
        cell=torch.as_tensor(atoms.cell.array, dtype=dtype).clone(),
        atom_types=torch.tensor(atom_types, dtype=torch.long),
        pbc=(bool(atoms.pbc[0]), bool(atoms.pbc[1]), bool(atoms.pbc[2])),
        total_charge=charge,
    )
    energy_value = labels.get("energy")
    energy = None if energy_value is None else float(energy_value)
    forces_value = labels.get("forces")
    if forces_value is None:
        forces_value = atoms.arrays.get("force")
    forces = (
        None
        if forces_value is None
        else torch.as_tensor(forces_value, dtype=dtype).clone()
    )
    mask_value = atoms.arrays.get("force_mask")
    mask = None if mask_value is None else _mask_tensor(mask_value, structure.positions.shape)
    weight = float(atoms.info.get("sample_weight", 1.0))
    if not math.isfinite(weight) or weight <= 0:
        raise QNEPError("sample weight must be positive and finite")
    structure.validate(len(model.config.type_names))
    return QNEPRecord(
        TrainingSample(structure, energy, forces, mask, weight),
        _optional_id(atoms.info.get("record_id"), "record_id"),
        _optional_id(atoms.info.get("group_id"), "group_id"),
    )


def load_records(path: str | Path, model: QNEPModel) -> tuple[QNEPRecord, ...]:
    source = Path(path)
    if not source.is_file():
        raise QNEPError(f"dataset file does not exist: {source}")
    records: list[QNEPRecord] = []
    frame = -1
    try:
        for frame, atoms in enumerate(iread(source, index=":", format="extxyz")):
            try:
                records.append(_record_from_atoms(atoms, model))
            except QNEPError as error:
                raise QNEPError(f"{source}: frame {frame}: {error}") from error
            except (ValueError, TypeError, OverflowError) as error:
                raise QNEPError(
                    f"{source}: frame {frame}: invalid record metadata: {error}"
                ) from error
    except QNEPError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError) as error:
        raise QNEPError(f"{source}: frame {frame + 1}: invalid extxyz: {error}") from error
    if not records:
        raise QNEPError(f"{source}: extxyz dataset is empty")
    return tuple(records)
