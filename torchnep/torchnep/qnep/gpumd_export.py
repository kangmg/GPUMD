from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, Protocol

import torch

from ..constants import ELEMENTS
from ..model import NEPModel
from .errors import QNEPError

MAX_NEIGHBORS: Final = 819


class _NativeConfig(Protocol):
    @property
    def type_names(self) -> tuple[str, ...]: ...

    @property
    def cutoff_radial(self) -> float: ...

    @property
    def cutoff_angular(self) -> float: ...

    @property
    def n_max_radial(self) -> int: ...

    @property
    def n_max_angular(self) -> int: ...

    @property
    def basis_size_radial(self) -> int: ...

    @property
    def basis_size_angular(self) -> int: ...

    @property
    def l_max(self) -> tuple[int, int, int]: ...

    @property
    def neuron(self) -> int: ...

    @property
    def reciprocal_cutoff_factor(self) -> float: ...

    @property
    def sqrt_epsilon_inf(self) -> float: ...


class _ExportModel(Protocol):
    @property
    def config(self) -> _NativeConfig: ...

    @property
    def nep(self) -> NEPModel: ...

    @property
    def charge_weights(self) -> torch.nn.ParameterList: ...

    def state_dict(self) -> Mapping[str, torch.Tensor]: ...


def _parameter_blocks(
    model: _ExportModel,
) -> Iterator[tuple[str, torch.Tensor, tuple[int, ...]]]:
    config, state = model.config, model.state_dict()
    types, neurons = len(config.type_names), config.neuron
    orders = config.l_max[0] + bool(config.l_max[1]) + bool(config.l_max[2])
    dim = config.n_max_radial + 1 + (config.n_max_angular + 1) * orders
    for species in range(types):
        prefix = f"nep.fitting_nets.{species}"
        yield f"w0[{species}]", state[f"{prefix}.w0"].T, (neurons, dim)
        yield f"b0[{species}]", state[f"{prefix}.b0"], (neurons,)
        yield f"w1[{species}]", state[f"{prefix}.w1"], (neurons,)
        yield (
            f"charge_weights[{species}]",
            state[f"charge_weights.{species}"],
            (neurons,),
        )
    yield (
        "sqrt_epsilon_inf",
        torch.tensor(config.sqrt_epsilon_inf, dtype=torch.float64),
        (),
    )
    yield "b1", state["nep.b1"], ()
    radial_shape = (config.n_max_radial + 1, config.basis_size_radial + 1, types, types)
    angular_shape = (
        config.n_max_angular + 1,
        config.basis_size_angular + 1,
        types,
        types,
    )
    yield "c_param_2", state["nep.c_param_2"].permute(2, 3, 0, 1), radial_shape
    angular = state.get("nep.c_param_3")
    angular = (
        torch.zeros(angular_shape) if angular is None else angular.permute(2, 3, 0, 1)
    )
    yield "c_param_3", angular, angular_shape
    yield "q_scaler", state["nep.q_scaler"], (dim,)


def validate_gpumd_export(model: _ExportModel) -> None:
    """Reject configurations or parameters not representable by native mode 2."""
    config, nep = model.config, model.nep
    if config.reciprocal_cutoff_factor != 1.0:
        raise QNEPError("native qNEP export requires reciprocal_cutoff_factor == 1.0")
    if config.cutoff_angular > config.cutoff_radial:
        raise QNEPError("native qNEP export requires angular cutoff <= radial cutoff")
    if any(symbol not in ELEMENTS for symbol in config.type_names):
        raise QNEPError("native qNEP export requires recognized chemical elements")
    if nep.zbl is not None or nep.rc_radial_per_type is not None:
        raise QNEPError("native qNEP export requires uniform cutoffs without ZBL")
    for name, value, maximum in (
        ("n_max_radial", config.n_max_radial, 12),
        ("n_max_angular", config.n_max_angular, 8),
        ("basis_size_radial", config.basis_size_radial, 16),
        ("basis_size_angular", config.basis_size_angular, 12),
        ("neuron", config.neuron, 120),
    ):
        if value > maximum:
            raise QNEPError(f"native qNEP export requires {name} <= {maximum}")
    for cutoff in (config.cutoff_radial, config.cutoff_angular):
        native_cutoff = torch.tensor(cutoff, dtype=torch.float32).item()
        if not math.isfinite(native_cutoff) or native_cutoff <= 0:
            raise QNEPError(
                "native qNEP cutoffs must remain positive finite float32 values"
            )
    if len(nep.fitting_nets) != len(config.type_names) or len(
        model.charge_weights
    ) != len(config.type_names):
        raise QNEPError("native qNEP network count differs from configured elements")
    for name, tensor, shape in _parameter_blocks(model):
        if tuple(tensor.shape) != shape:
            raise QNEPError(f"native qNEP parameter shape is invalid: {name}")
        if (
            not torch.isfinite(tensor).all()
            or not torch.isfinite(tensor.to(torch.float32)).all()
        ):
            raise QNEPError(f"native qNEP parameters must be finite in float32: {name}")


def export_gpumd(model: _ExportModel, path: str | Path) -> None:
    """Atomically write native nep4_charge2 with conservative neighbor capacity.

    Both neighbor capacities are 819, enlarged to 1024 by GPUMD. The BEC-only
    sqrt_epsilon_inf metadata is preserved; the default 1.0 is not a fitted
    dielectric response. The destination parent directory must already exist.
    """
    validate_gpumd_export(model)
    config = model.config
    lines = [
        f"nep4_charge2 {len(config.type_names)} " + " ".join(config.type_names),
        f"cutoff {config.cutoff_radial:.17g} {config.cutoff_angular:.17g} "
        + f"{MAX_NEIGHBORS} {MAX_NEIGHBORS}",
        f"n_max {config.n_max_radial} {config.n_max_angular}",
        f"basis_size {config.basis_size_radial} {config.basis_size_angular}",
        "l_max " + " ".join(str(value) for value in config.l_max),
        f"ANN {config.neuron} 0",
    ]
    for _, tensor, _ in _parameter_blocks(model):
        values = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        lines.extend(f"{value.item():.17e}" for value in values)
    text = "\n".join(lines) + "\n"
    destination = Path(path)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            _ = stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
