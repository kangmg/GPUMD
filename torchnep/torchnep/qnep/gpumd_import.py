from __future__ import annotations

from pathlib import Path

import torch

from .errors import QNEPError
from .model import QNEPConfig, QNEPModel


def load_gpumd_reference(path: str | Path) -> QNEPModel:
    """Import uniform-cutoff, non-ZBL nep4_charge2 weights for E/F parity checks."""
    lines = Path(path).read_text().splitlines()
    header = lines[0].split()
    if header[0] != "nep4_charge2" or len(header) != 2 + int(header[1]):
        raise QNEPError("reference import requires a non-ZBL nep4_charge2 model")
    expected = ("cutoff", "n_max", "basis_size", "l_max", "ANN")
    fields = [line.split() for line in lines[1:6]]
    if tuple(row[0] for row in fields) != expected:
        raise QNEPError("unsupported qNEP header layout")
    if tuple(len(row) for row in fields) != (5, 3, 3, 4, 3):
        raise QNEPError(
            "reference import supports uniform cutoffs and three l_max entries"
        )
    config = QNEPConfig(
        type_names=tuple(header[2:]),
        cutoff_radial=float(fields[0][1]),
        cutoff_angular=float(fields[0][2]),
        n_max_radial=int(fields[1][1]),
        n_max_angular=int(fields[1][2]),
        basis_size_radial=int(fields[2][1]),
        basis_size_angular=int(fields[2][2]),
        l_max=tuple(int(value) for value in fields[3][1:]),
        neuron=int(fields[4][1]),
    )
    model = QNEPModel(config).double()
    values = torch.tensor(
        [float(token) for line in lines[6:] for token in line.split()],
        dtype=torch.float64,
    )
    types, neurons, dim = len(config.type_names), config.neuron, model.nep.dim
    radial_count = types**2 * (config.n_max_radial + 1) * (config.basis_size_radial + 1)
    angular_count = (
        types**2 * (config.n_max_angular + 1) * (config.basis_size_angular + 1)
    )
    count = types * (dim + 3) * neurons + 2 + radial_count + angular_count + dim
    if len(values) != count or not torch.isfinite(values).all():
        raise QNEPError("qNEP parameter count or numeric values are invalid")
    offset = 0
    with torch.no_grad():
        for species, net in enumerate(model.nep.fitting_nets):
            net.w0.copy_(
                values[offset : offset + neurons * dim].reshape(neurons, dim).T
            )
            offset += neurons * dim
            net.b0.copy_(values[offset : offset + neurons])
            offset += neurons
            net.w1.copy_(values[offset : offset + neurons])
            offset += neurons
            model.charge_weights[species].copy_(values[offset : offset + neurons])
            offset += neurons
        model.config = config._replace(sqrt_epsilon_inf=values[offset].item())
        offset += 1
        model.nep.b1.copy_(values[offset])
        offset += 1
        radial = values[offset : offset + radial_count].reshape(
            config.n_max_radial + 1, config.basis_size_radial + 1, types, types
        )
        model.nep.c_param_2.copy_(radial.permute(2, 3, 0, 1))
        offset += radial_count
        if model.nep.c_param_3 is not None:
            angular = values[offset : offset + angular_count].reshape(
                config.n_max_angular + 1, config.basis_size_angular + 1, types, types
            )
            model.nep.c_param_3.copy_(angular.permute(2, 3, 0, 1))
        offset += angular_count
        model.nep.q_scaler.copy_(values[offset : offset + dim])
    return model
