from __future__ import annotations

from pathlib import Path
from typing import Final

import torch

from .. import __version__
from .model import QNEPConfig, QNEPModel
from .errors import QNEPError

FORMAT: Final = "torchnep-qnep-mode2-reference-v1"


def save_checkpoint(model: QNEPModel, path: str | Path) -> None:
    torch.save(
        {
            "format": FORMAT,
            "producer_version": __version__,
            "config": model.config._asdict(),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> QNEPModel:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload["format"] != FORMAT:
        raise QNEPError("unsupported qNEP checkpoint format")
    config = QNEPConfig(**payload["config"])
    model = QNEPModel(config).to(
        device=device, dtype=payload["state_dict"]["nep.b1"].dtype
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model
