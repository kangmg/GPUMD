from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from ase.io import read

from .checkpoint import save_checkpoint
from .data_io import load_records
from .errors import QNEPError
from .model import QNEPConfig, QNEPModel
from .run_config import QNEPDatasets, QNEPRunConfig
from .structure import QNEPStructure
from .trainer import train_qnep
from .training import TrainingConfig, TrainingSample, fit
from .training_checkpoint import TrainingCheckpointState, read_training_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a neutral 3D qNEP mode 2 reference checkpoint"
    )
    parser.add_argument(
        "dataset", type=Path, help="extxyz with frozen energy and/or forces labels"
    )
    parser.add_argument("--elements", nargs="+")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--precision", choices=("float32", "float64"))
    parser.add_argument("--early-stop-patience", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--snapshot-id")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--cutoff-radial", type=float)
    parser.add_argument("--cutoff-angular", type=float)
    parser.add_argument("--n-max-radial", type=int)
    parser.add_argument("--n-max-angular", type=int)
    parser.add_argument("--basis-size-radial", type=int)
    parser.add_argument("--basis-size-angular", type=int)
    parser.add_argument("--l-max", nargs=3, type=int)
    parser.add_argument("--neuron", type=int)
    parser.add_argument("--reciprocal-cutoff-factor", type=float)
    parser.add_argument("--energy-weight", type=float)
    parser.add_argument("--force-weight", type=float)
    parser.add_argument("--charge-weight", type=float)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    return parser


def _is_new_route(arguments: argparse.Namespace) -> bool:
    return any(
        getattr(arguments, name) is not None
        for name in (
            "validation",
            "output_dir",
            "epochs",
            "batch_size",
            "resume",
            "precision",
            "early_stop_patience",
            "checkpoint_every",
            "snapshot_id",
        )
    )


def _select_route(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> str:
    legacy = arguments.checkpoint is not None or arguments.steps is not None
    training = _is_new_route(arguments)
    if legacy and training:
        parser.error("legacy and training options cannot be combined")
    if legacy:
        if arguments.checkpoint is None:
            parser.error("legacy training requires --checkpoint")
        if arguments.elements is None:
            parser.error("legacy training requires --elements")
        return "legacy"
    if not training:
        parser.error("provide --checkpoint for legacy training or --validation for training")
    if arguments.validation is None:
        parser.error("training requires --validation")
    if arguments.output_dir is None:
        parser.error("training requires --output-dir")
    if arguments.resume is None and arguments.elements is None:
        parser.error("fresh training requires --elements")
    return "training"


def _legacy_model(arguments: argparse.Namespace) -> QNEPModel:
    torch.manual_seed(0 if arguments.seed is None else arguments.seed)
    return (
        QNEPModel(
            QNEPConfig(
                type_names=tuple(arguments.elements),
                cutoff_radial=6.0
                if arguments.cutoff_radial is None
                else arguments.cutoff_radial,
                cutoff_angular=4.0
                if arguments.cutoff_angular is None
                else arguments.cutoff_angular,
            )
        )
        .double()
        .to("cpu" if arguments.device is None else arguments.device)
    )


def _legacy_train(arguments: argparse.Namespace) -> None:
    model = _legacy_model(arguments)
    samples: list[TrainingSample] = []
    for atoms in read(arguments.dataset, index=":", format="extxyz"):
        labels = {} if atoms.calc is None else atoms.calc.results
        energy = labels.get("energy")
        forces = labels.get("forces")
        mask = atoms.arrays.get("force_mask")
        if energy is None and forces is None:
            raise QNEPError("each structure must provide an energy or forces label")
        samples.append(
            TrainingSample(
                QNEPStructure.from_atoms(atoms, model),
                None if energy is None else float(energy),
                None
                if forces is None
                else torch.as_tensor(
                    forces, dtype=torch.float64, device=model.nep.b1.device
                ),
                None
                if mask is None
                else torch.as_tensor(mask, dtype=torch.bool, device=model.nep.b1.device),
                float(atoms.info.get("sample_weight", 1.0)),
            )
        )
    losses = fit(
        model,
        samples,
        TrainingConfig(
            100 if arguments.steps is None else arguments.steps,
            0.001 if arguments.learning_rate is None else arguments.learning_rate,
            1.0 if arguments.energy_weight is None else arguments.energy_weight,
            1.0 if arguments.force_weight is None else arguments.force_weight,
            0.01 if arguments.charge_weight is None else arguments.charge_weight,
        ),
    )
    save_checkpoint(model, arguments.checkpoint)
    print(f"steps={len(losses)} initial_loss={losses[0]:.8g} final_loss={losses[-1]:.8g}")
    print(f"checkpoint={arguments.checkpoint}")


def _model_config(
    arguments: argparse.Namespace, resume: TrainingCheckpointState | None
) -> QNEPConfig:
    if resume is None:
        base = QNEPConfig(type_names=tuple(arguments.elements))
    else:
        base = resume.model_config
    values = base._asdict()
    for argument, field in (
        ("elements", "type_names"),
        ("cutoff_radial", "cutoff_radial"),
        ("cutoff_angular", "cutoff_angular"),
        ("n_max_radial", "n_max_radial"),
        ("n_max_angular", "n_max_angular"),
        ("basis_size_radial", "basis_size_radial"),
        ("basis_size_angular", "basis_size_angular"),
        ("l_max", "l_max"),
        ("neuron", "neuron"),
        ("reciprocal_cutoff_factor", "reciprocal_cutoff_factor"),
    ):
        value = getattr(arguments, argument)
        if value is not None:
            values[field] = tuple(value) if argument in {"elements", "l_max"} else value
    config = QNEPConfig(**values)
    if resume is not None and config != resume.model_config:
        raise QNEPError("explicit architecture arguments conflict with the resume checkpoint")
    return config


def _run_config(
    arguments: argparse.Namespace, resume: TrainingCheckpointState | None
) -> QNEPRunConfig:
    saved = None if resume is None else resume.resolved_config

    def value(name: str, default):
        explicit = getattr(arguments, name)
        if explicit is not None:
            return explicit
        if saved is not None:
            return saved[name]
        return default

    return QNEPRunConfig(
        output_dir=arguments.output_dir,
        epochs=value("epochs", 100),
        batch_size=value("batch_size", 1),
        learning_rate=value("learning_rate", 0.001),
        energy_weight=value("energy_weight", 1.0),
        force_weight=value("force_weight", 1.0),
        charge_weight=value("charge_weight", 0.01),
        seed=value("seed", 0),
        device=value("device", "cpu"),
        precision=value("precision", "float64"),
        early_stop_patience=value("early_stop_patience", None),
        checkpoint_every=value("checkpoint_every", 1),
        resume_from=arguments.resume,
    )


def _training_model(config: QNEPConfig, precision: str, seed: int) -> QNEPModel:
    torch.manual_seed(seed)
    dtype = torch.float32 if precision == "float32" else torch.float64
    return QNEPModel(config).to(dtype=dtype)


def _training_run(arguments: argparse.Namespace) -> None:
    resume = None if arguments.resume is None else read_training_checkpoint(arguments.resume)
    config = _run_config(arguments, resume)
    model = _training_model(
        _model_config(arguments, resume), config.precision, config.seed
    )
    datasets = QNEPDatasets(
        train=load_records(arguments.dataset, model),
        validation=load_records(arguments.validation, model),
        snapshot_id=(
            arguments.snapshot_id
            if arguments.snapshot_id is not None or resume is None
            else resume.snapshot_id
        ),
    )
    result = train_qnep(model, datasets, config)
    print(
        f"completed_epoch={result.completed_epoch} optimizer_step={result.optimizer_step} "
        f"best_epoch={result.best_epoch} stop_reason={result.stop_reason}"
    )
    print(f"best_checkpoint={result.best_inference_path}")
    print(f"latest_checkpoint={result.latest_inference_path}")
    print(f"training_checkpoint={result.training_checkpoint_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        route = _select_route(parser, arguments)
        if route == "legacy":
            _legacy_train(arguments)
        else:
            _training_run(arguments)
    except QNEPError as error:
        print(f"qnep: {error}", file=sys.stderr)
        return 1
    return 0
