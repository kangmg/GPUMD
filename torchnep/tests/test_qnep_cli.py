import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from test_qnep_reference import system
from torchnep.qnep import QNEPCalculator, load_checkpoint


def run_cli(arguments, directory):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    return subprocess.run(
        [sys.executable, "-m", "torchnep.qnep", *arguments],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_help(tmp_path):
    result = run_cli(["--help"], tmp_path)
    assert result.returncode == 0
    assert "--checkpoint" in result.stdout


def test_cli_trains_checkpoint_consumed_by_ase(tmp_path):
    model, atoms, structure = system()
    target = model(structure)
    atoms.calc = SinglePointCalculator(
        atoms, energy=target.total_energy.item(), forces=target.forces.detach().numpy()
    )
    write(tmp_path / "train.xyz", atoms, format="extxyz")
    result = run_cli(
        [
            "train.xyz",
            "--elements",
            "H",
            "O",
            "--steps",
            "5",
            "--checkpoint",
            "qnep.pt",
        ],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    atoms.calc = QNEPCalculator(load_checkpoint(tmp_path / "qnep.pt"))
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()


def test_cli_rejects_nonperiodic_training_structure(tmp_path):
    _, atoms, _ = system()
    atoms.pbc = False
    atoms.calc = SinglePointCalculator(atoms, energy=0.0)
    write(tmp_path / "train.xyz", atoms, format="extxyz")
    result = run_cli(
        [
            "train.xyz",
            "--elements",
            "H",
            "O",
            "--steps",
            "1",
            "--checkpoint",
            "qnep.pt",
        ],
        tmp_path,
    )
    assert result.returncode != 0
    assert not (tmp_path / "qnep.pt").exists()
