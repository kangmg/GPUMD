import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read, write

from torchnep.qnep import QNEPStructure, load_gpumd_reference

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "tests_pytest/fixtures/models/qnep_mode2_water.txt"
STRUCTURE = ROOT / "tests_pytest/fixtures/structures/water-nat63-from-md.xyz"
BATIO3_MODEL = ROOT / "tests_pytest/fixtures/models/qnep_mode2_BaTiO3.txt"
BATIO3_STRUCTURE = ROOT / "tests_pytest/fixtures/structures/BaTiO3-nat40-rattled.xyz"


def derive_network_only_model(source, destination):
    lines = source.read_text().splitlines(keepends=True)
    header = lines[0].split(maxsplit=1)
    assert header[0] == "nep4_zbl_charge2"
    assert lines[1].split()[0] == "zbl"
    lines[0] = lines[0].replace("nep4_zbl_charge2", "nep4_charge2", 1)
    destination.write_text("".join([lines[0], *lines[2:]]))
    return destination


def run_gpumd_case(binary, model_path, structure_path, case_dir):
    case_dir.mkdir(parents=True, exist_ok=True)
    model = load_gpumd_reference(model_path)
    atoms = read(structure_path)
    initial_positions = atoms.positions.copy()
    reference = model(QNEPStructure.from_atoms(atoms, model))
    write(case_dir / "model.xyz", atoms, format="extxyz")
    (case_dir / "run.in").write_text(
        f"potential {model_path}\nkspace ewald\n"
        "dump_xyz 1 result.xyz precision double force\n"
        "dump_thermo 1\nvelocity 1e-24\ntime_step 1e-8\nensemble nve\nrun 1\n"
    )
    with (case_dir / "stdout.log").open("w") as stdout:
        subprocess.run(
            [binary],
            cwd=case_dir,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=60,
        )
    energy = np.loadtxt(case_dir / "thermo.out")[2]
    result = read(case_dir / "result.xyz")
    forces = result.get_forces()
    displacement = result.positions - initial_positions
    fractional = np.linalg.solve(atoms.cell.array.T, displacement.T).T
    minimum_image = (fractional - np.rint(fractional)) @ atoms.cell.array
    metrics = {
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "model": str(model_path),
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "structure": str(structure_path),
        "structure_sha256": hashlib.sha256(Path(structure_path).read_bytes()).hexdigest(),
        "torch_total_energy_eV": reference.total_energy.item(),
        "gpumd_total_energy_eV": float(energy),
        "energy_difference_eV": reference.total_energy.item() - energy,
        "energy_difference_eV_per_atom": (
            reference.total_energy.item() - energy
        )
        / len(atoms),
        "force_max_error_eV_per_A": float(
            np.max(abs(reference.forces.detach().numpy() - forces))
        ),
        "position_raw_max_change_A": float(np.max(abs(displacement))),
        "position_minimum_image_max_change_A": float(np.max(abs(minimum_image))),
    }
    (case_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(metrics)
    return reference, energy, forces, metrics


def test_imported_qnep_has_consistent_total_force():
    model = load_gpumd_reference(MODEL)
    atoms = read(STRUCTURE)
    reference = model(QNEPStructure.from_atoms(atoms, model))
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions[0, 0] += 1e-5
    minus.positions[0, 0] -= 1e-5
    ep = model(QNEPStructure.from_atoms(plus, model)).total_energy.item()
    em = model(QNEPStructure.from_atoms(minus, model)).total_energy.item()
    assert reference.forces[0, 0].item() == pytest.approx(-(ep - em) / 2e-5, abs=2e-6)
    assert reference.electrostatic_energy.item() > 0


@pytest.mark.skipif(not os.environ.get("GPUMD_QNEP_BINARY"), reason="requires GPUMD")
def test_qnep_matches_gpumd_ewald(tmp_path):
    evidence = os.environ.get("QNEP_EVIDENCE")
    case_dir = Path(evidence) / "task-8-parity" / "water" if evidence else tmp_path
    reference, energy, forces, metrics = run_gpumd_case(
        os.environ["GPUMD_QNEP_BINARY"], MODEL, STRUCTURE, case_dir
    )
    assert metrics["position_minimum_image_max_change_A"] <= 1e-8
    assert reference.total_energy.item() == pytest.approx(energy, abs=3e-4, rel=0)
    np.testing.assert_allclose(
        reference.forces.detach().numpy(), forces, atol=2e-4, rtol=2e-4
    )


@pytest.mark.skipif(not os.environ.get("GPUMD_QNEP_BINARY"), reason="requires GPUMD")
def test_network_only_batio3_matches_gpumd_ewald(tmp_path):
    evidence = os.environ.get("QNEP_EVIDENCE")
    case_dir = Path(evidence) / "task-8-parity" / "batio3" if evidence else tmp_path
    case_dir.mkdir(parents=True, exist_ok=True)
    derived = derive_network_only_model(BATIO3_MODEL, case_dir / "network-only.nep.txt")
    reference, energy, forces, metrics = run_gpumd_case(
        os.environ["GPUMD_QNEP_BINARY"], derived, BATIO3_STRUCTURE, case_dir
    )
    assert metrics["position_minimum_image_max_change_A"] <= 1e-8
    assert abs(reference.total_energy.item() - energy) / len(forces) <= 1e-5
    np.testing.assert_allclose(
        reference.forces.detach().numpy(), forces, atol=2e-4, rtol=2e-4
    )


def test_original_zbl_batio3_model_is_rejected():
    with pytest.raises(ValueError):
        load_gpumd_reference(BATIO3_MODEL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_reference_and_force_loss_match_cpu():
    from test_qnep_reference import system

    cpu, atoms, structure = system()
    reference = cpu(structure, create_graph=True)
    gpu = cpu.cuda()
    result = gpu(QNEPStructure.from_atoms(atoms, gpu), create_graph=True)
    torch.testing.assert_close(
        result.forces.cpu(), reference.forces, atol=1e-10, rtol=1e-8
    )
    result.forces.square().mean().backward()
    assert all(torch.isfinite(weight.grad).all() for weight in gpu.charge_weights)
