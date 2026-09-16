from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from torchnep.qnep.data_io import load_records
from torchnep.qnep.errors import QNEPError
from torchnep.qnep.model import QNEPConfig, QNEPModel
from torchnep.qnep.records import QNEPRecord, validate_datasets
from torchnep.qnep.run_config import QNEPDatasets, QNEPRunConfig
from torchnep.qnep.structure import QNEPStructure
from torchnep.qnep.training import TrainingSample


def _model() -> QNEPModel:
    return QNEPModel(QNEPConfig(type_names=("H",))).double()


def _record(
    *,
    x: float = 0.0,
    energy: float | None = 0.0,
    weight: float = 1.0,
    masked_force: float = float("nan"),
    record_id: str | None = None,
    group_id: str | None = None,
) -> QNEPRecord:
    structure = QNEPStructure(
        positions=torch.tensor([[x, 0.0, 0.0]], dtype=torch.float64),
        cell=torch.eye(3, dtype=torch.float64) * 8.0,
        atom_types=torch.tensor([0]),
    )
    forces = torch.tensor([[1.0, masked_force, 3.0]], dtype=torch.float64)
    mask = torch.tensor([[True, False, True]])
    return QNEPRecord(
        TrainingSample(structure, energy, forces, mask, weight),
        record_id,
        group_id,
    )


def _config(tmp_path: Path, **changes: float | str | Path | None) -> QNEPRunConfig:
    values: dict[str, float | str | Path | None] = {"output_dir": tmp_path / "run"}
    values.update(changes)
    return QNEPRunConfig(**values)


def test_roundtrip_preserves_zero_masks_weights_and_opaque_ids(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "mixed.xyz"
    path.write_text(
        "1\n"
        'Lattice="8 0 0 0 8 0 0 0 8" pbc="T T T" energy=0 '
        "sample_weight=2 record_id=row-0 group_id=traj-a "
        "Properties=species:S:1:pos:R:3:forces:R:3:force_mask:I:3\n"
        "H 0 0 0 1 nan 3 1 0 1\n"
        "1\n"
        'Lattice="8 0 0 0 8 0 0 0 8" pbc="T T T" '
        "sample_weight=3 Properties=species:S:1:pos:R:3:forces:R:3\n"
        "H 1 0 0 4 5 6\n",
        encoding="utf-8",
    )

    # When
    records = load_records(path, _model())
    copy = tmp_path / "renamed.xyz"
    shutil.copyfile(path, copy)
    copied = load_records(copy, _model())

    # Then
    assert records[0].sample.energy == 0.0
    assert records[0].sample.weight == 2.0
    assert records[0].sample.force_mask is not None
    assert records[0].sample.force_mask.tolist() == [[True, False, True]]
    assert torch.isnan(records[0].sample.forces[0, 1])
    assert records[0].record_id == "row-0"
    assert records[0].group_id == "traj-a"
    assert records[1].sample.energy is None
    assert records[1].sample.weight == 3.0
    config = _config(tmp_path)
    assert validate_datasets(
        QNEPDatasets(records, (_record(x=9.0),)), _model(), config
    ) == validate_datasets(QNEPDatasets(copied, (_record(x=9.0),)), _model(), config)


def test_fingerprint_is_filename_independent_and_ignores_masked_force_bytes(
    tmp_path: Path,
) -> None:
    # Given
    model = _model()
    config = _config(tmp_path)
    first = _record(x=-0.0, masked_force=float("nan"))
    second = _record(masked_force=987.0)

    # When
    first_ids = validate_datasets(QNEPDatasets((first,), (_record(x=1.0),)), model, config)
    second_ids = validate_datasets(
        QNEPDatasets((second,), (_record(x=1.0),)), model, config
    )

    # Then
    assert first_ids == second_ids
    assert len(first_ids.train) == 64
    assert len(first_ids.validation) == 64
    assert len(first_ids.settings) == 64
    assert torch.signbit(first.sample.structure.positions[0, 0])


@pytest.mark.parametrize("change", ["label", "weight", "geometry", "order"])
def test_fingerprint_changes_with_curated_content(
    tmp_path: Path, change: str
) -> None:
    # Given
    model = _model()
    config = _config(tmp_path)
    train = (_record(x=0.0, energy=1.0), _record(x=1.0, energy=2.0))
    baseline = validate_datasets(
        QNEPDatasets(train, (_record(x=2.0),)), model, config
    ).train
    changed = {
        "label": (_record(x=0.0, energy=9.0), train[1]),
        "weight": (_record(x=0.0, energy=1.0, weight=2.0), train[1]),
        "geometry": (_record(x=0.5, energy=1.0), train[1]),
        "order": tuple(reversed(train)),
    }[change]

    # When
    fingerprint = validate_datasets(
        QNEPDatasets(changed, (_record(x=2.0),)), model, config
    ).train

    # Then
    assert fingerprint != baseline


def test_exact_geometry_leakage_ignores_label_differences(tmp_path: Path) -> None:
    # Given
    datasets = QNEPDatasets(
        (_record(energy=1.0),),
        (_record(energy=99.0, weight=7.0),),
    )

    # When
    with pytest.raises(QNEPError, match="geometry.*train.*validation"):
        validate_datasets(datasets, _model(), _config(tmp_path))


@pytest.mark.parametrize("field", ["record_id", "group_id"])
def test_explicit_identity_leakage_is_rejected(tmp_path: Path, field: str) -> None:
    # Given
    kwargs = {field: "shared"}
    datasets = QNEPDatasets(
        (_record(x=0.0, **kwargs),),
        (_record(x=1.0, **kwargs),),
    )

    # When
    with pytest.raises(QNEPError, match=field):
        validate_datasets(datasets, _model(), _config(tmp_path))


@pytest.mark.parametrize(
    ("mask", "weight", "error"),
    [
        ("1 0.5 1", 1.0, "force_mask"),
        ("1 0 1", 0.0, "weight"),
    ],
)
def test_invalid_extxyz_values_have_frame_context(
    tmp_path: Path, mask: str, weight: float, error: str
) -> None:
    # Given
    path = tmp_path / "bad.xyz"
    path.write_text(
        "1\n"
        f'Lattice="8 0 0 0 8 0 0 0 8" pbc="T T T" sample_weight={weight} '
        "Properties=species:S:1:pos:R:3:forces:R:3:force_mask:R:3\n"
        f"H 0 0 0 1 2 3 {mask}\n",
        encoding="utf-8",
    )

    # When
    with pytest.raises(QNEPError, match=rf"{path}.*frame 0.*{error}"):
        load_records(path, _model())


@pytest.mark.parametrize(
    ("metadata", "species", "error"),
    [("total_charge=1", "H", "neutral"), ("", "He", "unsupported element")],
)
def test_unsupported_charge_or_element_has_frame_context(
    tmp_path: Path, metadata: str, species: str, error: str
) -> None:
    # Given
    path = tmp_path / "unsupported.xyz"
    path.write_text(
        "1\n"
        f'Lattice="8 0 0 0 8 0 0 0 8" pbc="T T T" energy=1 {metadata} '
        "Properties=species:S:1:pos:R:3\n"
        f"{species} 0 0 0\n",
        encoding="utf-8",
    )

    # When
    with pytest.raises(QNEPError, match=rf"{path}.*frame 0.*{error}"):
        load_records(path, _model())


def test_dataset_and_config_boundaries_reject_inactive_or_invalid_input(
    tmp_path: Path,
) -> None:
    # Given
    inactive = _record(energy=None)
    inactive = QNEPRecord(
        inactive.sample._replace(force_mask=torch.zeros((1, 3), dtype=torch.bool))
    )

    # When
    with pytest.raises(QNEPError, match="active energy or force"):
        validate_datasets(
            QNEPDatasets((inactive,), (_record(x=1.0),)),
            _model(),
            _config(tmp_path),
        )
    wrong_dtype = _record()
    wrong_dtype = QNEPRecord(
        wrong_dtype.sample._replace(forces=wrong_dtype.sample.forces.float())
    )
    with pytest.raises(QNEPError, match="forces.*dtype"):
        validate_datasets(
            QNEPDatasets((wrong_dtype,), (_record(x=1.0),)),
            _model(),
            _config(tmp_path),
        )
    with pytest.raises(QNEPError, match="batch_size"):
        _config(tmp_path, batch_size=0)
    with pytest.raises(QNEPError, match="loss weights"):
        _config(tmp_path, energy_weight=float("nan"))
    with pytest.raises(QNEPError, match="precision"):
        _config(tmp_path, precision="float16")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "foreign.txt").write_text("foreign", encoding="utf-8")
    with pytest.raises(QNEPError, match="nonempty output_dir"):
        QNEPRunConfig(output_dir=occupied)
