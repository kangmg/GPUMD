from pathlib import Path

import numpy as np
import pytest

from torchnep.data import parse_nep_in
from torchnep.train import train_nep


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ("early_stop_patience 3\n", 3),
        ("early_stop_patience 0\n", 0),
        ("early_stop 5\nearly_stop_patience 3\n", 5),
        ("early_stop_patience 3\nearly_stop 5\n", 5),
        ("early_stop 0\nearly_stop_patience 3\n", 0),
        ("", 0),
    ],
)
def test_existing_early_stop_inputs(
    tmp_path: Path, settings: str, expected: int
) -> None:
    config_path = tmp_path / "nep.in"
    config_path.write_text("type 1 H\n" + settings)

    config = parse_nep_in(str(config_path))

    assert config["early_stop"] == expected


def test_legacy_patience_stops_training(tmp_path: Path) -> None:
    config_path = tmp_path / "nep.in"
    config_path.write_text(
        "type 1 H\ncutoff 3 2\nn_max 1 1\nbasis_size 1 1\n"
        "l_max 1 0 0\nneuron 4\nepoch 20\nbatch 1\nlr 0\n"
        "lambda_e 0\nlambda_f 1\nlambda_v 0\nearly_stop_patience 2\n"
    )
    data_path = tmp_path / "train.xyz"
    data_path.write_text(
        '2\nLattice="10 0 0 0 10 0 0 0 10" energy=0 '
        "Properties=species:S:1:pos:R:3:force:R:3\n"
        "H 0 0 0 0 0 0\nH 1 0 0 0 0 0\n"
    )
    output = tmp_path / "out"

    train_nep(
        config_file=str(config_path),
        data_file=str(data_path),
        output_dir=str(output),
        device="cpu",
        use_compile=False,
        run_seed=0,
        restart=False,
    )

    losses = np.loadtxt(output / "loss.out", ndmin=2)
    assert 2 <= losses[-1, 0] < 20
    assert np.isfinite(losses).all()
    assert (output / "nep_best.txt").is_file()
    assert (output / "nep_final.txt").is_file()
    assert (output / "checkpoint.pt").is_file()
