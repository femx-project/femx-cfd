from pathlib import Path

import h5py
import pytest

from tools.synthetic.case import build_from_file


CONFIG = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "inverse"
    / "straight_tube"
    / "synthetic.yaml"
)


def test_builds_complete_synthetic_case_from_yaml(tmp_path: Path) -> None:
    output = build_from_file(CONFIG, tmp_path / "generated")

    for name in (
        "mesh.msh",
        "observations.h5",
        "initial_inlet.h5",
    ):
        assert (output / name).is_file()
    with h5py.File(output / "initial_inlet.h5") as h5:
        assert h5["inlet/velocity"].shape[-1] == 2
        assert "normal" not in h5["inlet"]
    assert not (output / "var" / "case.yaml").exists()
    assert not (output / "envar" / "case.yaml").exists()

    with pytest.raises(FileExistsError, match="already exist"):
        build_from_file(CONFIG, output)
