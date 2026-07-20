from pathlib import Path

import h5py
import pytest
import yaml

from tools.synthetic.case import build_from_file


CONFIG = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "inverse"
    / "straight_tube"
    / "synthetic.yaml"
)


def test_builds_complete_synthetic_case_from_yaml(tmp_path: Path) -> None:
    events = []
    output = build_from_file(CONFIG, tmp_path / "generated", progress=events.append)

    for name in (
        "mesh.msh",
        "reference/flow_full.h5",
        "reference/flow_full.xdmf",
        "reference/flow_window.h5",
        "reference/flow_window.xdmf",
        "reference/flow_observation.h5",
        "reference/flow_observation.xdmf",
        "observation/velocity.h5",
        "observation/velocity.pvd",
        "observation/velocity/velocity_00000.vtu",
        "inlet/initial_velocity.h5",
    ):
        assert (output / name).is_file()
    with h5py.File(output / "reference/flow_observation.h5") as h5:
        assert "Mesh" in h5
        assert "Data" in h5
        assert len(h5["Data"]) == 4
    with h5py.File(output / "reference/flow_full.h5") as h5:
        assert len(h5["Data"]) > 4
    with h5py.File(output / "reference/flow_window.h5") as h5:
        assert len(h5["Data"]) > 4
    observation_xdmf = (output / "reference/flow_observation.xdmf").read_text(
        encoding="utf-8"
    )
    assert "flow_observation.h5:/Data/Step00000/velocity" in observation_xdmf
    assert "flow_full.h5:/Data/Step00000/velocity" not in observation_xdmf
    with h5py.File(output / "inlet/initial_velocity.h5") as h5:
        assert h5["inlet/velocity"].shape[-1] == 2
        assert "normal" not in h5["inlet"]
    assert not (output / "var" / "case.yaml").exists()
    assert not (output / "envar" / "case.yaml").exists()
    phases = [event["phase"] for event in events if event["type"] == "solve"]
    assert "forward" in phases

    with pytest.raises(FileExistsError, match="already exist"):
        build_from_file(CONFIG, output)


def test_builds_one_selected_phase_and_constant_vector_inlet(tmp_path: Path) -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["observations"]["counts"] = [2, 5]
    document["observations"]["selected_phases"] = [2]
    config = tmp_path / "synthetic.yaml"
    config.write_text(yaml.safe_dump(document), encoding="utf-8")

    output = build_from_file(config, tmp_path / "selected")

    with h5py.File(output / "observation/velocity.h5") as h5:
        assert h5["velocity/points"].shape == (10, 2)
        assert h5["velocity/times"][...].tolist() == [0.25]
        assert h5["velocity/values"].shape == (1, 10, 2)
    pvd = (output / "observation/velocity.pvd").read_text(encoding="utf-8")
    assert 'timestep="0.25"' in pvd
    assert (output / "observation/velocity/velocity_00000.vtu").is_file()
    with h5py.File(output / "inlet/initial_velocity.h5") as h5:
        assert h5["inlet/times"][...].tolist() == [0.0, 1.0]
        assert h5["inlet/velocity"].shape[-1] == 2
        assert "normal" not in h5["inlet"]
