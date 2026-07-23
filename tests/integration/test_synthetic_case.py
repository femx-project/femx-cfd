from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import h5py
import numpy as np
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
        "mesh/mesh.msh",
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
        assert "mesh" in h5
        assert "data" in h5
        assert len(h5["data"]) == 4
    with h5py.File(output / "reference/flow_full.h5") as h5:
        assert len(h5["data"]) > 4
    with h5py.File(output / "reference/flow_window.h5") as h5:
        assert len(h5["data"]) > 4
    observation_xdmf = (output / "reference/flow_observation.xdmf").read_text(
        encoding="utf-8"
    )
    assert "flow_observation.h5:/data/step_000000/velocity" in observation_xdmf
    assert "flow_full.h5:/data/step_000000/velocity" not in observation_xdmf
    with h5py.File(output / "inlet/initial_velocity.h5") as h5:
        assert h5["inlet/velocity"].shape[-1] == 2
        assert "normal" not in h5["inlet"]
    assert not (output / "var" / "case.yaml").exists()
    assert not (output / "envar" / "case.yaml").exists()
    phases = [event["phase"] for event in events if event["type"] == "solve"]
    assert "forward" in phases

    with pytest.raises(FileExistsError, match="already exist"):
        build_from_file(CONFIG, output)


def test_builds_synthetic_case_from_existing_mesh(tmp_path: Path) -> None:
    source = tmp_path / "source.msh"
    shutil.copyfile(CONFIG.parent / "mesh" / "mesh.msh", source)
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["mesh"] = {"path": "source.msh", "length_unit": "m"}
    config = _write_config(tmp_path / "synthetic.yaml", document)

    output = build_from_file(config, tmp_path / "from-file")

    assert source.is_file()
    assert not (output / "mesh").exists()
    assert (output / "observation/velocity.h5").is_file()


def test_reuses_existing_condition_local_mesh(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    mesh_path = output / "mesh" / "mesh.msh"
    mesh_path.parent.mkdir(parents=True)
    shutil.copyfile(CONFIG.parent / "mesh" / "mesh.msh", mesh_path)
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["mesh"] = {"path": "mesh/mesh.msh", "length_unit": "m"}
    config = _write_config(output / "synthetic.yaml", document)
    modified = mesh_path.stat().st_mtime_ns

    result = build_from_file(config)

    assert result == output
    assert mesh_path.stat().st_mtime_ns == modified
    assert (output / "observation/velocity.h5").is_file()


def test_builds_poisson_fourier_inlet_on_solver_time_grid(tmp_path: Path) -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["time"]["dt"] = 0.125
    document["reconstruction"] = {
        "kind": "poisson_fourier",
        "normal": "auto",
        "depth": 0.5,
        "fourier_order": 1,
    }

    output = build_from_file(
        _write_config(tmp_path / "synthetic.yaml", document),
        tmp_path / "poisson-fourier",
    )

    with h5py.File(output / "inlet/initial_velocity.h5") as h5:
        times = h5["inlet/times"][...]
        velocity = h5["inlet/velocity"][...]
    np.testing.assert_allclose(times, np.linspace(0.0, 1.0, 9))
    np.testing.assert_allclose(velocity[-1], velocity[0])
    np.testing.assert_allclose(velocity[..., 1], 0.0, atol=1.0e-14)
    singular_values = np.linalg.svd(velocity[..., 0], compute_uv=False)
    assert singular_values[1] / singular_values[0] < 1.0e-12


def test_uses_configured_synthetic_output_directory(tmp_path: Path) -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["output"]["directory"] = "configured-output"
    config = _write_config(tmp_path / "synthetic.yaml", document)

    output = build_from_file(config)

    assert output == tmp_path / "configured-output"
    assert (output / "observation/velocity.h5").is_file()


def test_builds_one_selected_phase_and_constant_vector_inlet(tmp_path: Path) -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["observations"]["counts"] = [2, 5]
    document["observations"]["selected_phases"] = [2]
    document["observations"]["noise"] = {
        "snr": 20.0,
        "reference": "max",
        "seed": 42,
    }
    config = tmp_path / "synthetic.yaml"
    config.write_text(yaml.safe_dump(document), encoding="utf-8")

    output = build_from_file(config, tmp_path / "selected")

    with h5py.File(output / "observation/velocity.h5") as h5:
        assert h5["velocity/points"].shape == (10, 2)
        assert h5["velocity/times"][...].tolist() == [0.25]
        assert h5["velocity/values"].shape == (1, 10, 2)
        attrs = h5["velocity"].attrs
        assert attrs["noise_model"] == "independent_gaussian"
        assert attrs["noise_specification"] == "snr"
        assert attrs["noise_seed"] == 42
        assert attrs["noise_snr"] == 20.0
        assert attrs["noise_reference"] == "max"
        assert attrs["noise_std"] == attrs["signal_scale"] / 20.0
    pvd = (output / "observation/velocity.pvd").read_text(encoding="utf-8")
    assert 'timestep="0.25"' in pvd
    assert (output / "observation/velocity/velocity_00000.vtu").is_file()
    with h5py.File(output / "inlet/initial_velocity.h5") as h5:
        assert h5["inlet/times"][...].tolist() == [0.0, 1.0]
        assert h5["inlet/velocity"].shape[-1] == 2
        assert "normal" not in h5["inlet"]


def test_masks_observation_lattice_to_fluid_mesh(tmp_path: Path) -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["observations"].update(
        {
            "lower": [-2.5, -0.4],
            "upper": [2.5, 0.4],
            "counts": [11, 3],
            "mask": "mesh",
        }
    )

    output = build_from_file(
        _write_config(tmp_path / "synthetic.yaml", document),
        tmp_path / "masked",
    )

    with h5py.File(output / "observation/velocity.h5") as h5:
        assert h5["velocity/points"].shape == (27, 2)
        assert h5["velocity/values"].shape == (4, 27, 2)


def test_thins_reference_flows_but_samples_observations_from_full_truth(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["output"]["every"] = 2

    output = build_from_file(
        _write_config(tmp_path / "synthetic.yaml", document),
        tmp_path / "thinned",
    )

    with h5py.File(output / "reference/flow_full.h5") as h5:
        assert len(h5["data"]) == 5
    with h5py.File(output / "reference/flow_window.h5") as h5:
        assert len(h5["data"]) == 3
    with h5py.File(output / "reference/flow_observation.h5") as h5:
        assert len(h5["data"]) == 4

    full_times = [
        float(time.attrib["Value"])
        for time in ET.parse(output / "reference/flow_full.xdmf").iter("Time")
    ]
    window_times = [
        float(time.attrib["Value"])
        for time in ET.parse(output / "reference/flow_window.xdmf").iter("Time")
    ]
    observation_times = [
        float(time.attrib["Value"])
        for time in ET.parse(output / "reference/flow_observation.xdmf").iter("Time")
    ]
    assert full_times == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert window_times == [0.0, 0.5, 1.0]
    assert observation_times == [0.0, 0.25, 0.5, 0.75]


def _write_config(path: Path, document: dict) -> Path:
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path
