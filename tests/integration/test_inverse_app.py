"""End-to-end tests for portable Var and EnVar case execution."""

import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import h5py
import pytest
import yaml

from femx_cfd.data import load_inlet
from femx_cfd.runtime.runner import run_case


EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "inverse" / "straight_tube"
LOWER_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


@pytest.mark.parametrize("method", ("var", "envar"))
def test_runs_portable_inverse_case(method: str, tmp_path: Path) -> None:
    output = run_case(
        EXAMPLE / method / "case.yaml",
        tmp_path / method,
        progress_kind="console",
    )

    status = json.loads((output / "status.json").read_text())
    manifest = json.loads((output / "result.json").read_text())
    assert status["format"] == "femx-cfd-inverse-status"
    assert status["state"] == "complete"
    assert manifest["format"] == "femx-cfd-inverse-result"
    assert manifest["version"] == 3
    assert manifest["method"] == method
    assert (output / manifest["flow_window"]).is_file()
    assert (output / manifest["flow_observation"]).is_file()
    assert (output / manifest["observation"]).is_file()
    assert (output / manifest["initial_inlet"]).is_file()
    assert "ensemble_initial_states" not in manifest
    with h5py.File(output / manifest["data"]) as result:
        names = []
        result.visit(names.append)
        attribute_names = list(result.attrs)
        result.visititems(lambda _name, item: attribute_names.extend(item.attrs.keys()))
        assert all(
            LOWER_SNAKE_CASE.fullmatch(component)
            for name in names
            for component in name.split("/")
        )
        assert all(LOWER_SNAKE_CASE.fullmatch(name) for name in attribute_names)
        assert result.attrs["format_version"] == 3
        assert result.attrs["kind"] == f"{method}_result"
        assert "estimate/metrics" in result
        assert "estimate/fields" in result
        assert "mesh" in result
        assert "Data" not in result
        assert list(result["mesh"]) == ["geometry", "topology"]
        assert list(result["estimate/fields"]) == [
            "step_000000",
            "step_000001",
            "step_000002",
            "step_000003",
            "step_000004",
        ]
        assert list(result["estimate/observation_fields"]) == [
            "step_000000",
            "step_000001",
            "step_000002",
            "step_000003",
        ]
        if method == "envar":
            estimate = result["estimate"]
            assert estimate["parameters"].shape == (28,)
            assert estimate["coefficients"].shape == (4,)
            assert estimate["parameter_history"].shape == (2, 28)
            assert estimate["coefficient_history"].shape == (2, 4)
            assert estimate["coefficient_covariance"].shape == (4, 4)
            names = estimate["coefficient_names"].asstr()[...].tolist()
            assert names == [
                "flow_mean_x",
                "flow_mean_y",
                "flow_sin1_x",
                "flow_sin1_y",
            ]
            assert estimate["metrics"].attrs["outer_iterations"] == 2
            assert estimate["metrics"].attrs[
                "nonlinear_observation_misfit_history"
            ].shape == (2,)
            assert list(result["ensemble_basis"]) == ["outer_001"]
            outer = result["ensemble_basis/outer_001"]
            assert outer["coefficient_names"].asstr()[...].tolist() == names
            assert outer["state_times"][0] == 0.0
            assert outer["reference/fields/velocity"].shape == (5, 104, 3)
            assert outer["reference/fields/pressure"].shape == (5, 104)
            assert outer["perturbations/mode_000/fields/velocity"].shape == (5, 104, 3)
        else:
            assert result["estimate/parameters"].shape == (24,)
            assert "ensemble_basis" not in result
        assert "EnsembleInitial" not in result
        assert "SampleData" not in result
    if method == "envar":
        assert manifest["ensemble_basis"] == [
            "estimate/ensemble_basis_outer_001.xdmf",
        ]
        basis_view = (output / manifest["ensemble_basis"][0]).read_text(
            encoding="utf-8"
        )
        assert 'Function="$0 + $1"' in basis_view
        assert (
            "result.h5:/ensemble_basis/outer_001/perturbations/mode_000/fields/velocity"
        ) in basis_view
    else:
        assert "ensemble_basis" not in manifest
    phase_times = [
        float(element.attrib["Value"])
        for element in ET.parse(output / manifest["flow_observation"]).findall(
            "./Domain/Grid/Grid/Time"
        )
    ]
    with h5py.File(EXAMPLE / "observation" / "velocity.h5") as observation:
        assert phase_times == pytest.approx(observation["velocity/times"][...])
    pvd = (output / manifest["observation"]).read_text(encoding="utf-8")
    assert 'file="velocity/velocity_00000.vtu"' in pvd


def test_can_save_all_envar_outer_bases(tmp_path: Path) -> None:
    document = yaml.safe_load((EXAMPLE / "envar" / "case.yaml").read_text())
    document["mesh"]["path"] = str(EXAMPLE / "mesh" / "mesh.msh")
    document["observations"]["path"] = str(EXAMPLE / "observation" / "velocity.h5")
    document["initialization"]["inlet"]["path"] = str(
        EXAMPLE / "inlet" / "initial_velocity.h5"
    )
    document["output"]["ensemble_basis"] = "all"
    case_path = tmp_path / "all_outer_bases.yaml"
    case_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    output = run_case(case_path, tmp_path / "all_outer_bases", progress_kind="console")

    manifest = json.loads((output / "result.json").read_text())
    assert manifest["ensemble_basis"] == [
        "estimate/ensemble_basis_outer_000.xdmf",
        "estimate/ensemble_basis_outer_001.xdmf",
    ]
    with h5py.File(output / manifest["data"]) as result:
        assert list(result["ensemble_basis"]) == ["outer_000", "outer_001"]


def test_derives_initial_inlet_from_observations(tmp_path: Path) -> None:
    document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
    document["mesh"]["path"] = str(EXAMPLE / "mesh" / "mesh.msh")
    document["observations"]["path"] = str(EXAMPLE / "observation" / "velocity.h5")
    document["initialization"]["inlet"] = {
        "source": "observations",
        "projection": {
            "depth": 0.5,
            "axial_decay": 0.25,
            "laplacian_modes": 3,
            "ridge": 0.001,
        },
    }
    case_path = tmp_path / "case.yaml"
    case_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    output = run_case(case_path, tmp_path / "result", progress_kind="console")

    inlet = load_inlet(output / "inlet" / "initial_velocity.h5")
    assert inlet.source == "velocity"
    assert inlet.boundary == "inlet"
    assert inlet.vals.ndim == 3
    with h5py.File(output / "inlet" / "initial_velocity.h5") as h5:
        assert h5.attrs["format_version"] == 2
        assert "velocity" in h5["inlet"]
        assert "normal" not in h5["inlet"]


def test_runs_optional_normal_control(tmp_path: Path) -> None:
    document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
    document["mesh"]["path"] = str(EXAMPLE / "mesh" / "mesh.msh")
    document["observations"]["path"] = str(EXAMPLE / "observation" / "velocity.h5")
    document["initialization"]["inlet"]["path"] = str(
        EXAMPLE / "inlet" / "initial_velocity.h5"
    )
    document["control"] = {
        "boundary": "inlet",
        "kind": "normal",
        "normal": "auto",
    }
    document["algorithm"]["control_bounds"] = [0.0, 0.2]
    case_path = tmp_path / "normal.yaml"
    case_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    output = run_case(case_path, tmp_path / "normal", progress_kind="console")

    with h5py.File(output / "estimate" / "result.h5") as result:
        assert result["estimate/parameters"].shape == (12,)
