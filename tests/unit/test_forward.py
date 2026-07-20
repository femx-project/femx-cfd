import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import yaml

from femx_cfd import (
    CASE_FORMAT,
    CASE_VERSION,
    load_forward_case,
    validate_forward_case,
)
from femx_cfd.cases import load_case
from femx_cfd.cli import run_cli
from femx_cfd.data import InletVelocityData, save_inlet
from femx_cfd.workflows.forward import _file_inlet, solve_forward


MESH = Path(__file__).resolve().parents[1] / "data" / "2d_tiny_tube.msh"


def make_case(mesh="mesh.msh"):
    return {
        "format": CASE_FORMAT,
        "version": CASE_VERSION,
        "mesh": str(mesh),
        "fluid": {"rho": 1.0, "mu": 0.01},
        "time": {"dt": 0.25, "end_time": 0.25},
        "inlet": {
            "boundary": "inlet",
            "normal": "auto",
            "profile": {"kind": "uniform", "value": 0.1},
        },
        "boundary_conditions": [
            {"boundary": "wall", "kind": "velocity", "value": [0.0, 0.0]},
            {"boundary": "outlet", "kind": "pressure", "value": 0.0},
        ],
        "solver": {"backend": "dense"},
    }


class ForwardCaseTest(unittest.TestCase):
    def test_migrates_version_1_no_slip_to_explicit_velocity(self):
        case = make_case(MESH)
        case["version"] = 1
        case["boundary_conditions"][0] = {
            "boundary": "wall",
            "kind": "no_slip",
        }
        case["boundary_conditions"][1].pop("value")

        validated = validate_forward_case(case)

        self.assertEqual(validated["version"], 2)
        self.assertEqual(
            validated["boundary_conditions"][0],
            {
                "boundary": "wall",
                "kind": "velocity",
                "value": [0.0, 0.0],
            },
        )
        self.assertEqual(case["version"], 1)
        self.assertEqual(case["boundary_conditions"][0]["kind"], "no_slip")
        self.assertNotIn("value", case["boundary_conditions"][1])
        self.assertEqual(validated["boundary_conditions"][1]["value"], 0.0)

    def test_reports_source_and_migrated_versions(self):
        case = make_case(MESH)
        case["version"] = 1
        case["boundary_conditions"][0] = {
            "boundary": "wall",
            "kind": "no_slip",
        }
        with tempfile.TemporaryDirectory() as directory:
            case_path = Path(directory) / "case.yaml"
            case_path.write_text(yaml.safe_dump(case), encoding="utf-8")

            loaded = load_case(case_path)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = run_cli(["validate", str(case_path)])

        self.assertEqual(loaded.source_version, 1)
        self.assertEqual(loaded.version, 2)
        self.assertEqual(code, 0)
        self.assertEqual(
            stdout.getvalue().strip(),
            "valid: femx-cfd-forward version 1 (migrates to version 2)",
        )

    def test_rejects_no_slip_in_version_2(self):
        case = make_case(MESH)
        case["boundary_conditions"][0] = {
            "boundary": "wall",
            "kind": "no_slip",
        }

        with self.assertRaisesRegex(ValueError, "velocity.*pressure"):
            validate_forward_case(case)

    def test_rejects_invalid_velocity_value(self):
        case = make_case(MESH)
        case["boundary_conditions"][0]["value"] = [0.0]

        with self.assertRaisesRegex(ValueError, "2 or 3 finite values"):
            validate_forward_case(case)

    def test_requires_explicit_pressure_value_in_version_2(self):
        case = make_case(MESH)
        case["boundary_conditions"][1].pop("value")

        with self.assertRaisesRegex(ValueError, "value must be finite"):
            validate_forward_case(case)

    def test_loads_mesh_relative_to_case_file_and_applies_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            shutil.copyfile(MESH, directory / "mesh.msh")
            case_path = directory / "case.yaml"
            case_path.write_text(
                yaml.safe_dump(make_case()),
                encoding="utf-8",
            )

            case = load_forward_case(case_path)

        self.assertEqual(
            case["mesh"],
            {
                "path": str((directory / "mesh.msh").resolve()),
                "length_unit": "m",
            },
        )
        self.assertEqual(case["inlet"]["period"], 0.25)
        self.assertEqual(
            case["inlet"]["waveform"],
            {"kind": "uniform", "value": 1.0},
        )
        self.assertEqual(case["inlet"]["source"], "analytic")

    def test_loads_prepared_inlet_relative_to_case_file(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            shutil.copyfile(MESH, directory / "mesh.msh")
            inlet_path = directory / "inlet.h5"
            inlet_path.touch()
            document = make_case()
            document["inlet"] = {
                "source": "file",
                "boundary": "inlet",
                "path": "inlet.h5",
            }
            case_path = directory / "case.yaml"
            case_path.write_text(yaml.safe_dump(document), encoding="utf-8")

            case = load_forward_case(case_path)

        self.assertEqual(case["inlet"]["source"], "file")
        self.assertEqual(case["inlet"]["path"], str(inlet_path.resolve()))

    def test_rejects_unknown_inlet_source(self):
        case = make_case(MESH)
        case["inlet"]["source"] = "measurement"

        with self.assertRaisesRegex(ValueError, "analytic.*file"):
            validate_forward_case(case)

    def test_rejects_nonintegral_time_step_count(self):
        case = make_case(MESH)
        case["time"]["end_time"] = 0.3

        with self.assertRaisesRegex(ValueError, "integer multiple"):
            validate_forward_case(case)

    def test_rejects_duplicate_inlet_boundary_condition(self):
        case = make_case(MESH)
        case["boundary_conditions"][0]["boundary"] = "inlet"

        with self.assertRaisesRegex(ValueError, "inlet"):
            validate_forward_case(case)

    def test_accepts_explicit_mesh_length_unit(self):
        case = make_case(MESH)
        case["mesh"] = {"path": str(MESH), "length_unit": "mm"}

        validated = validate_forward_case(case)

        self.assertEqual(validated["mesh"]["length_unit"], "mm")

    def test_passes_named_options_to_the_selected_solver(self):
        case = make_case(MESH)
        case["solver"] = {
            "backend": "resolve",
            "options": {"solve": "fgmres", "rtol": 1.0e-9},
        }
        problem = mock.Mock()
        solver = problem.create_solver.return_value
        solver.solve.return_value = mock.sentinel.trajectory

        with (
            mock.patch(
                "femx_cfd.workflows.forward._make_forward_problem",
                return_value=(problem, np.asarray([1.0])),
            ),
            mock.patch(
                "femx_cfd.workflows.forward.femx.solver_backends",
                return_value=("resolve",),
            ),
        ):
            solution = solve_forward(case)

        problem.create_solver.assert_called_once_with(
            "resolve",
            options={"solve": "fgmres", "rtol": 1.0e-9},
        )
        self.assertIs(solution.trajectory, mock.sentinel.trajectory)

    def test_repeats_periodic_prepared_inlet_for_forward_interval(self):
        inlet = InletVelocityData(
            boundary="inlet",
            times=np.asarray([0.0, 0.25, 0.5]),
            node_ids=np.asarray([4, 5]),
            coords=np.asarray([[0.0, -0.5], [0.0, 0.5]]),
            vals=np.asarray(
                [
                    [[1.0, 4.0], [2.0, 5.0]],
                    [[1.5, 6.0], [2.5, 7.0]],
                    [[1.0, 4.0], [2.0, 5.0]],
                ]
            ),
            periodic=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inlet.h5"
            save_inlet(path, inlet)

            prepared = _file_inlet(
                {
                    "time": {"dt": 0.25, "end_time": 1.0},
                    "inlet": {
                        "source": "file",
                        "boundary": "inlet",
                        "path": str(path),
                    },
                },
                np.asarray([1.0, 0.0]),
            )

        np.testing.assert_allclose(prepared.times, [0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(
            prepared.vals,
            [[1.5, 2.5], [1.0, 2.0], [1.5, 2.5], [1.0, 2.0]],
        )
        self.assertFalse(prepared.periodic)

    def test_rejects_prepared_inlet_for_another_boundary(self):
        inlet = InletVelocityData(
            boundary="other",
            times=np.asarray([0.0, 0.25]),
            node_ids=np.asarray([4]),
            coords=np.asarray([[0.0, 0.0]]),
            vals=np.asarray([[[1.0, 0.0]], [[1.0, 0.0]]]),
            periodic=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inlet.h5"
            save_inlet(path, inlet)

            with self.assertRaisesRegex(ValueError, "boundary"):
                _file_inlet(
                    {
                        "time": {"dt": 0.25, "end_time": 0.25},
                        "inlet": {
                            "source": "file",
                            "boundary": "inlet",
                            "path": str(path),
                        },
                    },
                    np.asarray([1.0, 0.0]),
                )


if __name__ == "__main__":
    unittest.main()
