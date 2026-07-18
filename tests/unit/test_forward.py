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
from femx_cfd.workflows.forward import solve_forward


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
            {"boundary": "wall", "kind": "no_slip"},
            {"boundary": "outlet", "kind": "pressure", "value": 0.0},
        ],
        "solver": {"backend": "dense"},
    }


class ForwardCaseTest(unittest.TestCase):
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
            "backend": "resolve-cuda",
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
                return_value=("resolve-cuda",),
            ),
        ):
            solution = solve_forward(case)

        problem.create_solver.assert_called_once_with(
            "resolve-cuda",
            options={"solve": "fgmres", "rtol": 1.0e-9},
        )
        self.assertIs(solution.trajectory, mock.sentinel.trajectory)


if __name__ == "__main__":
    unittest.main()
