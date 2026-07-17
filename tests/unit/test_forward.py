import shutil
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from femx_cfd.forward import (
    CASE_FORMAT,
    CASE_VERSION,
    load_forward_case,
    validate_forward_case,
)


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
            OmegaConf.save(OmegaConf.create(make_case()), case_path)

            case = load_forward_case(case_path)

        self.assertEqual(case["mesh"], str((directory / "mesh.msh").resolve()))
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


if __name__ == "__main__":
    unittest.main()
