import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import yaml

from femx_cfd.cli import run_forward_cli as run_cli
from femx_cfd.forward import CASE_FORMAT, CASE_VERSION


MESH = Path(__file__).resolve().parents[1] / "data" / "2d_tiny_tube.msh"


class ForwardApplicationTest(unittest.TestCase):
    def test_runs_case_and_writes_machine_readable_result(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            shutil.copyfile(MESH, directory / "mesh.msh")
            case = {
                "format": CASE_FORMAT,
                "version": CASE_VERSION,
                "mesh": {"path": "mesh.msh", "length_unit": "mm"},
                "fluid": {"rho": 1.0, "mu": 0.01},
                "time": {"dt": 0.25, "end_time": 0.25},
                "inlet": {
                    "boundary": "inlet",
                    "normal": "auto",
                    "profile": {"kind": "uniform", "value": 0.1},
                },
                "boundary_conditions": [
                    {"boundary": "wall", "kind": "no_slip"},
                    {
                        "boundary": "outlet",
                        "kind": "pressure",
                        "value": 0.0,
                    },
                ],
                "solver": {"backend": "dense"},
            }
            case_path = directory / "case.yaml"
            case_path.write_text(yaml.safe_dump(case), encoding="utf-8")
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                code = run_cli([str(case_path), "--progress", "json"])

            output = directory / "result"
            self.assertEqual(code, 0)
            self.assertTrue((output / "flow.h5").is_file())
            self.assertTrue((output / "flow.xdmf").is_file())
            with h5py.File(output / "flow.h5", "r") as h5:
                self.assertEqual(len(h5["Data"]), 2)
                self.assertAlmostEqual(
                    float(abs(h5["Mesh/Geometry"][...]).max()),
                    1.0e-3,
                )

            status = json.loads((output / "status.json").read_text())
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(status["state"], "complete")
            self.assertEqual(result["flow"], "flow.xdmf")

            events = [json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual(events[0]["type"], "stage")
            self.assertEqual(events[-1]["type"], "stage")
            self.assertIn("complete", events[-1]["message"])

    def test_does_not_overwrite_an_existing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            shutil.copyfile(MESH, directory / "mesh.msh")
            case_path = directory / "case.yaml"
            case_path.write_text(
                yaml.safe_dump(
                    {
                        "format": CASE_FORMAT,
                        "version": CASE_VERSION,
                        "mesh": {"path": "mesh.msh", "length_unit": "m"},
                        "fluid": {"rho": 1.0, "mu": 0.01},
                        "time": {"dt": 0.25, "end_time": 0.25},
                        "inlet": {
                            "boundary": "inlet",
                            "profile": {"kind": "uniform", "value": 0.1},
                        },
                        "boundary_conditions": [
                            {"boundary": "wall", "kind": "no_slip"},
                            {"boundary": "outlet", "kind": "pressure"},
                        ],
                        "solver": {"backend": "dense"},
                    },
                ),
                encoding="utf-8",
            )
            output = directory / "result"
            output.mkdir()
            status = output / "status.json"
            status.write_text('{"state": "complete"}\n')

            with contextlib.redirect_stderr(io.StringIO()):
                code = run_cli([str(case_path)])

            self.assertEqual(code, 1)
            self.assertEqual(status.read_text(), '{"state": "complete"}\n')


if __name__ == "__main__":
    unittest.main()
