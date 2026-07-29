import contextlib
import io
import json
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import yaml

from femx_cfd.cli import run_cli
from femx_cfd.cases.forward import CASE_FORMAT, CASE_VERSION
from femx_cfd.data import InletVelocityData, save_inlet


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
                "boundary_conditions": [
                    {
                        "boundary": "inlet",
                        "kind": "velocity",
                        "source": "analytic",
                        "normal": "auto",
                        "profile": {"kind": "uniform", "value": 0.1},
                    },
                    {
                        "boundary": "wall",
                        "kind": "velocity",
                        "source": "analytic",
                        "value": [0.0, 0.0],
                    },
                    {
                        "boundary": "outlet",
                        "kind": "pressure",
                        "source": "analytic",
                        "value": 0.0,
                    },
                ],
                "solver": {"backend": "dense"},
            }
            case_path = directory / "case.yaml"
            case_path.write_text(yaml.safe_dump(case), encoding="utf-8")
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                code = run_cli(["run", str(case_path), "--progress", "json"])

            output = directory / "result"
            self.assertEqual(code, 0)
            self.assertTrue((output / "flow.h5").is_file())
            self.assertTrue((output / "flow.xdmf").is_file())
            with h5py.File(output / "flow.h5", "r") as h5:
                self.assertEqual(len(h5["data"]), 2)
                self.assertAlmostEqual(
                    float(abs(h5["mesh/geometry"][...]).max()),
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

    def test_uses_configured_output_and_writes_every_n_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            shutil.copyfile(MESH, directory / "mesh.msh")
            case = {
                "format": CASE_FORMAT,
                "version": CASE_VERSION,
                "output": {"directory": "runs/thinned", "every": 2},
                "mesh": {"path": "mesh.msh", "length_unit": "mm"},
                "fluid": {"rho": 1.0, "mu": 0.01},
                "time": {"dt": 0.25, "end_time": 1.0},
                "boundary_conditions": [
                    {
                        "boundary": "inlet",
                        "kind": "velocity",
                        "source": "analytic",
                        "normal": "auto",
                        "profile": {"kind": "uniform", "value": 0.1},
                    },
                    {
                        "boundary": "wall",
                        "kind": "velocity",
                        "source": "analytic",
                        "value": [0.0, 0.0],
                    },
                    {
                        "boundary": "outlet",
                        "kind": "pressure",
                        "source": "analytic",
                        "value": 0.0,
                    },
                ],
                "solver": {"backend": "dense"},
            }
            case_path = directory / "case.yaml"
            case_path.write_text(yaml.safe_dump(case), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                code = run_cli(["run", str(case_path)])

            output = directory / "runs/thinned"
            self.assertEqual(code, 0)
            with h5py.File(output / "flow.h5") as h5:
                self.assertEqual(
                    list(h5["data"]),
                    ["step_000000", "step_000001", "step_000002"],
                )
                self.assertTrue(bool(h5.attrs["complete"]))
                self.assertEqual(
                    [
                        int(h5[f"data/step_{index:06d}"].attrs["solver_level"])
                        for index in range(3)
                    ],
                    [0, 2, 4],
                )
            self.assertFalse((output / "flow.partial.h5").exists())
            times = [
                float(item.attrib["Value"])
                for item in ET.parse(output / "flow.xdmf").iter("Time")
            ]
            np.testing.assert_allclose(times, [0.0, 0.5, 1.0])

    def test_runs_with_prepared_inlet_hdf5(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            shutil.copyfile(MESH, directory / "mesh.msh")
            save_inlet(
                directory / "inlet.h5",
                InletVelocityData(
                    boundary="inlet",
                    times=np.asarray([0.0, 0.25]),
                    node_ids=np.asarray([1]),
                    coords=np.asarray([[0.0, 0.0005, 0.0]]),
                    vals=np.asarray([[[0.1, 0.03]], [[0.1, 0.03]]]),
                    periodic=True,
                    source="analytic",
                ),
            )
            case = {
                "format": CASE_FORMAT,
                "version": CASE_VERSION,
                "mesh": {"path": "mesh.msh", "length_unit": "mm"},
                "fluid": {"rho": 1.0, "mu": 0.01},
                "time": {"dt": 0.25, "end_time": 0.25},
                "boundary_conditions": [
                    {
                        "source": "file",
                        "boundary": "inlet",
                        "kind": "velocity",
                        "path": "inlet.h5",
                    },
                    {
                        "boundary": "wall",
                        "kind": "velocity",
                        "source": "analytic",
                        "value": [0.0, 0.0],
                    },
                    {
                        "boundary": "outlet",
                        "kind": "pressure",
                        "source": "analytic",
                        "value": 0.0,
                    },
                ],
                "solver": {"backend": "dense"},
            }
            case_path = directory / "case.yaml"
            case_path.write_text(yaml.safe_dump(case), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                code = run_cli(["run", str(case_path)])

            self.assertEqual(code, 0)
            self.assertTrue((directory / "result" / "flow.h5").is_file())
            snapshot = yaml.safe_load(
                (directory / "result" / "case.yaml").read_text(encoding="utf-8")
            )
            inlet = next(
                condition
                for condition in snapshot["boundary_conditions"]
                if condition.get("source") == "file"
            )
            self.assertEqual(inlet["path"], "../inlet.h5")

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
                        "boundary_conditions": [
                            {
                                "boundary": "inlet",
                                "kind": "velocity",
                                "source": "analytic",
                                "profile": {"kind": "uniform", "value": 0.1},
                            },
                            {
                                "boundary": "wall",
                                "kind": "velocity",
                                "source": "analytic",
                                "value": [0.0, 0.0],
                            },
                            {
                                "boundary": "outlet",
                                "kind": "pressure",
                                "source": "analytic",
                                "value": 0.0,
                            },
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
                code = run_cli(["run", str(case_path)])

            self.assertEqual(code, 1)
            self.assertEqual(status.read_text(), '{"state": "complete"}\n')


if __name__ == "__main__":
    unittest.main()
