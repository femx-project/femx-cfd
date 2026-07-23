import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from femx_cfd import VelocityData, load_case, save_velocity
from femx_cfd.runtime.output import (
    bundle_result,
    trajectory_levels,
    write_velocity_series,
)
from femx_cfd.runtime.runner import result_path


class ResultPathTest(unittest.TestCase):
    def test_explicit_result_path_is_relative_to_the_current_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            case = Path(tmp_dir) / "nested" / "case.yaml"

            result = result_path(case, "runs/example")

        self.assertEqual(result, (Path.cwd() / "runs/example").resolve())
        self.assertEqual(result_path(case), case.parent / "result")

    def test_uses_the_case_output_directory_unless_cli_overrides_it(self):
        case = load_case(
            Path(__file__).resolve().parents[2]
            / "examples/forward/straight_tube/case.yaml"
        )

        self.assertEqual(result_path(case), case.path.parent / "result")
        self.assertEqual(
            result_path(case, "runs/override"),
            (Path.cwd() / "runs/override").resolve(),
        )


class TrajectoryLevelsTest(unittest.TestCase):
    def test_keeps_uniform_initial_and_final_levels(self):
        np.testing.assert_array_equal(trajectory_levels(10, 5), [0, 5, 10])

    def test_rejects_a_nonuniform_final_interval(self):
        with self.assertRaisesRegex(ValueError, "must divide"):
            trajectory_levels(10, 4)


class ResultBundleTest(unittest.TestCase):
    def test_xdmf_data_is_merged_into_result_h5(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src = root / "source"
            out = root / "run"
            src.mkdir()

            with h5py.File(src / "algorithm.h5", "w") as h5:
                h5.create_dataset("param", data=[1.0, 2.0])
            with h5py.File(src / "estimate.h5", "w") as h5:
                h5.create_group("mesh").create_dataset("geometry", data=[[0, 0]])
                h5.create_group("data").create_dataset("velocity", data=[[3.0]])
            (src / "estimate.xdmf").write_text(
                '<DataItem Format="HDF">estimate.h5:/data/velocity</DataItem>\n',
                encoding="utf-8",
            )

            bundle_result(
                src / "algorithm.h5",
                src / "estimate.xdmf",
                out,
                {"objective": 1.25},
            )

            with h5py.File(out / "result.h5") as h5:
                np.testing.assert_allclose(h5["param"], [1.0, 2.0])
                self.assertEqual(list(h5["mesh"]), ["geometry"])
                np.testing.assert_allclose(
                    h5["estimate/fields/velocity"],
                    [[3.0]],
                )
                self.assertEqual(
                    h5["estimate/metrics"].attrs["objective"],
                    1.25,
                )
            self.assertIn(
                "result.h5:/estimate/fields/velocity",
                (out / "flow.xdmf").read_text(encoding="utf-8"),
            )


class ObservationVtuTest(unittest.TestCase):
    def test_writes_binary_vtu_temporal_point_cloud(self):
        velocity = VelocityData(
            points=np.asarray(((0.0, 0.0), (1.0, 0.0))),
            times=np.asarray((0.0, 0.5)),
            vals=np.asarray(
                (
                    ((1.0, 0.0), (2.0, 0.0)),
                    ((1.5, 0.0), (2.5, 0.0)),
                )
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            heavy = Path(tmp_dir) / "velocity.h5"
            save_velocity(heavy, velocity)
            pvd = write_velocity_series(heavy.with_suffix(".pvd"), velocity)
            pvd_text = pvd.read_text(encoding="utf-8")
            first = Path(tmp_dir) / "velocity/velocity_00000.vtu"
            second = Path(tmp_dir) / "velocity/velocity_00001.vtu"
            first_text = first.read_text(encoding="utf-8")

            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())

        self.assertEqual(pvd.name, "velocity.pvd")
        self.assertIn('timestep="0"', pvd_text)
        self.assertIn('timestep="0.5"', pvd_text)
        self.assertIn('file="velocity/velocity_00000.vtu"', pvd_text)
        self.assertIn('file="velocity/velocity_00001.vtu"', pvd_text)
        self.assertIn('NumberOfPoints="2"', first_text)
        self.assertIn('NumberOfCells="2"', first_text)
        self.assertIn('Name="velocity"', first_text)
        self.assertEqual(first_text.count('format="binary"'), 5)


if __name__ == "__main__":
    unittest.main()
