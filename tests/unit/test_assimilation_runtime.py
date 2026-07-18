import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from femx_cfd import VelocityData, save_velocity
from femx_cfd.runtime.output import bundle_result, write_observations


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
                h5.create_group("Mesh").create_dataset("Geometry", data=[[0, 0]])
                h5.create_group("Data").create_dataset("velocity", data=[[3.0]])
            (src / "estimate.xdmf").write_text(
                '<DataItem Format="HDF">estimate.h5:/Data/velocity</DataItem>\n',
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
                np.testing.assert_allclose(h5["Data/velocity"], [[3.0]])
                self.assertEqual(h5["metrics"].attrs["objective"], 1.25)
            self.assertIn(
                "result.h5:/Data/velocity",
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
            pvd = write_observations(heavy, velocity)
            pvd_text = pvd.read_text(encoding="utf-8")
            first = Path(tmp_dir) / "observations/observation_00000.vtu"
            second = Path(tmp_dir) / "observations/observation_00001.vtu"
            first_text = first.read_text(encoding="utf-8")

            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())

        self.assertEqual(pvd.name, "observations.pvd")
        self.assertIn('timestep="0"', pvd_text)
        self.assertIn('timestep="0.5"', pvd_text)
        self.assertIn('file="observations/observation_00000.vtu"', pvd_text)
        self.assertIn('file="observations/observation_00001.vtu"', pvd_text)
        self.assertIn('NumberOfPoints="2"', first_text)
        self.assertIn('NumberOfCells="2"', first_text)
        self.assertIn('Name="velocity"', first_text)
        self.assertEqual(first_text.count('format="binary"'), 5)


if __name__ == "__main__":
    unittest.main()
