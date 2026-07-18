import tempfile
import unittest
from pathlib import Path

import femx
import h5py
import numpy as np

from femx_cfd import (
    InletData,
    VelocityData,
    add_velocity_noise,
    load_inlet,
    load_velocity,
    make_velocity_obs,
    save_inlet,
    save_velocity,
)


MESH_FILE = Path(__file__).resolve().parents[1] / "data" / "2d_tiny_tube.msh"


class InletDataTest(unittest.TestCase):
    def test_h5_round_trip(self):
        inlet = InletData(
            boundary="inlet",
            times=np.array([0.0, 0.5, 1.0]),
            node_ids=np.array([4, 7]),
            coords=np.array([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]]),
            normal=np.array([2.0, 0.0]),
            vals=np.array([[1.0, 2.0], [2.0, 3.0], [1.0, 2.0]]),
            source="analytic",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "inlet.h5"
            save_inlet(path, inlet)
            with h5py.File(path) as h5:
                self.assertNotIn("control_kind", h5.attrs)
                self.assertIn("normal_velocity", h5["inlet"])
            loaded = load_inlet(path)

        self.assertEqual(loaded.boundary, "inlet")
        self.assertEqual(loaded.source, "analytic")
        self.assertTrue(loaded.periodic)
        np.testing.assert_array_equal(loaded.times, inlet.times)
        np.testing.assert_array_equal(loaded.node_ids, inlet.node_ids)
        np.testing.assert_array_equal(loaded.coords, inlet.coords)
        np.testing.assert_allclose(loaded.normal, [1.0, 0.0])
        np.testing.assert_array_equal(loaded.vals, inlet.vals)
        np.testing.assert_array_equal(loaded.params, [1.0, 2.0, 2.0, 3.0])

    def test_rejects_legacy_femx_inv_format(self):
        inlet = InletData(
            boundary="inlet",
            times=np.array([0.0, 1.0]),
            node_ids=np.array([4]),
            coords=np.array([[0.0, 0.0]]),
            normal=np.array([1.0, 0.0]),
            vals=np.array([[1.0], [1.0]]),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "inlet.h5"
            save_inlet(path, inlet)
            with h5py.File(path, "a") as h5:
                h5.attrs["format"] = "femx-inv"

            with self.assertRaisesRegex(ValueError, "not a femx-cfd"):
                load_inlet(path)

    def test_orders_parameters_for_problem_nodes(self):
        inlet = InletData(
            boundary="inlet",
            times=np.array([0.0, 0.5, 1.0]),
            node_ids=np.array([4, 7]),
            coords=np.zeros((2, 3)),
            normal=np.array([1.0, 0.0]),
            vals=np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]),
        )
        ctr = type("Control", (), {"times": (0.0, 0.5, 1.0)})()
        prob = type(
            "Problem",
            (),
            {"ctr": ctr, "ctr_mesh_node_ids": np.array([7, 4])},
        )()

        np.testing.assert_array_equal(
            inlet.params_for(prob),
            [2.0, 1.0, 4.0, 3.0],
        )


class VelocityDataTest(unittest.TestCase):
    def make_velocity(self):
        return VelocityData(
            points=np.array([[0.25, 0.5], [0.75, 0.5]]),
            times=np.array([0.5, 1.0]),
            vals=np.arange(8, dtype=float).reshape(2, 2, 2),
        )

    def test_h5_round_trip(self):
        velocity = self.make_velocity()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "velocity.h5"
            save_velocity(path, velocity)
            loaded = load_velocity(path)

        np.testing.assert_array_equal(loaded.points, velocity.points)
        np.testing.assert_array_equal(loaded.times, velocity.times)
        np.testing.assert_array_equal(loaded.vals, velocity.vals)

    def test_creates_observation_directly_from_velocity(self):
        velocity = self.make_velocity()
        model = femx.NavierStokesModel(
            MESH_FILE,
            num_steps=2,
            dt=0.5,
            rho=1.0,
            mu=0.01,
        )

        obs = make_velocity_obs(velocity, model)

        np.testing.assert_array_equal(obs.values, velocity.vals.reshape(-1))
        np.testing.assert_array_equal(obs.std, np.ones(obs.op.num_obs))

    def test_adds_reproducible_velocity_noise(self):
        velocity = self.make_velocity()

        first = add_velocity_noise(velocity, std=0.1, seed=42)
        second = add_velocity_noise(velocity, std=0.1, seed=42)

        np.testing.assert_array_equal(first.vals, second.vals)
        self.assertFalse(np.array_equal(first.vals, velocity.vals))
        np.testing.assert_array_equal(first.points, velocity.points)
        np.testing.assert_array_equal(first.times, velocity.times)


if __name__ == "__main__":
    unittest.main()
