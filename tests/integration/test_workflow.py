import tempfile
import unittest
from pathlib import Path

import femx
import h5py
import numpy as np

from apps.common.artifacts import write_truth
from femx_cfd import (
    EnVar,
    VelocityData,
    make_velocity_obs,
)


class EnVarWorkflowTest(unittest.TestCase):
    @staticmethod
    def make_prob():
        mesh_file = (
            Path(__file__).resolve().parents[1] / "data" / "2d_tiny_tube.msh"
        )
        model = femx.NavierStokesModel(
            mesh_file,
            num_steps=3,
            dt=0.25,
            rho=1.0,
            mu=0.01,
        )
        prob = femx.NavierStokesProblem(model)
        prob.add_bc(femx.DirichletBC("wall", "velocity", (0.0, 0.0)))
        prob.add_bc(femx.DirichletBC("outlet", "pressure", 0.0))
        prob.add_ctr(
            femx.NormalVelocityControl(
                "inlet",
                normal=(1.0, 0.0),
                times=(0.0, 0.5),
            )
        )
        op = femx.VelocityObservationOperator(
            model,
            points=((0.25, 0.5), (0.75, 0.5)),
            times=(0.25, 0.5, 0.75),
            components=(0, 1),
        )
        return prob, op

    def test_velocity_data_and_end_to_end_solve(self):
        prob, op = self.make_prob()
        basis = femx.EnsembleBasis(
            mean=np.array([0.1, 0.1]),
            perturbations=np.array([[0.1], [0.2]]),
        )
        truth_param = basis.evaluate(np.array([1.0]))
        truth = prob.solve(truth_param)
        velocity = VelocityData(
            points=np.asarray(op.points),
            times=np.asarray(op.times),
            vals=op.reshape(op.predict(truth)),
        )
        obs = make_velocity_obs(velocity, prob.model)

        envar = EnVar(
            prob,
            obs,
            basis,
            prior_weight=0.0,
        )
        res = envar.solve()

        self.assertEqual(envar.backend, "dense")
        np.testing.assert_allclose(res.coeff, [1.0])
        np.testing.assert_allclose(res.param, truth_param)
        self.assertLess(res.nonlinear_misfit, 1.0e-20)
        self.assertIsInstance(
            envar.obs_obj,
            femx.TimeLeastSquaresObjective,
        )
        self.assertEqual(res.num_forwards, 3)
        self.assertEqual(envar.solver.num_solves, 3)
        self.assertEqual(
            envar.solver.assembly_calls,
            3 * prob.model.num_steps,
        )

    def test_writes_complete_truth_trajectory(self):
        prob, _ = self.make_prob()
        truth = prob.solve(np.array([0.1, 0.1]))

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = Path(tmp_dir) / "truth.h5"
            xdmf_path = Path(tmp_dir) / "truth.xdmf"
            write_truth(h5_path, prob.model, truth)

            self.assertTrue(h5_path.is_file())
            self.assertTrue(xdmf_path.is_file())
            with h5py.File(h5_path, "r") as h5:
                self.assertEqual(
                    len(h5["Data"]),
                    prob.model.num_steps + 1,
                )

    def test_coefficient_bounds_are_applied_to_navier_stokes_envar(self):
        prob, op = self.make_prob()
        basis = femx.EnsembleBasis(
            mean=np.array([0.1, 0.1]),
            perturbations=np.array([[0.1], [0.2]]),
        )
        truth = prob.solve(basis.evaluate(np.array([1.0])))
        obs = femx.GaussianObservation(
            op,
            values=op.predict(truth),
            std=0.01,
        )
        envar = EnVar(
            prob,
            obs,
            basis,
            prior_weight=0.0,
        )

        res = envar.solve(bounds=(-0.25, 0.25))

        np.testing.assert_allclose(res.coeff, [0.25], atol=1.0e-8)
        self.assertEqual(res.num_forwards, 3)

    def test_uses_precomputed_observation_basis(self):
        prob, op = self.make_prob()
        basis = femx.EnsembleBasis(
            mean=np.array([0.1, 0.1]),
            perturbations=np.array([[0.1], [0.2]]),
        )
        mean_pred = op.predict(prob.solve(basis.mean))
        member_pred = op.predict(prob.solve(basis.evaluate([1.0])))
        obs_basis = femx.EnsembleBasis(
            mean_pred,
            (member_pred - mean_pred)[:, None],
        )
        obs = femx.GaussianObservation(op, member_pred, std=1.0)

        envar = EnVar(
            prob,
            obs,
            basis,
            obs_basis=obs_basis,
            basis_forwards=2,
            prior_weight=0.0,
        )
        res = envar.solve()

        np.testing.assert_allclose(res.coeff, [1.0])
        self.assertEqual(envar.solver.num_solves, 1)
        self.assertEqual(res.num_forwards, 3)


if __name__ == "__main__":
    unittest.main()
