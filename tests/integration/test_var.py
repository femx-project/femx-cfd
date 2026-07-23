import unittest
from pathlib import Path

import femx
import numpy as np

from femx_cfd import Var
from femx_cfd.models.navier_stokes import make_reg


class VarTest(unittest.TestCase):
    @staticmethod
    def make_prob_and_obs():
        mesh_file = Path(__file__).resolve().parents[1] / "data" / "2d_tiny_tube.msh"
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
            components=(0,),
        )
        truth_param = np.array([0.3, 0.7])
        truth = prob.solve(truth_param)
        obs = femx.GaussianObservation(op, values=op.predict(truth))
        return prob, obs, truth_param

    @staticmethod
    def make_prob_and_obs_with_init_mode(periodic=False):
        mesh_file = Path(__file__).resolve().parents[1] / "data" / "2d_tiny_tube.msh"
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
                times=(0.0, 0.5, 0.75) if periodic else (0.0, 0.5),
                periodic=periodic,
            )
        )
        prob.build()

        constrained = np.union1d(prob.fixed_dofs, prob.ctr_state_dofs)
        free_velocity = np.setdiff1d(model.velocity_dofs, constrained)
        modes = np.zeros((model.num_states, 1))
        modes[free_velocity[::2], 0] = 0.5
        prob.add_init_ctr(femx.InitialStateControl(modes, mean=prob.initial_state))
        prob.build()

        np.testing.assert_array_equal(modes[prob.fixed_dofs], 0.0)
        np.testing.assert_array_equal(modes[prob.ctr_state_dofs], 0.0)
        pressure_dofs = np.setdiff1d(np.arange(model.num_states), model.velocity_dofs)
        np.testing.assert_array_equal(modes[pressure_dofs], 0.0)

        op = femx.VelocityObservationOperator(
            model,
            points=((0.25, 0.5), (0.75, 0.5)),
            times=(0.25, 0.5, 0.75),
            components=(0,),
        )
        truth_param = np.array([0.3, 0.3, 0.7])
        truth = prob.solve(truth_param)
        obs = femx.GaussianObservation(op, values=op.predict(truth))
        return prob, obs

    def test_initial_mode_adjoint_gradient_matches_central_difference(self):
        prob, obs = self.make_prob_and_obs_with_init_mode()
        bg = np.array([0.0, 0.2, 0.4])
        reg = make_reg(
            prob,
            bg=bg,
            bg_weight=0.2,
            init_weight=0.05,
            space_weight=0.1,
            time_weight=0.05,
        )
        var = Var(prob, obs, reg=reg.obj, init_params=reg.bg)
        param = np.array([0.1, 0.15, 0.5])

        init_only = bg.copy()
        init_only[0] = 0.1
        self.assertAlmostEqual(
            reg.bg_obj.value(prob.solve(init_only), init_only),
            0.5 * 0.05 * 0.1**2,
        )
        self.assertEqual(reg.init_weight, 0.05)

        _, grad = var.value_grad(param)
        self.assertGreater(abs(grad[0]), 1.0e-4)
        step = 1.0e-6
        finite_diff = np.empty(param.size)
        for i in range(param.size):
            perturb = np.zeros(param.size)
            perturb[i] = step
            finite_diff[i] = (
                var.reduced.value(param + perturb) - var.reduced.value(param - perturb)
            ) / (2.0 * step)

        np.testing.assert_allclose(
            grad,
            finite_diff,
            rtol=2.0e-6,
            atol=1.0e-9,
        )

    def test_periodic_q0_and_initial_gradient_match_central_difference(self):
        prob, obs = self.make_prob_and_obs_with_init_mode(periodic=True)
        param = np.array([0.1, 0.15, 0.5])
        reg = make_reg(
            prob,
            bg=np.array([0.0, 0.2, 0.4]),
            bg_weight=0.2,
            space_weight=0.1,
            time_weight=0.05,
        )
        var = Var(prob, obs, reg=reg.obj, init_params=reg.bg)

        traj = prob.solve(param)
        normal_dof = prob.ctr_state_dofs[0]
        self.assertAlmostEqual(traj.values[0, normal_dof], param[1])
        self.assertAlmostEqual(traj.values[-1, normal_dof], param[1])

        _, grad = var.value_grad(param)
        step = 1.0e-6
        finite_diff = np.empty(param.size)
        for i in range(param.size):
            perturb = np.zeros(param.size)
            perturb[i] = step
            finite_diff[i] = (
                var.reduced.value(param + perturb) - var.reduced.value(param - perturb)
            ) / (2.0 * step)

        np.testing.assert_allclose(
            grad,
            finite_diff,
            rtol=2.0e-6,
            atol=1.0e-9,
        )

    def _assert_init_mode_backend_matches_dense(self, backend):
        prob, obs = self.make_prob_and_obs_with_init_mode()
        param = np.array([0.1, 0.15, 0.5])
        dense_obj, dense_grad = Var(prob, obs).value_grad(param)
        obj, grad = Var(prob, obs, backend=backend).value_grad(param)

        np.testing.assert_allclose(obj, dense_obj, rtol=2.0e-7)
        np.testing.assert_allclose(
            grad,
            dense_grad,
            rtol=2.0e-6,
            atol=2.0e-9,
        )

    def test_initial_mode_resolve_matches_dense(self):
        if "resolve" not in femx.solver_backends():
            self.skipTest("femx was built without ReSolve")
        self._assert_init_mode_backend_matches_dense("resolve")

    def test_initial_mode_petsc_matches_dense(self):
        if "petsc" not in femx.solver_backends():
            self.skipTest("femx was built without PETSc")
        self._assert_init_mode_backend_matches_dense("petsc")

    def test_value_grad_includes_regularization(self):
        prob, obs, _ = self.make_prob_and_obs()
        bg = np.array([0.2, 0.4])
        reg = make_reg(
            prob,
            bg=bg,
            bg_weight=2.0,
            space_weight=4.0,
            time_weight=3.0,
        )
        var = Var(prob, obs, reg=reg.obj, init_params=reg.bg)
        param = np.array([0.1, 0.6])

        obj, grad = var.value_grad(param)
        traj = prob.solve(param)
        expected_obj = var.obs_obj.value(traj, param) + reg.obj.value(traj, param)
        self.assertAlmostEqual(obj, expected_obj, places=13)

        surface = prob.model.mesh.boundary(prob.ctr.boundary)
        local = {int(node): i for i, node in enumerate(surface.mesh_node_ids)}
        active = [local[int(node)] for node in prob.ctr_mesh_node_ids]
        stiffness, mass, _ = surface.laplacian_matrices()
        stiffness = stiffness[active][:, active]
        mass = mass[active][:, active]
        centered = (param - bg).reshape(prob.param_shape)
        levels = param.reshape(prob.param_shape)
        expected_bg = 0.5 * 2.0 * sum(vals @ mass @ vals for vals in centered)
        expected_space = 0.5 * 4.0 * sum(vals @ stiffness @ vals for vals in levels)
        expected_time = 0.5 * 3.0 * np.sum(np.diff(centered, axis=0) ** 2)
        self.assertAlmostEqual(reg.bg_obj.value(traj, param), expected_bg)
        self.assertAlmostEqual(reg.space_obj.value(traj, param), expected_space)
        self.assertAlmostEqual(reg.time_obj.value(traj, param), expected_time)

        step = 1.0e-6
        finite_diff = np.empty(param.size)
        for i in range(param.size):
            perturb = np.zeros(param.size)
            perturb[i] = step
            finite_diff[i] = (
                var.reduced.value(param + perturb) - var.reduced.value(param - perturb)
            ) / (2.0 * step)
        np.testing.assert_allclose(
            grad,
            finite_diff,
            rtol=2.0e-6,
            atol=1.0e-9,
        )

    def test_value_grad_reports_solve_progress(self):
        prob, obs, _ = self.make_prob_and_obs()
        events = []

        Var(prob, obs).value_grad(
            np.array([0.1, 0.2]),
            prog=events.append,
        )

        phases = [event["phase"] for event in events]
        self.assertIn("forward-begin", phases)
        self.assertIn("forward-end", phases)
        self.assertIn("adjoint-begin", phases)
        self.assertIn("adjoint-end", phases)

    def test_recovers_synthetic_ctr_with_unit_obs_sigma(self):
        prob, obs, truth_param = self.make_prob_and_obs()
        var = Var(prob, obs)

        self.assertEqual(var.backend, "dense")

        res = var.solve(
            np.array([0.1, 0.2]),
            max_itrs=100,
            grad_tol=1.0e-10,
            func_tol=1.0e-15,
        )

        self.assertTrue(res.converged, res.msg)
        np.testing.assert_allclose(
            res.param,
            truth_param,
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertLess(res.obs_misfit, 1.0e-13)
        self.assertEqual(res.reg, 0.0)
        self.assertLess(res.grad_norm, 1.0e-8)
        np.testing.assert_allclose(
            res.pred,
            obs.values,
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_resolve_recovers_synthetic_ctr(self):
        if "resolve" not in femx.solver_backends():
            self.skipTest("femx was built without ReSolve")

        prob, obs, truth_param = self.make_prob_and_obs()
        res = Var(prob, obs, backend="resolve").solve(
            np.array([0.1, 0.2]),
            max_itrs=100,
            grad_tol=1.0e-10,
            func_tol=1.0e-15,
        )

        self.assertTrue(res.converged, res.msg)
        np.testing.assert_allclose(
            res.param,
            truth_param,
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertLess(res.obs_misfit, 1.0e-13)
        self.assertLess(res.grad_norm, 1.0e-8)

    def test_petsc_recovers_synthetic_ctr(self):
        if "petsc" not in femx.solver_backends():
            self.skipTest("femx was built without PETSc")

        prob, obs, truth_param = self.make_prob_and_obs()
        res = Var(prob, obs, backend="petsc").solve(
            np.array([0.1, 0.2]),
            max_itrs=100,
            grad_tol=1.0e-10,
            func_tol=1.0e-15,
        )

        self.assertTrue(res.converged, res.msg)
        np.testing.assert_allclose(
            res.param,
            truth_param,
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertLess(res.obs_misfit, 1.0e-13)
        self.assertLess(res.grad_norm, 1.0e-8)

    def test_tao_recovers_bounded_synthetic_ctr(self):
        if "petsc" not in femx.solver_backends():
            self.skipTest("femx was built without PETSc/TAO")

        prob, obs, truth_param = self.make_prob_and_obs()
        events = []
        res = Var(prob, obs).solve(
            np.array([0.1, 0.2]),
            opt="tao",
            bnds=((0.0, 0.4), (0.0, 1.0)),
            max_itrs=100,
            grad_tol=1.0e-10,
            prog=events.append,
        )

        self.assertTrue(res.converged, res.msg)
        np.testing.assert_allclose(
            res.param,
            truth_param,
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertLess(res.obs_misfit, 1.0e-13)
        self.assertLess(res.grad_norm, 1.0e-8)
        self.assertTrue(any(event["type"] == "optimizer" for event in events))
        phases = {event["phase"] for event in events if event["type"] == "solve"}
        self.assertIn("forward-begin", phases)
        self.assertIn("adjoint-step", phases)

    def test_defaults_init_param_to_bg(self):
        prob, obs, truth_param = self.make_prob_and_obs()
        var = Var(prob, obs, init_params=truth_param)

        res = var.solve(max_itrs=1)

        np.testing.assert_array_equal(res.param, truth_param)
        self.assertEqual(res.num_fun, 1)

    def test_rejects_invalid_solver_options(self):
        prob, obs, _ = self.make_prob_and_obs()
        var = Var(prob, obs)

        with self.assertRaisesRegex(ValueError, "must be positive"):
            var.solve(max_itrs=0)
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            var.solve(grad_tol=np.inf)
        with self.assertRaisesRegex(ValueError, "'scipy' or 'tao'"):
            var.solve(opt="other")


if __name__ == "__main__":
    unittest.main()
