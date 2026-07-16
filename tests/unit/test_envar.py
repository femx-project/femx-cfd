import unittest
from unittest.mock import patch

import femx
import numpy as np

from femx_cfd.envar import _LinearProblem, _build_obs_basis


class LinearEnVarTest(unittest.TestCase):
    def test_uses_femx_ensemble_basis(self):
        basis = femx.EnsembleBasis(np.zeros(1), np.ones((1, 1)))
        obs_basis = _build_obs_basis(basis, lambda params: params)

        self.assertIsInstance(obs_basis, femx.EnsembleBasis)

    def test_reports_ensemble_forward_progress(self):
        basis = femx.EnsembleBasis(np.zeros(2), np.eye(2))
        events = []

        _build_obs_basis(basis, lambda params: params, prog=events.append)

        self.assertEqual(
            events,
            [
                {
                    "type": "ensemble",
                    "phase": "mean",
                    "status": "begin",
                    "step": 1,
                    "total": 3,
                },
                {
                    "type": "ensemble",
                    "phase": "mean",
                    "status": "end",
                    "step": 1,
                    "total": 3,
                },
                {
                    "type": "ensemble",
                    "phase": "member",
                    "status": "begin",
                    "step": 2,
                    "total": 3,
                },
                {
                    "type": "ensemble",
                    "phase": "member",
                    "status": "end",
                    "step": 2,
                    "total": 3,
                },
                {
                    "type": "ensemble",
                    "phase": "member",
                    "status": "begin",
                    "step": 3,
                    "total": 3,
                },
                {
                    "type": "ensemble",
                    "phase": "member",
                    "status": "end",
                    "step": 3,
                    "total": 3,
                },
            ],
        )

    def test_recovers_linear_problem_in_ensemble_subspace(self):
        param_basis = femx.EnsembleBasis(
            mean=np.array([1.0, -1.0]),
            perturbations=np.array([[2.0, 0.0], [0.0, 0.5]]),
        )
        obs_matrix = np.array([[1.0, 2.0], [-1.0, 1.0]])
        obs_basis = _build_obs_basis(
            param_basis,
            lambda params: obs_matrix @ params,
        )
        truth_coeff = np.array([0.25, -2.0])
        obs = obs_matrix @ param_basis.evaluate(truth_coeff)

        res = _LinearProblem(
            param_basis,
            obs_basis,
            prior_weight=0.0,
        ).solve(obs, std=0.1)

        np.testing.assert_allclose(res.coeff, truth_coeff)
        np.testing.assert_allclose(
            res.param,
            param_basis.evaluate(truth_coeff),
        )
        self.assertAlmostEqual(res.obj, 0.0, places=24)
        self.assertLess(res.grad_norm, 1.0e-12)

    def test_prior_regularizes_coefficients(self):
        basis = femx.EnsembleBasis(np.zeros(2), np.eye(2))
        res = _LinearProblem(
            basis,
            basis,
            prior_weight=1.0,
        ).solve(np.array([2.0, -4.0]))

        np.testing.assert_allclose(res.coeff, [1.0, -2.0])
        self.assertAlmostEqual(res.obs_misfit, 2.5)
        self.assertAlmostEqual(res.prior, 2.5)
        self.assertAlmostEqual(res.obj, 5.0)

    def test_avoids_squaring_observation_basis_condition_number(self):
        basis = femx.EnsembleBasis(np.zeros(2), np.eye(2))
        obs_basis = femx.EnsembleBasis(
            np.zeros(2),
            np.diag([1.0e-8, 1.0]),
        )
        truth_coeff = np.array([1.0, 2.0])

        res = _LinearProblem(
            basis,
            obs_basis,
            prior_weight=0.0,
        ).solve(obs_basis.evaluate(truth_coeff))

        np.testing.assert_allclose(res.coeff, truth_coeff)

    def test_scalar_bounds_limit_every_coefficient(self):
        basis = femx.EnsembleBasis(np.zeros(2), np.eye(2))

        res = _LinearProblem(
            basis,
            basis,
            prior_weight=0.0,
        ).solve(np.array([2.0, -4.0]), bounds=(-1.0, 1.0))

        np.testing.assert_allclose(res.coeff, [1.0, -1.0], atol=1.0e-8)
        self.assertAlmostEqual(res.obs_misfit, 5.0, places=7)
        self.assertAlmostEqual(res.obj, 5.0, places=7)
        self.assertAlmostEqual(res.grad_norm, np.sqrt(10.0), places=7)

    def test_per_coefficient_bounds_include_prior_in_bounded_solve(self):
        basis = femx.EnsembleBasis(np.zeros(2), np.eye(2))

        res = _LinearProblem(
            basis,
            basis,
            prior_weight=1.0,
        ).solve(
            np.array([4.0, -4.0]),
            bounds=((-0.5, 0.5), (-3.0, -1.0)),
        )

        np.testing.assert_allclose(res.coeff, [0.5, -2.0], atol=1.0e-8)
        self.assertAlmostEqual(res.obs_misfit, 8.125, places=7)
        self.assertAlmostEqual(res.prior, 2.125, places=7)
        self.assertAlmostEqual(res.obj, 10.25, places=7)
        self.assertAlmostEqual(res.grad_norm, 3.0, places=7)

    def test_none_bounds_keep_numpy_least_squares_path(self):
        basis = femx.EnsembleBasis(np.zeros(2), np.eye(2))

        with patch(
            "femx_cfd.envar.lsq_linear",
            side_effect=AssertionError("bounded solver must not be called"),
        ):
            res = _LinearProblem(
                basis,
                basis,
                prior_weight=0.0,
            ).solve(np.array([2.0, -4.0]))

        np.testing.assert_allclose(res.coeff, [2.0, -4.0])

    def test_validates_coefficient_bounds(self):
        basis = femx.EnsembleBasis(np.zeros(2), np.eye(2))
        prob = _LinearProblem(basis, basis, prior_weight=0.0)
        obs = np.zeros(2)

        invalid = (
            (1.0, TypeError),
            ((-1.0, 0.0, 1.0), ValueError),
            ((((0.0, 1.0),),), ValueError),
            ((0.0, 0.0), ValueError),
            ((np.nan, 1.0), ValueError),
            (((-1.0, 1.0), (2.0, 1.0)), ValueError),
            ((False, 1.0), TypeError),
        )
        for bounds, error in invalid:
            with self.subTest(bounds=bounds):
                with self.assertRaises(error):
                    prob.solve(obs, bounds=bounds)

    def test_none_endpoints_are_unbounded(self):
        basis = femx.EnsembleBasis(np.zeros(2), np.eye(2))

        res = _LinearProblem(
            basis,
            basis,
            prior_weight=0.0,
        ).solve(
            np.array([2.0, -4.0]),
            bounds=((None, 1.0), (-5.0, None)),
        )

        np.testing.assert_allclose(res.coeff, [1.0, -4.0], atol=1.0e-8)


if __name__ == "__main__":
    unittest.main()
