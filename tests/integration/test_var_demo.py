import unittest

import femx
import numpy as np

from femx_cfd import Var
from tests.integration.rectangle_demo import make_problem, truth_parameter


class RectangleVarDemoTest(unittest.TestCase):
    def test_reduces_objective_on_demo_mesh(self):
        if "resolve" not in femx.solver_backends():
            self.skipTest("femx was built without ReSolve")

        prob, op = make_problem()
        truth_param = truth_parameter(prob)
        truth = prob.solve(truth_param, backend="resolve")
        obs = femx.GaussianObservation(
            op,
            values=op.predict(truth),
            std=0.01,
        )
        var = Var(prob, obs, backend="resolve")
        init_params = np.full(prob.num_param, 0.05)
        init_obj = var.reduced.value(init_params)

        res = var.solve(
            init_params,
            bnds=[(0.0, 0.3)] * prob.num_param,
            max_itrs=3,
            grad_tol=0.0,
            func_tol=0.0,
        )

        self.assertGreater(prob.model.num_states, 0)
        self.assertEqual(prob.num_param, 31)
        self.assertEqual(obs.values.size, 248)
        self.assertLess(res.obj, init_obj)
        self.assertTrue(np.all(res.param >= 0.0))
        self.assertTrue(np.all(res.param <= 0.3))


if __name__ == "__main__":
    unittest.main()
