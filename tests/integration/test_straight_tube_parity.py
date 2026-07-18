import unittest

import femx
import numpy as np

from femx_cfd import Var
from tests.integration.rectangle_demo import make_problem, truth_parameter


class StraightTubeParityTest(unittest.TestCase):
    def test_demo_objective_and_gradient_are_stable(self):
        if "petsc" not in femx.solver_backends():
            self.skipTest("femx was built without PETSc")

        prob, op = make_problem()
        truth = prob.solve(truth_parameter(prob), backend="petsc")
        obs = femx.GaussianObservation(
            op,
            values=op.predict(truth),
            std=1.0,
        )
        var = Var(prob, obs, backend="petsc")
        init_params = np.full(prob.num_param, 1.0e-12)

        obj, grad = var.value_grad(init_params)

        self.assertAlmostEqual(obj, 1.2076350898, delta=1.0e-8)
        self.assertAlmostEqual(np.linalg.norm(grad), 3.0572382563, delta=1.0e-8)


if __name__ == "__main__":
    unittest.main()
