import unittest
from pathlib import Path

import femx
import numpy as np

from femx_cfd.envar import _LinearProblem, _build_obs_basis


class FemxControlledEnVarIntegrationTest(unittest.TestCase):
    def test_recovers_a_ctr_mode(self):
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

        def predict(params):
            return op.predict(prob.solve(params))

        param_basis = femx.EnsembleBasis(
            mean=np.array([0.1, 0.1]),
            perturbations=np.array([[0.1], [0.2]]),
        )
        obs_basis = _build_obs_basis(param_basis, predict)
        truth_param = param_basis.evaluate(np.array([1.0]))
        obs = predict(truth_param)

        res = _LinearProblem(
            param_basis,
            obs_basis,
            prior_weight=0.0,
        ).solve(obs, std=0.01)

        np.testing.assert_allclose(res.coeff, [1.0], atol=1.0e-12)
        np.testing.assert_allclose(res.param, truth_param)
        self.assertLess(res.obs_misfit, 1.0e-20)


if __name__ == "__main__":
    unittest.main()
