import unittest
from pathlib import Path

import femx
import numpy as np

from femx_cfd import InletData, VelocityData
from femx_cfd.models import navier_stokes as ns


MESH_FILE = Path(__file__).resolve().parents[1] / "data" / "2d_tiny_tube.msh"


class ParallelEnsembleIntegrationTest(unittest.TestCase):
    def test_builds_spinup_and_observation_bases_with_petsc_processes(self):
        cfg = {
            "mesh": str(MESH_FILE),
            "control": {
                "boundary": "inlet",
                "kind": "normal",
            },
            "bc": [
                {
                    "boundary": "wall",
                    "field": "velocity",
                    "components": "all",
                    "space": {"kind": "uniform", "value": [0.0, 0.0]},
                    "time": {"kind": "uniform"},
                },
                {
                    "boundary": "outlet",
                    "field": "pressure",
                    "space": {"kind": "uniform", "value": 0.0},
                    "time": {"kind": "uniform"},
                },
            ],
            "fluid": {"rho": 1.0, "mu": 0.01},
            "time": {"T": 0.5, "dt": 0.25},
            "simulation": {"end_time": 0.5},
            "solver": {"backend": "petsc"},
            "ensemble": {
                "workers": 2,
                "modes": {
                    "spatial": {"uniform": {"kind": "uniform"}},
                    "temporal": {"mean": {"kind": "constant"}},
                    "combined": [
                        {
                            "name": "flow_mean",
                            "spatial": "uniform",
                            "temporal": "mean",
                            "scale": 0.1,
                        }
                    ],
                },
            },
        }
        mesh = femx.Mesh.read(MESH_FILE)
        surface = mesh.boundary("inlet")
        active = surface.interior_node_ids
        inlet = InletData(
            boundary="inlet",
            times=np.array([0.0, 0.5]),
            node_ids=np.asarray(surface.mesh_node_ids)[active],
            coords=np.asarray(surface.coordinates)[active],
            normal=np.array([1.0, 0.0]),
            vals=np.full((2, active.size), 0.1),
        )
        velocity = VelocityData(
            points=np.array([[0.25, 0.5], [0.75, 0.5]]),
            times=np.array([0.25, 0.5]),
            vals=np.zeros((2, 2, 2)),
        )

        events = []
        spin = ns.build_ensemble(cfg, inlet, progress=events.append)
        prob = ns.make_problem(
            cfg,
            times=inlet.times,
            periodic=inlet.periodic,
            normal=inlet.normal,
            init_mean=spin.init_mean,
            init_modes=spin.init_modes,
        )
        param_basis = ns.make_param_basis(
            prob,
            spin.ctr_mean,
            spin.ctr_modes,
        )
        obs_basis = ns.build_obs_basis(
            cfg,
            inlet,
            spin,
            velocity,
            param_basis,
            progress=events.append,
        )

        self.assertEqual(spin.init_modes.shape[1], 1)
        self.assertEqual(obs_basis.num_coefficients, 1)
        self.assertEqual(obs_basis.value_size, 8)
        self.assertTrue(
            any(
                event["type"] == "ensemble_solve" and event["stage"] == "spin-up"
                for event in events
            )
        )
        self.assertTrue(
            any(
                event["type"] == "ensemble_solve" and event["stage"] == "observation"
                for event in events
            )
        )


if __name__ == "__main__":
    unittest.main()
