import unittest
from types import SimpleNamespace

import numpy as np

from femx_cfd.models.navier_stokes.ensemble import _analytic_ctr_modes


class EnsembleModeTest(unittest.TestCase):
    def test_peak_scaling_removes_initial_waveform_modulation(self):
        problem = SimpleNamespace(
            ctr=SimpleNamespace(
                times=np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
                kind="vector",
            ),
            num_ctr_levels=4,
            param_shape=(4, 2),
            model=SimpleNamespace(mesh=SimpleNamespace(dimension=2)),
        )
        mean = np.asarray(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.0],
            ]
        ).reshape(-1)
        config = {
            "time": {"T": 1.0},
            "ensemble": {
                "profile_scaling": "peak",
                "modes": {
                    "spatial": {"uniform": {"kind": "uniform"}},
                    "temporal": {"sin1": {"kind": "sine", "order": 1}},
                    "combined": [
                        {
                            "name": "flow_sin1",
                            "spatial": "uniform",
                            "temporal": "sin1",
                            "scale": 0.1,
                        }
                    ],
                },
            },
        }

        names, modes = _analytic_ctr_modes(config, problem, mean)

        self.assertEqual(names, ["flow_sin1_x", "flow_sin1_y"])
        x_mode = modes[:, 0].reshape(4, 1, 2)
        np.testing.assert_allclose(
            x_mode[:, 0, 0],
            [0.0, 0.2, 0.0, -0.2],
            atol=1.0e-15,
        )
        np.testing.assert_allclose(x_mode[:, 0, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
