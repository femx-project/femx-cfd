import unittest

import numpy as np

from femx_cfd.prep.synthetic import obs_times


class AssimilationProblemTest(unittest.TestCase):
    def test_observation_phases_cover_one_cycle(self):
        cfg = {"time": {"T": 1.0}, "observation": {"phases": 10}}

        np.testing.assert_allclose(obs_times(cfg), np.arange(10) / 10.0)
        np.testing.assert_allclose(
            obs_times(cfg, cycle=2),
            1.0 + np.arange(10) / 10.0,
        )


if __name__ == "__main__":
    unittest.main()
