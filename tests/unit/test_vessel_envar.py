import unittest

from femx_cfd.models.navier_stokes.problem import num_steps


class NavierStokesProblemTest(unittest.TestCase):
    def test_num_steps_requires_an_actual_integer_multiple(self):
        cfg = {
            "time": {"dt": 0.01},
            "simulation": {"end_time": 0.48},
        }
        self.assertEqual(num_steps(cfg), 48)

        cfg["time"]["dt"] = 0.01000001
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            num_steps(cfg)


if __name__ == "__main__":
    unittest.main()
