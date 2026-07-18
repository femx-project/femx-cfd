"""Tests for portable-to-model fixed boundary translation."""

import unittest

from femx_cfd.workflows.boundaries import fixed_conditions


class FixedBoundaryTest(unittest.TestCase):
    def test_translates_explicit_velocity_and_pressure_conditions(self):
        case = {
            "boundary_conditions": [
                {
                    "boundary": "wall",
                    "kind": "velocity",
                    "value": [0.0, 0.0],
                },
                {"boundary": "outlet", "kind": "pressure", "value": 0.0},
            ]
        }

        conditions = fixed_conditions(case, dimension=2)

        self.assertEqual(conditions[0]["field"], "velocity")
        self.assertEqual(conditions[0]["space"]["value"], [0.0, 0.0])
        self.assertEqual(conditions[1]["field"], "pressure")
        self.assertEqual(conditions[1]["space"]["value"], 0.0)

    def test_rejects_velocity_vector_for_another_mesh_dimension(self):
        case = {
            "boundary_conditions": [
                {
                    "boundary": "wall",
                    "kind": "velocity",
                    "value": [0.0, 0.0, 0.0],
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "wall.*2 values"):
            fixed_conditions(case, dimension=2)


if __name__ == "__main__":
    unittest.main()
