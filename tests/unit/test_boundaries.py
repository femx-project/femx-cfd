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
                    "source": "analytic",
                    "value": [0.0, 0.0],
                },
                {
                    "boundary": "outlet",
                    "kind": "pressure",
                    "source": "analytic",
                    "value": 0.0,
                },
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
                    "source": "analytic",
                    "value": [0.0, 0.0, 0.0],
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "wall.*2 values"):
            fixed_conditions(case, dimension=2)

    def test_ignores_profile_driven_analytic_condition(self):
        case = {
            "boundary_conditions": [
                {
                    "boundary": "inlet",
                    "kind": "velocity",
                    "source": "analytic",
                    "profile": {"kind": "uniform", "value": 0.1},
                },
                {
                    "boundary": "wall",
                    "kind": "velocity",
                    "source": "analytic",
                    "value": [0.0, 0.0],
                },
            ]
        }

        conditions = fixed_conditions(case, dimension=2)

        self.assertEqual(len(conditions), 1)
        self.assertEqual(conditions[0]["boundary"], "wall")


if __name__ == "__main__":
    unittest.main()
