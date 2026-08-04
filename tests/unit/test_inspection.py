import unittest

import numpy as np

from femx_cfd.runtime.inspection import _mean_bulk_speed


class LineInlet:
    dimension = 1
    coordinates = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    elements = np.asarray([[0, 1], [1, 2]])
    mesh_node_ids = np.asarray([10, 11, 12])


class InletInspectionTest(unittest.TestCase):
    def test_area_and_time_averages_discrete_inlet_velocity(self):
        mean_speed = _mean_bulk_speed(
            LineInlet(),
            node_ids=np.asarray([11]),
            values=np.asarray([[2.0], [4.0], [2.0]]),
            times=np.asarray([0.0, 0.5, 1.0]),
        )

        self.assertEqual(mean_speed, 1.5)

    def test_rejects_values_for_nodes_outside_the_inlet(self):
        with self.assertRaisesRegex(ValueError, "outside the inlet"):
            _mean_bulk_speed(
                LineInlet(),
                node_ids=np.asarray([99]),
                values=np.asarray([[1.0], [1.0]]),
                times=np.asarray([0.0, 1.0]),
            )

    def test_uses_time_intervals_for_nonuniform_samples(self):
        mean_speed = _mean_bulk_speed(
            LineInlet(),
            node_ids=np.asarray([11]),
            values=np.asarray([[2.0], [4.0], [4.0]]),
            times=np.asarray([0.0, 0.25, 1.0]),
        )

        self.assertEqual(mean_speed, 1.875)


if __name__ == "__main__":
    unittest.main()
