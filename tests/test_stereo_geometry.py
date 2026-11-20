import math
import unittest

import numpy as np

from space_grasp.estimation.messages import TargetEstimate


class StereoGeometryTests(unittest.TestCase):
    def test_known_diameter_depth(self) -> None:
        focal_px = 610.0
        diameter = 0.024
        radius_px = 34.6
        depth = focal_px * diameter / (2.0 * radius_px)
        self.assertAlmostEqual(depth, 0.2116, places=3)

    def test_estimate_rejects_oracle_source(self) -> None:
        with self.assertRaises(AssertionError):
            TargetEstimate(
                timestamp=0.0,
                position=np.zeros(3),
                velocity=np.zeros(3),
                source="oracle",
                valid=True,
            )

    def test_midpoint_triangulation(self) -> None:
        left_origin = np.array([0.0, -0.1, 0.0])
        right_origin = np.array([0.0, 0.1, 0.0])
        target = np.array([0.8, 0.0, 0.0])
        left_dir = target - left_origin
        right_dir = target - right_origin
        left_dir /= np.linalg.norm(left_dir)
        right_dir /= np.linalg.norm(right_dir)
        identity = np.eye(3)
        system = np.zeros((3, 3))
        rhs = np.zeros(3)
        for origin, direction in ((left_origin, left_dir), (right_origin, right_dir)):
            projector = identity - np.outer(direction, direction)
            system += projector
            rhs += projector @ origin
        recovered = np.linalg.solve(system, rhs)
        self.assertLess(np.linalg.norm(recovered - target), 1e-9)
        self.assertTrue(math.isfinite(float(recovered[0])))


if __name__ == "__main__":
    unittest.main()
