import unittest

import numpy as np

from space_grasp.control.visibility_supervisor import (
    VisibilitySupervisor,
    recenter_image_offset,
)
from space_grasp.estimation.messages import BallObservation


def _observation(u: float, v: float, radius: float = 20.0) -> BallObservation:
    return BallObservation(
        timestamp=0.0,
        camera_id="wrist",
        center_uv=np.array([u, v], dtype=np.float64),
        radius_px=radius,
        confidence=0.9,
        image_size=(640, 480),
    )


class VisibilityMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.supervisor = VisibilitySupervisor(warning_margin_px=40, critical_margin_px=20)

    def test_center_is_normal(self) -> None:
        decision = self.supervisor.evaluate([_observation(320, 240)])
        self.assertEqual(decision.mode, "NORMAL")
        self.assertTrue(decision.close_allowed)
        self.assertGreater(decision.margin_px, 40)

    def test_near_edge_triggers_recovery(self) -> None:
        decision = self.supervisor.evaluate([_observation(30, 240, radius=20)])
        self.assertEqual(decision.mode, "VISIBILITY_RECOVERY")
        self.assertFalse(decision.close_allowed)

    def test_lost_without_observations(self) -> None:
        decision = self.supervisor.evaluate([])
        self.assertEqual(decision.mode, "LOST")

    def test_recenter_pushes_toward_center(self) -> None:
        offset_x, offset_y = recenter_image_offset(_observation(600, 240), gain=0.5)
        self.assertLess(offset_x, 0.0)
        self.assertAlmostEqual(offset_y, 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
