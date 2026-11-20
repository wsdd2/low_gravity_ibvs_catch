from __future__ import annotations

import time

from space_grasp.control.visibility_supervisor import VisibilityDecision
from space_grasp.estimation.messages import BallObservation, TargetEstimate
from space_grasp.evaluation.oracle_metrics import OracleMetrics


class StructuredLogger:
    def __init__(self, interval_s: float = 0.6) -> None:
        self.interval_s = float(interval_s)
        self._last = 0.0

    def should_emit(self) -> bool:
        now = time.monotonic()
        if now - self._last < self.interval_s:
            return False
        self._last = now
        return True

    def log_bundle(
        self,
        observation: BallObservation | None,
        estimate: TargetEstimate | None,
        oracle: OracleMetrics | None,
        state: str,
        position_error: float | None,
        relative_speed: float | None,
        visibility: VisibilityDecision | None,
        close_allowed: bool,
    ) -> None:
        if not self.should_emit():
            return
        if observation is not None:
            print(
                "[vision] "
                f"t={observation.timestamp:.3f} camera={observation.camera_id} "
                f"center_uv=({observation.center_uv[0]:.1f},{observation.center_uv[1]:.1f}) "
                f"radius_px={observation.radius_px:.1f} "
                f"confidence={observation.confidence:.2f} "
                f"fov_margin={observation.fov_margin_px():.1f}"
            )
        if estimate is not None:
            print(
                "[estimate] "
                f"source={estimate.source} valid={estimate.valid} "
                f"p_hat={estimate.position.round(3)} v_hat={estimate.velocity.round(3)} "
                f"confidence={estimate.confidence:.2f} "
                f"measurement_age={estimate.measurement_age:.3f} "
                f"position_std={estimate.position_std_norm:.3f}"
            )
        if oracle is not None:
            print(
                "[oracle-metrics] "
                f"p_gt={np_round(oracle.p_gt)} v_gt={np_round(oracle.v_gt)} "
                f"p_error={oracle.p_error:.3f} v_error={oracle.v_error:.3f}"
            )
        vis = visibility
        print(
            "[control] "
            f"state={state} "
            f"position_error={None if position_error is None else round(position_error, 3)} "
            f"relative_speed={None if relative_speed is None else round(relative_speed, 3)} "
            f"fov_margin={None if vis is None else round(vis.margin_px, 1)} "
            f"close_allowed={close_allowed}"
        )


def np_round(values) -> str:
    return "[" + " ".join(f"{float(value):.3f}" for value in values) + "]"
