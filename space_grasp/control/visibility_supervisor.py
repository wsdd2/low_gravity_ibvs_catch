from __future__ import annotations

from dataclasses import dataclass

from space_grasp.estimation.messages import BallObservation


@dataclass
class VisibilityDecision:
    mode: str
    margin_px: float
    camera_id: str | None
    close_allowed: bool
    slow_down: bool


class VisibilitySupervisor:
    """Keep the target inside a safe image margin before closing."""

    def __init__(self, warning_margin_px: float, critical_margin_px: float) -> None:
        self.warning_margin_px = float(warning_margin_px)
        self.critical_margin_px = float(critical_margin_px)

    def evaluate(self, observations: list[BallObservation]) -> VisibilityDecision:
        if not observations:
            return VisibilityDecision(
                mode="LOST",
                margin_px=-1.0,
                camera_id=None,
                close_allowed=False,
                slow_down=True,
            )
        best = min(observations, key=lambda item: item.fov_margin_px())
        margin = best.fov_margin_px()
        if margin < self.critical_margin_px:
            mode = "VISIBILITY_RECOVERY"
        elif margin < self.warning_margin_px:
            mode = "RECENTER_AND_SLOW_DOWN"
        else:
            mode = "NORMAL"
        return VisibilityDecision(
            mode=mode,
            margin_px=margin,
            camera_id=best.camera_id,
            close_allowed=mode == "NORMAL",
            slow_down=mode != "NORMAL",
        )


def recenter_image_offset(
    observation: BallObservation,
    gain: float,
    max_step: float = 0.03,
) -> tuple[float, float]:
    """Return a camera-plane offset that pushes the ball toward the image center."""
    width, height = observation.image_size
    error_x = (float(observation.center_uv[0]) - width * 0.5) / max(width, 1)
    error_y = (float(observation.center_uv[1]) - height * 0.5) / max(height, 1)
    offset_x = max(-max_step, min(max_step, -gain * error_x * max_step * 8.0))
    offset_y = max(-max_step, min(max_step, -gain * error_y * max_step * 8.0))
    return offset_x, offset_y
