from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BallObservation:
    timestamp: float
    camera_id: str
    center_uv: np.ndarray
    radius_px: float
    confidence: float
    image_size: tuple[int, int]
    source: str = "vision"

    def fov_margin_px(self) -> float:
        width, height = self.image_size
        u, v = float(self.center_uv[0]), float(self.center_uv[1])
        radius = float(self.radius_px)
        return min(u - radius, width - u - radius, v - radius, height - v - radius)


@dataclass
class TargetEstimate:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray | None = None
    confidence: float = 0.0
    source: str = "vision"
    valid: bool = False
    measurement_age: float = 0.0
    camera_id: str = "stereo"
    position_std: np.ndarray | None = None
    extras: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source == "oracle":
            raise AssertionError("control TargetEstimate.source must not be oracle")

    @property
    def position_std_norm(self) -> float:
        if self.position_std is None:
            return float("inf")
        return float(np.linalg.norm(self.position_std))
