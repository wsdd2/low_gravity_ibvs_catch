from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from space_grasp.estimation.messages import TargetEstimate


@dataclass
class OracleMetrics:
    timestamp: float
    p_gt: list[float]
    v_gt: list[float]
    p_hat: list[float]
    v_hat: list[float]
    p_error: float
    v_error: float

    @classmethod
    def from_states(
        cls,
        ground_truth_position: np.ndarray,
        ground_truth_velocity: np.ndarray | None,
        estimate: TargetEstimate | None,
    ) -> "OracleMetrics":
        p_gt = np.asarray(ground_truth_position, dtype=np.float64)
        v_gt = (
            np.asarray(ground_truth_velocity, dtype=np.float64)
            if ground_truth_velocity is not None
            else np.zeros(3)
        )
        if estimate is None or not estimate.valid:
            p_hat = np.full(3, np.nan)
            v_hat = np.full(3, np.nan)
            p_error = float("nan")
            v_error = float("nan")
        else:
            p_hat = np.asarray(estimate.position, dtype=np.float64)
            v_hat = np.asarray(estimate.velocity, dtype=np.float64)
            p_error = float(np.linalg.norm(p_hat - p_gt))
            v_error = float(np.linalg.norm(v_hat - v_gt))
        return cls(
            timestamp=time.monotonic(),
            p_gt=p_gt.tolist(),
            v_gt=v_gt.tolist(),
            p_hat=p_hat.tolist(),
            v_hat=v_hat.tolist(),
            p_error=p_error,
            v_error=v_error,
        )


@dataclass
class EpisodeRecorder:
    output_path: Path
    episode_id: int = 0
    started_at: float = 0.0
    samples: list[dict[str, float]] = field(default_factory=list)
    visible_frames: int = 0
    total_frames: int = 0
    retry_count: int = 0

    def start(self, episode_id: int) -> None:
        self.episode_id = episode_id
        self.started_at = time.monotonic()
        self.samples.clear()
        self.visible_frames = 0
        self.total_frames = 0
        self.retry_count = 0

    def observe(self, visible: bool, p_error: float | None = None) -> None:
        self.total_frames += 1
        if visible:
            self.visible_frames += 1
        if p_error is not None and np.isfinite(p_error):
            self.samples.append({"p_error": float(p_error)})

    def finish(self, result: str, reason: str, extra: dict[str, object] | None = None) -> None:
        errors = [sample["p_error"] for sample in self.samples]
        payload = {
            "episode_id": self.episode_id,
            "result": result,
            "reason": reason,
            "duration_s": time.monotonic() - self.started_at if self.started_at else 0.0,
            "visible_frame_ratio": (
                self.visible_frames / self.total_frames if self.total_frames else 0.0
            ),
            "position_rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            "retry_count": self.retry_count,
        }
        if extra:
            payload.update(extra)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
