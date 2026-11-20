"""Isaac Sim 6.1 zero-gravity UR5e scene for stereo visual catching."""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from collections import deque
from pathlib import Path

from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from space_grasp.config import load_experiment_config

CONFIG = load_experiment_config(ROOT / "configs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without the Isaac Sim window.")
    parser.add_argument("--seed", type=int, default=7, help="Random episode seed.")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Simulation seconds; 0 keeps running until the window closes.",
    )
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument(
        "--render-hz",
        type=int,
        default=int(CONFIG.perception.fps),
        help="GUI and RGB viewport refresh rate.",
    )
    parser.add_argument("--teleop-linear-speed", type=float, default=0.12, help="End-effector translation speed in m/s.")
    parser.add_argument(
        "--teleop-angular-speed",
        type=float,
        default=45.0,
        help="End-effector rotation speed in degrees/s.",
    )
    parser.add_argument("--disable-teleop", action="store_true", help="Disable keyboard end-effector control.")
    parser.add_argument(
        "--auto-grasp",
        action="store_true",
        help="Visually locate, track, and grasp the ball with stereo + wrist cameras.",
    )
    parser.add_argument(
        "--cheats",
        action="store_true",
        help="Oracle baseline: drive the gripper from simulator ground truth. Isolated from the vision path.",
    )
    parser.add_argument(
        "--oracle-baseline",
        action="store_true",
        help="Alias for --cheats; keep the ground-truth controller for upper-bound comparison.",
    )
    parser.add_argument(
        "--grasp-timeout",
        type=float,
        default=float(CONFIG.episode.timeout_s),
        help="Seconds to keep chasing before giving up.",
    )
    parser.add_argument("--ball-radius", type=float, default=0.045, help="Ball radius in meters.")
    parser.add_argument("--speed", type=float, default=0.035, help="Ball crossing speed in m/s.")
    parser.add_argument(
        "--despawn-margin",
        type=float,
        default=0.12,
        help="Distance beyond the workspace boundary before invalidation, in meters.",
    )
    parser.add_argument(
        "--respawn-delay",
        type=float,
        default=1.0,
        help="Delay between one target leaving and the next target spawning, in seconds.",
    )
    parser.add_argument("--single-pass", action="store_true", help="Do not respawn after the first target exits.")
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=None,
        help="Optional directory for two stereo and one wrist RGB stream.",
    )
    parser.add_argument(
        "--save-usd",
        type=Path,
        default=Path("outputs/space_grasp_scene.usda"),
        help="Export path for the authored scene before simulation.",
    )
    args, remaining_args = parser.parse_known_args()
    # Prevent scene-specific flags such as --duration from being interpreted
    # a second time by Omniverse Kit when SimulationApp starts.
    sys.argv = [sys.argv[0], *remaining_args]
    if args.physics_hz <= 0 or args.render_hz <= 0:
        parser.error("physics-hz and render-hz must be positive")
    if args.teleop_linear_speed <= 0 or args.teleop_angular_speed <= 0:
        parser.error("teleop speeds must be positive")
    if args.ball_radius <= 0 or args.ball_radius >= 0.15:
        parser.error("ball-radius must be greater than 0 and less than 0.15 m")
    if args.speed < 0:
        parser.error("speed cannot be negative")
    if args.oracle_baseline:
        args.cheats = True
    if args.grasp_timeout <= 0:
        parser.error("grasp-timeout must be positive")
    if args.despawn_margin <= 0:
        parser.error("despawn-margin must be positive")
    if args.respawn_delay < 0:
        parser.error("respawn-delay cannot be negative")
    return args


ARGS = parse_args()
APP = SimulationApp(
    {
        "headless": ARGS.headless,
        "width": 1280,
        "height": 720,
        "anti_aliasing": 2,
    }
)

# Omniverse/Isaac imports must happen after SimulationApp is created.
import numpy as np
import carb
import cv2
import carb.input
import omni.appwindow
import omni.timeline
import omni.usd
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.rendering_manager import RenderingManager, ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

app_utils.enable_extension("isaacsim.robot_motion.motion_generation")
APP.update()
import isaacsim.robot_motion.motion_generation.interface_config_loader as interface_config_loader
from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices, rot_matrices_to_quats
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from space_grasp.control.visibility_supervisor import (
    VisibilitySupervisor,
    recenter_image_offset,
)
from space_grasp.estimation.messages import BallObservation, TargetEstimate
from space_grasp.evaluation.oracle_metrics import EpisodeRecorder, OracleMetrics
from space_grasp.logging.structured_logger import StructuredLogger

CAMERA_WIDTH = int(CONFIG.cameras.width)
CAMERA_HEIGHT = int(CONFIG.cameras.height)


def disable_dlss() -> None:
    if bool(getattr(CONFIG.cameras, "use_dlss", False)) or bool(
        getattr(CONFIG.perception, "use_dlss", False)
    ):
        return
    settings = carb.settings.get_settings()
    for key, value in (
        ("/rtx/post/dlss/enabled", False),
        ("/rtx-transient/dlssg/enabled", False),
        ("/rtx/post/aa/algo", 0),
    ):
        try:
            settings.set(key, value)
        except Exception:
            continue
    print("[space-grasp] DLSS disabled for stereo-consistent ball edges")


disable_dlss()


TARGET_PATH = "/World/Target"
CAMERA_PATHS = [
    "/World/Cameras/StereoLeft",
    "/World/Cameras/StereoRight",
]
WRIST_CAMERA_PATH = "/World/Cameras/WristCamera"
ROBOT_PATH = "/World/UR5e"
ROBOT_TOOL_LINK_PATH = f"{ROBOT_PATH}/wrist_3_link"
CAMERA_VIEWPORT_SPECS = (
    ("/World/Cameras/StereoLeft", "Stereo Left"),
    ("/World/Cameras/StereoRight", "Stereo Right"),
    ("/World/Cameras/WristCamera", "Wrist RGB"),
)
WORKSPACE_MIN = np.array([0.08, -0.45, 0.05], dtype=np.float32)
WORKSPACE_MAX = np.array([0.92, 0.45, 0.88], dtype=np.float32)
LAUNCH_CENTER = np.array([0.22, 0.0, 0.14], dtype=np.float32)
CAPTURE_CENTER = np.array([0.56, 0.0, 0.42], dtype=np.float32)
CAPTURE_HALF_SIZE = np.array([0.08, 0.18, 0.15], dtype=np.float32)
# UR5e reach is ~0.85 m. Leave room for the 2F-85 offset (~0.13 m) and the base column.
REACHABLE_MIN = np.array([0.38, -0.22, 0.22], dtype=np.float32)
REACHABLE_MAX = np.array([0.70, 0.22, 0.52], dtype=np.float32)
REACHABLE_RADIAL = (0.34, 0.76)
DEFAULT_BALL_RADIUS = 0.045
BALL_RADIUS = ARGS.ball_radius
BALL_MASS = 0.18 * (BALL_RADIUS / DEFAULT_BALL_RADIUS) ** 3
DECK_CENTER = np.array([0.35, 0.0, -0.04], dtype=np.float32)
DECK_SIZE = np.array([1.20, 0.90, 0.08], dtype=np.float32)
DECK_MIN = DECK_CENTER - DECK_SIZE * 0.5
DECK_MAX = DECK_CENTER + DECK_SIZE * 0.5
READY_ARM_POSE = np.array([0.0, -1.35, 1.75, -1.95, -1.57, 0.0], dtype=np.float32)
GRIPPER_OPEN_POSITION = 0.0
GRIPPER_CLOSED_POSITION = 0.80
# 2F-85 stroke is about 85 mm. Close to a slight squeeze on the current ball.
GRIPPER_SQUEEZE_POSITION = float(
    np.clip(0.80 * (1.0 - (2.0 * BALL_RADIUS * 0.90) / 0.085), 0.38, 0.70)
)
PREGRASP_OFFSET = 0.22
GRASP_FRAME_OFFSET = 0.129
WRIST_CAMERA_MOUNT_TRANSLATION = np.array([0.0, 0.24, 0.10], dtype=np.float64)
WRIST_CAMERA_LOOK_AT = np.array([0.0, 0.0, PREGRASP_OFFSET], dtype=np.float64)
GRASP_TOOL_ROTATION = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)
FINGER_LINK_PATHS = (
    f"{ROBOT_PATH}/Gripper/Robotiq_2F_85/left_inner_finger",
    f"{ROBOT_PATH}/Gripper/Robotiq_2F_85/right_inner_finger",
)
FINGER_PAD_LOCAL = (
    np.array([0.0, -0.030, 0.118], dtype=np.float64),
    np.array([0.0, 0.030, 0.118], dtype=np.float64),
)


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions."""
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    result = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def axis_angle_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    half_angle = angle * 0.5
    return np.concatenate(([math.cos(half_angle)], axis * math.sin(half_angle)))


def quaternion_slerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    if dot < 0.0:
        end = -end
        dot = -dot
    if dot > 0.9995:
        result = start + fraction * (end - start)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    return (
        math.sin((1.0 - fraction) * theta) * start + math.sin(fraction * theta) * end
    ) / math.sin(theta)


def quaternion_angle(lhs: np.ndarray, rhs: np.ndarray) -> float:
    dot = abs(float(np.clip(np.dot(lhs / np.linalg.norm(lhs), rhs / np.linalg.norm(rhs)), -1.0, 1.0)))
    return 2.0 * math.acos(dot)


def wrist_camera_to_tool_rotation() -> np.ndarray:
    forward = WRIST_CAMERA_LOOK_AT - WRIST_CAMERA_MOUNT_TRANSLATION
    forward /= np.linalg.norm(forward)
    up_reference = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_reference)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.column_stack((right, up, -forward))


class WristCameraFollower:
    """Drive a world-space camera from the measured UR5e tool pose."""

    def __init__(self, stage, articulation: Articulation, camera_path: str) -> None:
        self.articulation = articulation
        config = interface_config_loader.load_supported_lula_kinematics_solver_config("UR5e")
        self.solver = LulaKinematicsSolver(**config)
        self.solver.set_robot_base_pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        self.arm_joint_names = self.solver.get_joint_names()
        self.arm_dof_indices = self.articulation.get_dof_indices(self.arm_joint_names).numpy()
        self.hand_frame = "tool0"
        self.mount_translation = WRIST_CAMERA_MOUNT_TRANSLATION

        camera_prim = stage.GetPrimAtPath(camera_path)
        xform = UsdGeom.Xformable(camera_prim)
        self.transform_op = xform.AddTransformOp()

    def update(self) -> None:
        joint_positions = self.articulation.get_dof_positions(
            dof_indices=self.arm_dof_indices
        ).numpy()[0]
        hand_position, hand_rotation = self.solver.compute_forward_kinematics(
            self.hand_frame, joint_positions
        )
        camera_position = hand_position + hand_rotation @ self.mount_translation
        camera_target = hand_position + hand_rotation @ WRIST_CAMERA_LOOK_AT
        self.transform_op.Set(
            camera_transform(
                tuple(float(value) for value in camera_position),
                tuple(float(value) for value in camera_target),
            )
        )


class ManipulatorKeyboardTeleop:
    """Keyboard Cartesian teleoperation backed by Lula inverse kinematics."""

    TRANSLATION_KEYS = {
        "W": np.array([1.0, 0.0, 0.0]),
        "S": np.array([-1.0, 0.0, 0.0]),
        "A": np.array([0.0, 1.0, 0.0]),
        "D": np.array([0.0, -1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
        "X": np.array([0.0, 0.0, -1.0]),
    }
    ROTATION_KEYS = {
        "I": np.array([1.0, 0.0, 0.0]),
        "K": np.array([-1.0, 0.0, 0.0]),
        "J": np.array([0.0, 1.0, 0.0]),
        "L": np.array([0.0, -1.0, 0.0]),
        "U": np.array([0.0, 0.0, 1.0]),
        "O": np.array([0.0, 0.0, -1.0]),
    }

    def __init__(
        self,
        articulation: Articulation,
        solver: LulaKinematicsSolver | None = None,
        arm_dof_indices: np.ndarray | None = None,
    ) -> None:
        self.articulation = articulation
        if solver is None:
            config = interface_config_loader.load_supported_lula_kinematics_solver_config("UR5e")
            solver = LulaKinematicsSolver(**config)
            solver.set_robot_base_pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        self.solver = solver
        self.end_effector_frame = "tool0"
        self.arm_joint_names = self.solver.get_joint_names()
        self.arm_dof_indices = (
            arm_dof_indices
            if arm_dof_indices is not None
            else self.articulation.get_dof_indices(self.arm_joint_names).numpy()
        )
        self.finger_dof_indices = self.articulation.get_dof_indices(["finger_joint"]).numpy()
        self.ready_arm_positions = READY_ARM_POSE.copy()
        self.target_position, target_rotation = self.solver.compute_forward_kinematics(
            self.end_effector_frame, self.ready_arm_positions
        )
        self.target_orientation = rot_matrices_to_quats(target_rotation)
        self.pressed_keys: set[str] = set()
        self.reset_requested = False
        self.gripper_target: float | None = None
        self.last_ik_warning_time = 0.0

        app_window = omni.appwindow.get_default_app_window()
        self.input_interface = carb.input.acquire_input_interface()
        self.keyboard = app_window.get_keyboard()
        self.keyboard_subscription = self.input_interface.subscribe_to_keyboard_events(
            self.keyboard, self._on_keyboard_event
        )

        print(
            "[space-grasp] UR5e keyboard teleop enabled\n"
            "  Move: W/S = +/-X, A/D = +/-Y, Z/X = +/-Z\n"
            "  Local RPY: I/K = +/-Roll, J/L = +/-Pitch, U/O = +/-Yaw\n"
            "  Gripper: V = open, C = close | T = reset arm | Space = stop"
        )

    def _on_keyboard_event(self, event, *_args) -> bool:
        raw_input = event.input
        key_name = (
            raw_input if isinstance(raw_input, str) else getattr(raw_input, "name", str(raw_input))
        ).upper()
        if event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            if key_name == "T" and event.type == carb.input.KeyboardEventType.KEY_PRESS:
                self.reset_requested = True
            elif key_name == "V" and event.type == carb.input.KeyboardEventType.KEY_PRESS:
                self.gripper_target = GRIPPER_OPEN_POSITION
            elif key_name == "C" and event.type == carb.input.KeyboardEventType.KEY_PRESS:
                self.gripper_target = GRIPPER_CLOSED_POSITION
            elif key_name == "SPACE":
                self.pressed_keys.clear()
            else:
                self.pressed_keys.add(key_name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.pressed_keys.discard(key_name)
        return True

    @staticmethod
    def _normalized_key_vector(keys: set[str], mapping: dict[str, np.ndarray]) -> np.ndarray:
        vector = sum((mapping[key] for key in keys if key in mapping), start=np.zeros(3))
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1.0 else vector

    def update(self, elapsed_time: float) -> None:
        if self.reset_requested:
            self.articulation.set_dof_position_targets(
                np.asarray([READY_ARM_POSE], dtype=np.float32),
                dof_indices=self.arm_dof_indices,
            )
            self.target_position, target_rotation = self.solver.compute_forward_kinematics(
                self.end_effector_frame, self.ready_arm_positions
            )
            self.target_orientation = rot_matrices_to_quats(target_rotation)
            self.reset_requested = False
            print("[space-grasp] UR5e teleop reset")

        if self.gripper_target is not None:
            targets = np.full((1, 1), self.gripper_target, dtype=np.float32)
            self.articulation.set_dof_position_targets(
                targets, dof_indices=self.finger_dof_indices
            )
            print(
                f"[space-grasp] gripper "
                f"{'open' if self.gripper_target == GRIPPER_OPEN_POSITION else 'closed'}"
            )
            self.gripper_target = None

        translation = self._normalized_key_vector(self.pressed_keys, self.TRANSLATION_KEYS)
        rotation = self._normalized_key_vector(self.pressed_keys, self.ROTATION_KEYS)
        if not np.any(translation) and not np.any(rotation):
            return

        previous_position = self.target_position.copy()
        previous_orientation = self.target_orientation.copy()
        self.target_position = np.clip(
            self.target_position + translation * ARGS.teleop_linear_speed * elapsed_time,
            np.array([0.12, -0.70, 0.10]),
            np.array([0.90, 0.70, 1.00]),
        )
        rotation_norm = np.linalg.norm(rotation)
        if rotation_norm > 0:
            delta_orientation = axis_angle_quaternion(
                rotation / rotation_norm,
                math.radians(ARGS.teleop_angular_speed) * elapsed_time * rotation_norm,
            )
            self.target_orientation = quaternion_multiply(
                self.target_orientation, delta_orientation
            )

        current_positions = self.articulation.get_dof_positions(
            dof_indices=self.arm_dof_indices
        ).numpy()[0]
        joint_targets, success = self.solver.compute_inverse_kinematics(
            self.end_effector_frame,
            self.target_position,
            self.target_orientation,
            warm_start=current_positions,
            position_tolerance=0.003,
            orientation_tolerance=0.04,
        )
        if success:
            self.articulation.set_dof_position_targets(
                np.asarray([joint_targets], dtype=np.float32),
                dof_indices=self.arm_dof_indices,
            )
        else:
            self.target_position = previous_position
            self.target_orientation = previous_orientation
            now = time.monotonic()
            if now - self.last_ik_warning_time > 1.0:
                print("[space-grasp] IK did not converge; command ignored")
                self.last_ik_warning_time = now

    def close(self) -> None:
        if self.keyboard_subscription is not None:
            self.input_interface.unsubscribe_to_keyboard_events(
                self.keyboard, self.keyboard_subscription
            )
            self.keyboard_subscription = None


class SceneResetHotkey:
    """Expose a GUI hotkey for resetting the full robot/target episode."""

    def __init__(self) -> None:
        self.requested = False
        app_window = omni.appwindow.get_default_app_window()
        self.input_interface = carb.input.acquire_input_interface()
        self.keyboard = app_window.get_keyboard()
        self.subscription = self.input_interface.subscribe_to_keyboard_events(
            self.keyboard, self._on_keyboard_event
        )

    def _on_keyboard_event(self, event, *_args) -> bool:
        raw_input = event.input
        key_name = (
            raw_input if isinstance(raw_input, str) else getattr(raw_input, "name", str(raw_input))
        ).upper()
        if key_name == "B" and event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.requested = True
        return True

    def consume(self) -> bool:
        requested = self.requested
        self.requested = False
        return requested

    def close(self) -> None:
        if self.subscription is not None:
            self.input_interface.unsubscribe_to_keyboard_events(
                self.keyboard, self.subscription
            )
            self.subscription = None


class MultiCameraBallTracker:
    """Triangulate the green textured ball from the external stereo pair."""

    def __init__(self, stage, camera_paths: list[str], resolution: tuple[int, int] = (640, 480)) -> None:
        import omni.replicator.core as rep

        self.stage = stage
        self.camera_paths = camera_paths
        self.resolution = resolution or (CAMERA_WIDTH, CAMERA_HEIGHT)
        self.render_products = []
        self.annotators = []
        for camera_path in camera_paths:
            render_product = rep.create.render_product(camera_path, resolution)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach(render_product)
            self.render_products.append(render_product)
            self.annotators.append(annotator)

    @staticmethod
    def _detect_ball_features(
        frame, allow_bright_fallback: bool = False
    ) -> tuple[np.ndarray, float] | None:
        if isinstance(frame, dict):
            frame = frame.get("data")
        if frame is None:
            return None
        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.size == 0:
            return None
        rgb = rgb[:, :, :3].astype(np.float32)
        if rgb.max() <= 1.5:
            rgb *= 255.0
        red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        mask = (
            (green > 75)
            & (green > red * 1.45)
            & (green > blue * 1.20)
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = [
            index
            for index in range(1, count)
            if 5 <= stats[index, cv2.CC_STAT_AREA] <= 6000
        ]
        if not candidates and allow_bright_fallback:
            bright_mask = (
                (red > 180) & (green > 180) & (blue > 180)
            ).astype(np.uint8)
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(
                bright_mask, connectivity=8
            )
            candidates = [
                index
                for index in range(1, count)
                if 20 <= stats[index, cv2.CC_STAT_AREA] <= 1500
            ]
        if not candidates:
            return None
        best = max(candidates, key=lambda index: stats[index, cv2.CC_STAT_AREA])
        component_yx = np.argwhere(labels == best)
        component_xy = component_yx[:, ::-1].astype(np.float32)
        _, radius = cv2.minEnclosingCircle(component_xy)
        pixel = centroids[best].astype(np.float64)
        if radius <= 1.0:
            radius = math.sqrt(stats[best, cv2.CC_STAT_AREA] / math.pi)
        return pixel, float(radius)

    @classmethod
    def _detect_ball_pixel(cls, frame) -> np.ndarray | None:
        features = cls._detect_ball_features(frame)
        return features[0] if features is not None else None

    def _pixel_to_world_ray(self, camera_path: str, pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(camera_path))
        width, height = self.resolution
        focal_length = float(camera.GetFocalLengthAttr().Get())
        horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
        vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
        fx = width * focal_length / horizontal_aperture
        fy = height * focal_length / vertical_aperture
        local_direction = Gf.Vec3d(
            (float(pixel[0]) - width * 0.5) / fx,
            -(float(pixel[1]) - height * 0.5) / fy,
            -1.0,
        )
        world_transform = UsdGeom.Xformable(camera.GetPrim()).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        local_ray = Gf.Vec3d(
            (float(pixel[0]) - width * 0.5) / fx,
            -(float(pixel[1]) - height * 0.5) / fy,
            -1.0,
        )
        ray_origin = np.array(world_transform.ExtractTranslation(), dtype=np.float64)
        ray_direction = np.array(
            world_transform.TransformDir(local_ray).GetNormalized(),
            dtype=np.float64,
        )
        origin_gf = world_transform.ExtractTranslation()
        direction_gf = world_transform.TransformDir(local_direction).GetNormalized()
        return (
            np.array(origin_gf, dtype=np.float64),
            np.array(direction_gf, dtype=np.float64),
        )

    def estimate(self) -> tuple[np.ndarray, list[BallObservation]] | None:
        rays = []
        observations: list[BallObservation] = []
        stamp = time.monotonic()
        max_residual = float(CONFIG.perception.max_stereo_residual_m)
        for camera_path, annotator in zip(self.camera_paths, self.annotators):
            features = self._detect_ball_features(annotator.get_data())
            if features is None:
                continue
            pixel, radius = features
            observations.append(
                BallObservation(
                    timestamp=stamp,
                    camera_id=camera_path.rsplit("/", 1)[-1],
                    center_uv=np.asarray(pixel, dtype=np.float64),
                    radius_px=float(radius),
                    confidence=0.85,
                    image_size=self.resolution,
                )
            )
            rays.append(self._pixel_to_world_ray(camera_path, pixel))
        if len(rays) < 2:
            return None

        identity = np.eye(3)
        system = np.zeros((3, 3), dtype=np.float64)
        rhs = np.zeros(3, dtype=np.float64)
        for origin, direction in rays:
            projector = identity - np.outer(direction, direction)
            system += projector
            rhs += projector @ origin
        try:
            position = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            return None

        residuals = [
            np.linalg.norm(np.cross(position - origin, direction))
            for origin, direction in rays
        ]
        if max(residuals) > max_residual:
            return None
        return position.astype(np.float32), observations

    def close(self) -> None:
        for annotator in self.annotators:
            annotator.detach()
        self.render_products.clear()


class WristImageTracker:
    """Extract wrist-camera target features and a world-space bearing ray."""

    def __init__(self, stage, camera_path: str, resolution: tuple[int, int] = (640, 480)) -> None:
        import omni.replicator.core as rep

        self.stage = stage
        self.camera_path = camera_path
        self.resolution = resolution or (CAMERA_WIDTH, CAMERA_HEIGHT)
        self.render_product = rep.create.render_product(camera_path, resolution)
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach(self.render_product)
        self.missed_frames = 0
        self.debug_frame_saved = False

    def observe(self) -> dict[str, object] | None:
        frame = self.annotator.get_data()
        features = MultiCameraBallTracker._detect_ball_features(frame)
        if features is None:
            self.missed_frames += 1
            if self.missed_frames >= 10 and not self.debug_frame_saved:
                raw = frame.get("data") if isinstance(frame, dict) else frame
                image = np.asarray(raw)
                if image.ndim == 3 and image.shape[2] >= 3:
                    output_path = Path("outputs/wrist_ibvs_no_detection.png")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(output_path),
                        cv2.cvtColor(image[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR),
                    )
                    print(f"[visual-grasp] wrist detection failed; saved {output_path}")
                    self.debug_frame_saved = True
            return None
        self.missed_frames = 0
        pixel, radius = features
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(self.camera_path))
        width, height = self.resolution
        focal_length = float(camera.GetFocalLengthAttr().Get())
        fx = width * focal_length / float(camera.GetHorizontalApertureAttr().Get())
        fy = height * focal_length / float(camera.GetVerticalApertureAttr().Get())
        depth = fx * BALL_RADIUS / max(radius, 1.0)

        world_transform = UsdGeom.Xformable(camera.GetPrim()).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        local_ray = Gf.Vec3d(
            (float(pixel[0]) - width * 0.5) / fx,
            -(float(pixel[1]) - height * 0.5) / fy,
            -1.0,
        )
        ray_origin = np.array(world_transform.ExtractTranslation(), dtype=np.float64)
        ray_direction = np.array(
            world_transform.TransformDir(local_ray).GetNormalized(),
            dtype=np.float64,
        )
        camera_to_world = GRASP_TOOL_ROTATION @ wrist_camera_to_tool_rotation()
        right = camera_to_world[:, 0].copy()
        down = camera_to_world[:, 1].copy()
        forward = -camera_to_world[:, 2].copy()
        right /= np.linalg.norm(right)
        down /= np.linalg.norm(down)
        forward /= np.linalg.norm(forward)
        return {
            "x": (float(pixel[0]) - width * 0.5) / fx,
            "y": (float(pixel[1]) - height * 0.5) / fy,
            "depth": float(depth),
            "pixel": pixel,
            "radius": radius,
            "right": right,
            "down": down,
            "forward": forward,
            "world_transform": world_transform,
            "ray_origin": ray_origin,
            "ray_direction": ray_direction,
            "image": np.asarray(frame.get("data") if isinstance(frame, dict) else frame),
            "observation": BallObservation(
                timestamp=time.monotonic(),
                camera_id="wrist",
                center_uv=np.asarray(pixel, dtype=np.float64),
                radius_px=float(radius),
                confidence=0.9,
                image_size=self.resolution,
            ),
        }

    def close(self) -> None:
        self.annotator.detach()
        self.render_product = None


class VisualGraspController:
    """Simple static-target grasp state machine driven by RGB triangulation."""

    def __init__(
        self,
        stage,
        articulation: Articulation,
        solver: LulaKinematicsSolver,
        arm_dof_indices: np.ndarray,
    ) -> None:
        self.articulation = articulation
        self.solver = solver
        self.arm_dof_indices = arm_dof_indices
        self.finger_dof_indices = articulation.get_dof_indices(["finger_joint"]).numpy()
        self.end_effector_frame = "tool0"
        self.state = "SEARCH"
        self.estimate_history: deque[np.ndarray] = deque(maxlen=5)
        self.ball_position: np.ndarray | None = None
        self.grasp_ball_position: np.ndarray | None = None
        self.latest_lift_estimate: np.ndarray | None = None
        self.close_elapsed = 0.0
        self.verify_elapsed = 0.0
        self.retry_count = 0
        self.ibvs_stable_frames = 0
        self.ibvs_frame_count = 0
        self.ibvs_target_position: np.ndarray | None = None
        self.lift_target: np.ndarray | None = None
        self.final_approach_target: np.ndarray | None = None
        self.last_warning_time = 0.0
        self.grasp_orientation = rot_matrices_to_quats(GRASP_TOOL_ROTATION)
        self.approach_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.solver.set_default_cspace_seeds(
            np.asarray(
                [
                    READY_ARM_POSE,
                    [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
                    [0.4, -1.2, 1.6, -1.9, -1.57, 0.0],
                    [-0.4, -1.2, 1.6, -1.9, -1.57, 0.0],
                ],
                dtype=np.float64,
            )
        )

    @property
    def gripper_center_offset(self) -> float:
        return GRASP_FRAME_OFFSET

    def _desired_wrist_features(self) -> tuple[float, float, float]:
        ball_in_tool = np.array([0.0, 0.0, PREGRASP_OFFSET])
        ball_in_camera = wrist_camera_to_tool_rotation().T @ (
            ball_in_tool - WRIST_CAMERA_MOUNT_TRANSLATION
        )
        depth = -float(ball_in_camera[2])
        return (
            float(ball_in_camera[0]) / depth,
            -float(ball_in_camera[1]) / depth,
            depth,
        )

    def reset(self, full: bool = False) -> None:
        self.state = "SEARCH"
        self.estimate_history.clear()
        self.ball_position = None
        self.grasp_ball_position = None
        self.latest_lift_estimate = None
        self.close_elapsed = 0.0
        self.verify_elapsed = 0.0
        self.ibvs_stable_frames = 0
        self.ibvs_frame_count = 0
        self.ibvs_target_position = None
        self.lift_target = None
        self.final_approach_target = None
        if full:
            self.retry_count = 0
        self._set_gripper(GRIPPER_OPEN_POSITION)

    def _current_arm_positions(self) -> np.ndarray:
        return self.articulation.get_dof_positions(
            dof_indices=self.arm_dof_indices
        ).numpy()[0]

    def _current_tool_position(self) -> np.ndarray:
        position, _ = self.solver.compute_forward_kinematics(
            self.end_effector_frame, self._current_arm_positions()
        )
        return position

    def _set_gripper(self, position: float) -> None:
        self.articulation.set_dof_position_targets(
            np.asarray([[position]], dtype=np.float32),
            dof_indices=self.finger_dof_indices,
        )

    def _command_pose(self, position: np.ndarray) -> bool:
        joint_targets, success = self.solver.compute_inverse_kinematics(
            self.end_effector_frame,
            position,
            self.grasp_orientation,
            warm_start=self._current_arm_positions(),
            position_tolerance=0.001,
            orientation_tolerance=0.01,
        )
        if success:
            self.articulation.set_dof_position_targets(
                np.asarray([joint_targets], dtype=np.float32),
                dof_indices=self.arm_dof_indices,
            )
            return True
        now = time.monotonic()
        if now - self.last_warning_time > 1.0:
            print(f"[visual-grasp] IK failed in state {self.state}")
            self.last_warning_time = now
        return False

    def update(
        self,
        fixed_estimate: np.ndarray | None,
        wrist_observation: dict[str, object] | None,
        elapsed_time: float,
    ) -> None:
        if self.state == "SEARCH":
            if fixed_estimate is None:
                return
            self.estimate_history.append(fixed_estimate)
            if len(self.estimate_history) < self.estimate_history.maxlen:
                return
            samples = np.stack(self.estimate_history)
            candidate = np.median(samples, axis=0)
            if np.max(np.linalg.norm(samples - candidate, axis=1)) > 0.025:
                return
            self.ball_position = candidate
            self._set_gripper(GRIPPER_OPEN_POSITION)
            self.state = "PREGRASP"
            print(f"[visual-grasp] target triangulated at {candidate.round(3)}")
            return

        if fixed_estimate is not None and self.state in ("PREGRASP", "IBVS_ALIGN"):
            self.ball_position = 0.8 * self.ball_position + 0.2 * fixed_estimate
        elif fixed_estimate is not None and self.state in ("CLOSE", "LIFT", "VERIFY"):
            self.latest_lift_estimate = fixed_estimate

        if self.ball_position is None:
            return
        vertical_offset = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if self.state == "PREGRASP":
            target = self.ball_position - self.approach_axis * PREGRASP_OFFSET
            if self._command_pose(target) and np.linalg.norm(self._current_tool_position() - target) < 0.025:
                self.ibvs_stable_frames = 0
                self.ibvs_target_position = self._current_tool_position().copy()
                self.state = "IBVS_ALIGN"
                print("[visual-grasp] PBVS pregrasp reached; switching to wrist IBVS")
        elif self.state == "IBVS_ALIGN":
            if wrist_observation is None:
                self.ibvs_stable_frames = 0
                return
            self.ibvs_frame_count += 1
            desired_x, desired_y, _ = self._desired_wrist_features()
            error_x = float(wrist_observation["x"]) - desired_x
            error_y = float(wrist_observation["y"]) - desired_y
            current_tool = self._current_tool_position()
            feature_converged = (
                abs(error_x) < 0.005
                and abs(error_y) < 0.005
            )
            if self.ibvs_frame_count % 20 == 0:
                print(
                    f"[visual-grasp] IBVS error x={error_x:.3f} "
                    f"y={error_y:.3f}"
                )
                if self.ibvs_frame_count == 20:
                    image = np.asarray(wrist_observation["image"])
                    cv2.imwrite(
                        "outputs/wrist_ibvs_tracking.png",
                        cv2.cvtColor(
                            image[:, :, :3].astype(np.uint8),
                            cv2.COLOR_RGB2BGR,
                        ),
                    )
            if feature_converged:
                self.ibvs_stable_frames += 1
            else:
                self.ibvs_stable_frames = 0
                camera_delta = np.array(
                    [
                        0.0,
                        -0.02 * error_x,
                        -0.02 * error_y,
                    ],
                    dtype=np.float64,
                )
                delta_norm = np.linalg.norm(camera_delta)
                if delta_norm > 0.004:
                    camera_delta *= 0.004 / delta_norm
                if self.ibvs_frame_count % 20 == 0:
                    print(
                        f"[visual-grasp] IBVS tool={self._current_tool_position().round(3)} "
                        f"delta={camera_delta.round(4)}"
                    )
                self.ibvs_target_position += camera_delta
                self._command_pose(self.ibvs_target_position)
            if self.ibvs_stable_frames >= 3:
                self.final_approach_target = self._current_tool_position().copy()
                self.final_approach_target[0] = (
                    self.ball_position[0] - self.gripper_center_offset
                )
                self.state = "FINAL_APPROACH"
                print(
                    f"[visual-grasp] wrist IBVS converged; "
                    f"starting axial PBVS approach"
                )
        elif self.state == "FINAL_APPROACH":
            target = self.final_approach_target
            if self._command_pose(target) and np.linalg.norm(
                self._current_tool_position() - target
            ) < 0.008:
                self.grasp_ball_position = self.ball_position.copy()
                self._set_gripper(GRIPPER_CLOSED_POSITION)
                self.close_elapsed = 0.0
                self.state = "CLOSE"
                print(
                    f"[visual-grasp] grasp frame reached; closing gripper"
                )
        elif self.state == "CLOSE":
            self.close_elapsed += elapsed_time
            self._set_gripper(GRIPPER_CLOSED_POSITION)
            if self.close_elapsed >= 1.0:
                self.lift_target = self._current_tool_position() + vertical_offset * 0.18
                self.state = "LIFT"
        elif self.state == "LIFT":
            target = self.lift_target
            if self._command_pose(target) and np.linalg.norm(self._current_tool_position() - target) < 0.025:
                self.verify_elapsed = 0.0
                self.state = "VERIFY"
                print("[visual-grasp] lift pose reached; verifying ball motion")
        elif self.state == "VERIFY":
            self.verify_elapsed += elapsed_time
            if self.latest_lift_estimate is not None:
                expected_ball = (
                    self._current_tool_position()
                    + self.approach_axis * self.gripper_center_offset
                )
                lifted = (
                    self.latest_lift_estimate[2] > self.grasp_ball_position[2] + 0.08
                    and np.linalg.norm(self.latest_lift_estimate - expected_ball) < 0.08
                )
                if lifted:
                    self.state = "DONE"
                    print(
                        f"[visual-grasp] grasp verified at "
                        f"{self.latest_lift_estimate.round(3)}"
                    )
                    return
            if self.verify_elapsed >= 1.5:
                self.retry_count += 1
                tool_position = self._current_tool_position()
                print(
                    f"[visual-grasp] grasp failed; retry={self.retry_count} "
                    f"ball={None if self.latest_lift_estimate is None else self.latest_lift_estimate.round(3)} "
                    f"tool={tool_position.round(3)} "
                    f"gripper_dofs={self.articulation.get_dof_positions().numpy()[0, 6:].round(3)}"
                )
                self.reset()


class ConstantVelocityKalmanFilter:
    """Six-state [position, velocity] filter for zero-gravity target motion."""

    def __init__(
        self,
        acceleration_sigma: float = 0.02,
        freeze_velocity: bool = False,
    ) -> None:
        self.acceleration_sigma = acceleration_sigma
        self.freeze_velocity = freeze_velocity
        self.state = np.zeros(6, dtype=np.float64)
        self.covariance = np.eye(6, dtype=np.float64)
        self.initialized = False

    def reset(self) -> None:
        self.state.fill(0.0)
        self.covariance = np.eye(6, dtype=np.float64)
        self.initialized = False

    def _pin_velocity(self) -> None:
        self.state[3:] = 0.0
        self.covariance[3:, :] = 0.0
        self.covariance[:, 3:] = 0.0
        self.covariance[3:, 3:] = np.eye(3) * 1e-6

    def predict(self, dt: float) -> None:
        if not self.initialized:
            return
        if self.freeze_velocity:
            self._pin_velocity()
            return
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        q = self.acceleration_sigma**2
        process = np.block(
            [
                [np.eye(3) * dt**4 / 4, np.eye(3) * dt**3 / 2],
                [np.eye(3) * dt**3 / 2, np.eye(3) * dt**2],
            ]
        ) * q
        self.state = transition @ self.state
        self.covariance = (
            transition @ self.covariance @ transition.T + process
        )

    def update(self, position: np.ndarray, measurement_sigma: float) -> None:
        measurement = np.asarray(position, dtype=np.float64)
        if not self.initialized:
            self.state[:3] = measurement
            self.state[3:] = 0.0
            self.covariance = np.diag(
                [measurement_sigma**2] * 3 + [0.40**2] * 3
            )
            self.initialized = True
            if self.freeze_velocity:
                self._pin_velocity()
            return
        observation = np.zeros((3, 6))
        observation[:, :3] = np.eye(3)
        noise = np.eye(3) * measurement_sigma**2
        innovation = measurement - observation @ self.state
        innovation_covariance = (
            observation @ self.covariance @ observation.T + noise
        )
        gain = (
            self.covariance
            @ observation.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.state += gain @ innovation
        self.covariance = (
            np.eye(6) - gain @ observation
        ) @ self.covariance
        if self.freeze_velocity:
            self._pin_velocity()

    @property
    def position(self) -> np.ndarray:
        return self.state[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:].copy()


class SimulatedGripperForceSensor:
    """Pad spring contact plus finger-motor effort, used as a grasp force proxy."""

    PAD_HALF = 0.008
    STIFFNESS = 250.0
    PINCH_FORCE = float(CONFIG.gripper.min_contact_force_n)
    HOLD_TIME = float(CONFIG.gripper.confirm_duration_s)
    EFFORT_PINCH = 0.35

    def __init__(self, articulation: Articulation, finger_dof_indices: np.ndarray) -> None:
        self.articulation = articulation
        self.finger_dof_indices = finger_dof_indices
        self.hold_elapsed = 0.0

    def reset(self) -> None:
        self.hold_elapsed = 0.0

    def finger_effort(self) -> float:
        try:
            return abs(
                float(
                    self.articulation.get_dof_efforts(
                        dof_indices=self.finger_dof_indices
                    ).numpy()[0, 0]
                )
            )
        except Exception:
            return 0.0

    def finger_position(self) -> float:
        try:
            return float(
                self.articulation.get_dof_positions(
                    dof_indices=self.finger_dof_indices
                ).numpy()[0, 0]
            )
        except Exception:
            return 0.0

    def pad_forces(
        self, ball: np.ndarray, pad_points: list[np.ndarray]
    ) -> tuple[float, float]:
        forces = []
        for pad in pad_points:
            gap = float(np.linalg.norm(ball - pad) - BALL_RADIUS - self.PAD_HALF)
            forces.append(self.STIFFNESS * max(0.0, -gap))
        if len(forces) < 2:
            return 0.0, 0.0
        return float(forces[0]), float(forces[1])

    def read(
        self,
        ball: np.ndarray | None,
        pad_points: list[np.ndarray] | None,
        dt: float,
        closing: bool,
    ) -> dict[str, float | bool]:
        left = right = 0.0
        if ball is not None and pad_points is not None and len(pad_points) >= 2:
            left, right = self.pad_forces(ball, pad_points)
        effort = self.finger_effort()
        grip = self.finger_position()
        both_pads = left >= self.PINCH_FORCE and right >= self.PINCH_FORCE
        pinched = closing and both_pads and grip > 0.18
        jammed = (
            closing
            and grip > 0.18
            and effort >= self.EFFORT_PINCH
            and min(left, right) > 0.0
        )
        if pinched or jammed:
            self.hold_elapsed += dt
        else:
            self.hold_elapsed = 0.0
        return {
            "left": left,
            "right": right,
            "effort": effort,
            "grip": grip,
            "pinched": pinched or jammed,
            "confirmed": self.hold_elapsed >= self.HOLD_TIME,
            "hold": self.hold_elapsed,
        }


class StereoKalmanGraspController:
    """Stereo search, Kalman tracking, wrist-bearing fusion and PBVS capture."""

    ACQUIRE_HITS = 6
    STABLE_SPAN = 0.025
    COMMAND_STEP = 0.04
    MIN_TOOL_HEIGHT = 0.12

    def __init__(
        self,
        stage,
        articulation: Articulation,
        solver: LulaKinematicsSolver,
        arm_dof_indices: np.ndarray,
    ) -> None:
        self.stage = stage
        self.articulation = articulation
        self.solver = solver
        self.arm_dof_indices = arm_dof_indices
        self.finger_dof_indices = articulation.get_dof_indices(
            ["finger_joint"]
        ).numpy()
        self.filter = ConstantVelocityKalmanFilter(freeze_velocity=False)
        self.state = "SEARCH_STEREO"
        self.stereo_hits = 0
        self.wrist_fusions = 0
        self.frame_count = 0
        self.close_elapsed = 0.0
        self.recent_stereo: deque[np.ndarray] = deque(maxlen=8)
        self.grasp_position: np.ndarray | None = None
        self.lift_target: np.ndarray | None = None
        self.latest_stereo_position: np.ndarray | None = None
        self.last_warning_time = 0.0
        self.last_status_time = 0.0
        self.cheat_tool_to_pads: np.ndarray | None = None
        self.cheat_offset_locked = False
        self.finger_prims = [RigidPrim(path) for path in FINGER_LINK_PATHS]
        self.force_sensor = SimulatedGripperForceSensor(
            articulation, self.finger_dof_indices
        )
        self.attempt_elapsed = 0.0
        self.retry_count = 0
        self.reopen_elapsed = 0.0
        self.miss_elapsed = 0.0
        self.live_ball: np.ndarray | None = None
        self.live_velocity = np.zeros(3, dtype=np.float64)
        self.visibility = VisibilitySupervisor(
            CONFIG.visibility.warning_margin_px,
            CONFIG.visibility.critical_margin_px,
        )
        self.structured_log = StructuredLogger(CONFIG.evaluation.log_interval_s)
        self.last_estimate: TargetEstimate | None = None
        self.last_observations: list[BallObservation] = []
        self.prev_tool: np.ndarray | None = None
        self.tool_velocity = np.zeros(3, dtype=np.float64)
        self.last_visibility_mode = "NORMAL"
        self.terminal_result: str | None = None
        self.terminal_reason = ""

    @property
    def _tag(self) -> str:
        return "cheats" if ARGS.cheats else "stereo-kf"

    def reset(self, full: bool = False) -> None:
        self.filter.reset()
        self.state = "SEARCH_STEREO"
        self.stereo_hits = 0
        self.wrist_fusions = 0
        self.frame_count = 0
        self.close_elapsed = 0.0
        self.recent_stereo.clear()
        self.grasp_position = None
        self.lift_target = None
        self.cheat_tool_to_pads = None
        self.cheat_offset_locked = False
        self.latest_stereo_position = None
        self.force_sensor.reset()
        self.attempt_elapsed = 0.0
        self.reopen_elapsed = 0.0
        self.miss_elapsed = 0.0
        self.live_ball = None
        self.live_velocity[:] = 0.0
        self.last_estimate = None
        self.last_observations = []
        self.prev_tool = None
        self.tool_velocity[:] = 0.0
        self.last_visibility_mode = "NORMAL"
        self.terminal_result = None
        self.terminal_reason = ""
        if full:
            self.retry_count = 0
        self._set_gripper(GRIPPER_OPEN_POSITION)

    def _current_arm_positions(self) -> np.ndarray:
        return self.articulation.get_dof_positions(
            dof_indices=self.arm_dof_indices
        ).numpy()[0]

    def _current_tool_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position, rotation = self.solver.compute_forward_kinematics(
            "tool0", self._current_arm_positions()
        )
        rotation = np.asarray(rotation, dtype=np.float64)
        return (
            np.asarray(position, dtype=np.float64),
            rotation,
            rot_matrices_to_quats(rotation),
        )

    def _current_tool_position(self) -> np.ndarray:
        return self._current_tool_pose()[0]

    def _approach_axis(self, rotation: np.ndarray) -> np.ndarray:
        axis = np.asarray(rotation, dtype=np.float64)[:, 2]
        return axis / np.linalg.norm(axis)

    def _finger_pad_points(self) -> list[np.ndarray] | None:
        """Physics-link poses, not USD child xforms (those stay frozen under Fabric)."""
        points = []
        for prim, local_offset in zip(self.finger_prims, FINGER_PAD_LOCAL):
            try:
                positions, orientations = prim.get_world_poses()
            except Exception:
                return None
            position = positions.numpy()[0]
            rotation = quats_to_rot_matrices(orientations.numpy()[0])
            if rotation.ndim == 3:
                rotation = rotation[0]
            points.append(position + rotation @ local_offset)
        if len(points) < 2:
            return None
        return points

    def _finger_pad_midpoint(self) -> np.ndarray | None:
        points = self._finger_pad_points()
        if points is None:
            return None
        return 0.5 * (points[0] + points[1])

    def _ball_in_jaws(
        self, ball: np.ndarray, tool: np.ndarray, pads: np.ndarray
    ) -> bool:
        """True when the ball sits between the fingertips and the wrist, not past the pads."""
        offset = np.asarray(pads, dtype=np.float64) - np.asarray(tool, dtype=np.float64)
        length = float(np.linalg.norm(offset))
        if length < 1e-6:
            return float(np.linalg.norm(ball - pads)) < 0.04
        axis = offset / length
        delta = np.asarray(ball, dtype=np.float64) - np.asarray(pads, dtype=np.float64)
        along = float(np.dot(delta, -axis))
        radial = float(np.linalg.norm(delta + along * axis))
        return -0.012 <= along <= 0.09 and radial <= 0.032

    def _grasp_point(
        self,
        tool_position: np.ndarray | None = None,
        rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        measured = self._finger_pad_midpoint()
        if measured is not None:
            return measured
        if tool_position is None or rotation is None:
            tool_position, rotation, _ = self._current_tool_pose()
        return tool_position + self._approach_axis(rotation) * GRASP_FRAME_OFFSET

    def _tool_target_for_ball(self, ball: np.ndarray, standoff: float) -> np.ndarray:
        tool, rotation, _ = self._current_tool_pose()
        fallback = ball - self._approach_axis(rotation) * (GRASP_FRAME_OFFSET + standoff)
        if not self.cheat_offset_locked:
            pads = self._finger_pad_midpoint()
            self.cheat_offset_locked = True
            if pads is None:
                print(f"[{self._tag}] finger physics pose unavailable; using tool +Z 0.129 m")
                self.cheat_tool_to_pads = None
                return fallback
            offset = pads - tool
            offset_norm = float(np.linalg.norm(offset))
            print(
                f"[{self._tag}] measured tool-to-pad-mid offset={offset.round(3)} "
                f"norm={offset_norm:.3f} m pads={pads.round(3)} tool={tool.round(3)}"
            )
            if 0.04 <= offset_norm <= 0.28:
                self.cheat_tool_to_pads = offset
            else:
                print(f"[{self._tag}] pad offset out of range; using tool +Z 0.129 m")
                self.cheat_tool_to_pads = None
                return fallback
        if self.cheat_tool_to_pads is None:
            return fallback
        direction = self.cheat_tool_to_pads / np.linalg.norm(self.cheat_tool_to_pads)
        return ball - self.cheat_tool_to_pads - direction * standoff

    def _set_gripper(self, position: float) -> None:
        self.articulation.set_dof_position_targets(
            np.asarray([[position]], dtype=np.float32),
            dof_indices=self.finger_dof_indices,
        )

    def _command_pose(self, position: np.ndarray, dt: float | None = None) -> bool:
        joints = self._current_arm_positions()
        current, _, current_orientation = self._current_tool_pose()
        remaining = np.asarray(position, dtype=np.float64) - current
        distance = float(np.linalg.norm(remaining))
        if ARGS.cheats or ARGS.auto_grasp:
            if distance > 0.20:
                step = 0.024
            elif distance > 0.08:
                step = 0.018
            else:
                step = 0.012
        else:
            step = self.COMMAND_STEP
        waypoint = (
            current + remaining / distance * step
            if distance > step
            else np.asarray(position, dtype=np.float64)
        )
        waypoint[2] = max(float(waypoint[2]), self.MIN_TOOL_HEIGHT)
        self.solver.set_default_cspace_seeds(np.asarray([joints], dtype=np.float64))
        position_tolerance = min(0.008, max(0.003, step * 0.4))
        solved, success = self.solver.compute_inverse_kinematics(
            "tool0",
            waypoint,
            current_orientation,
            warm_start=joints,
            position_tolerance=position_tolerance,
            orientation_tolerance=0.08,
        )
        if success:
            self.articulation.set_dof_position_targets(
                np.asarray([solved], dtype=np.float32),
                dof_indices=self.arm_dof_indices,
            )
            return True
        now = time.monotonic()
        if now - self.last_warning_time > 1.0:
            print(
                f"[{self._tag}] IK failed in state {self.state} "
                f"tool={current.round(3)} goal={np.asarray(position).round(3)}"
            )
            self.last_warning_time = now
        return False

    def _tracked_position(self) -> np.ndarray:
        velocity = self.filter.velocity
        if float(np.linalg.norm(velocity)) < 0.02:
            return self.filter.position
        return self.filter.position + velocity * 0.20

    def _target_is_acquired(self) -> bool:
        if (
            not self.filter.initialized
            or self.miss_elapsed > 0.12
            or len(self.recent_stereo) < int(CONFIG.perception.min_valid_frames)
        ):
            return False
        residual = float(
            np.linalg.norm(self.recent_stereo[-1] - self.filter.position)
        )
        samples = np.stack(self.recent_stereo)
        stationary = float(np.max(np.ptp(samples, axis=0))) < self.STABLE_SPAN
        return residual < 0.04 or (stationary and len(self.recent_stereo) >= self.ACQUIRE_HITS)

    def _fuse_wrist_bearing(self, observation: dict[str, object]) -> bool:
        if not self.filter.initialized:
            return False
        origin = np.asarray(observation["ray_origin"])
        direction = np.asarray(observation["ray_direction"])
        predicted = self.filter.position
        distance = float(np.dot(predicted - origin, direction))
        if distance <= 0:
            return False
        wrist_point = origin + direction * distance
        if np.linalg.norm(wrist_point - predicted) > 0.08:
            return False
        self.filter.update(wrist_point, measurement_sigma=0.012)
        self.wrist_fusions += 1
        return True

    def abort(self, reason: str) -> None:
        if self.state == "DONE":
            return
        self.state = "DONE"
        self.terminal_result = "ABORT"
        self.terminal_reason = reason
        print(f"[{self._tag}] abort: {reason}; press B to reset")

    def _intercept_position(self, ball: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        pads = self._finger_pad_midpoint()
        if pads is None:
            pads = self._current_tool_position()
        distance = float(np.linalg.norm(ball - pads))
        speed = float(np.linalg.norm(velocity))
        if speed < 0.01:
            return np.asarray(ball, dtype=np.float64)
        lead = float(np.clip(distance / max(speed, 0.08), 0.04, 0.35))
        predicted = np.asarray(ball, dtype=np.float64) + np.asarray(velocity, dtype=np.float64) * lead
        return np.clip(predicted, WORKSPACE_MIN.astype(np.float64), WORKSPACE_MAX.astype(np.float64))

    def _retry_chase(self, reason: str) -> None:
        self.retry_count += 1
        self.force_sensor.reset()
        self.close_elapsed = 0.0
        self.lift_target = None
        self.grasp_position = None
        self.reopen_elapsed = 0.35
        self._set_gripper(GRIPPER_OPEN_POSITION)
        self.state = "COARSE_APPROACH"
        print(
            f"[{self._tag}] grasp lost ({reason}); retry={self.retry_count} "
            f"t={self.attempt_elapsed:.1f}s remaining={ARGS.grasp_timeout - self.attempt_elapsed:.1f}s"
        )

    def _ingest_vision(
        self,
        stereo_measurement: np.ndarray | None,
        wrist_observation: dict[str, object] | None,
        dt: float,
    ) -> None:
        self.filter.predict(dt)
        saw_target = False
        if stereo_measurement is not None:
            measurement = np.asarray(stereo_measurement, dtype=np.float64)
            outlier = (
                self.filter.initialized
                and self.state not in ("SEARCH_STEREO",)
                and np.linalg.norm(measurement - self.filter.position) > 0.20
            )
            if not outlier:
                self.filter.update(measurement, measurement_sigma=0.012)
                self.latest_stereo_position = measurement.copy()
                self.recent_stereo.append(measurement.copy())
                self.stereo_hits += 1
                saw_target = True
        if (
            wrist_observation is not None
            and self.state not in ("SEARCH_STEREO", "VERIFY", "DONE")
            and self._fuse_wrist_bearing(wrist_observation)
        ):
            saw_target = True
        if saw_target:
            self.miss_elapsed = 0.0
        else:
            self.miss_elapsed += dt

    def _update_chase(self, dt: float, ball: np.ndarray, velocity: np.ndarray) -> None:
        if self.reopen_elapsed > 0.0:
            self.reopen_elapsed = max(0.0, self.reopen_elapsed - dt)
            self._set_gripper(GRIPPER_OPEN_POSITION)

        intercept = self._intercept_position(ball, velocity)
        tool, rotation, _ = self._current_tool_pose()
        if self.prev_tool is not None and dt > 1e-6:
            self.tool_velocity = (tool - self.prev_tool) / dt
        self.prev_tool = tool.copy()
        if self.last_visibility_mode == "VISIBILITY_RECOVERY":
            wrist_obs = next(
                (item for item in self.last_observations if item.camera_id == "wrist"),
                None,
            )
            if wrist_obs is not None:
                offset_x, offset_y = recenter_image_offset(
                    wrist_obs, float(CONFIG.visibility.recenter_gain)
                )
                intercept = intercept + np.array([0.0, offset_x, -offset_y], dtype=np.float64)
        if self.last_visibility_mode == "RECENTER_AND_SLOW_DOWN":
            dt = dt * 0.6
        pads = self._grasp_point(tool, rotation)
        pad_points = self._finger_pad_points()
        pad_error = float(np.linalg.norm(pads - ball))
        in_jaws = self._ball_in_jaws(ball, tool, pads)
        closing = self.state in ("CLOSE", "RETRACT_AND_TRANSPORT", "VERIFY") and self.reopen_elapsed <= 0.0
        reading = self.force_sensor.read(ball, pad_points, dt, closing)
        if in_jaws and closing and float(reading["grip"]) > 0.28:
            reading["pinched"] = True
            if float(reading["hold"]) >= float(CONFIG.gripper.confirm_duration_s) * 0.6:
                reading["confirmed"] = True
        slip_limit = BALL_RADIUS + (0.060 if not ARGS.cheats else 0.045)
        close_limit = float(CONFIG.controller.close_position_threshold_m)
        now = time.monotonic()
        if now - self.last_status_time > 0.6:
            print(
                f"[{self._tag}] {self.state} t={self.attempt_elapsed:.1f}s "
                f"pads={np.asarray(pads).round(3)} ball={ball.round(3)} "
                f"vel={velocity.round(3)} err={pad_error:.3f} "
                f"F=({reading['left']:.2f},{reading['right']:.2f})N "
                f"effort={reading['effort']:.2f} grip={reading['grip']:.2f} "
                f"retry={self.retry_count} miss={self.miss_elapsed:.2f}s"
            )
            self.last_status_time = now

        if self.state == "COARSE_APPROACH":
            if self.reopen_elapsed <= 0.0:
                self._set_gripper(GRIPPER_OPEN_POSITION)
            target = self._tool_target_for_ball(intercept, 0.12)
            self._command_pose(target, dt)
            if pad_error < 0.08 or np.linalg.norm(tool - target) < 0.03:
                self.state = "WRIST_TRACK_APPROACH"
                print(
                    f"[{self._tag}] coarse approach complete "
                    f"pads={np.asarray(pads).round(3)} ball={ball.round(3)}"
                )
        elif self.state == "WRIST_TRACK_APPROACH":
            if self.reopen_elapsed <= 0.0:
                self._set_gripper(GRIPPER_OPEN_POSITION)
            target = self._tool_target_for_ball(intercept, 0.0)
            self._command_pose(target, dt)
            if in_jaws or pad_error < float(CONFIG.controller.fine_threshold_m):
                self.state = "VELOCITY_SYNCHRONIZATION"
                print(
                    f"[{self._tag}] entering velocity sync "
                    f"err={pad_error:.3f} m in_jaws={in_jaws} "
                    f"ball_speed={float(np.linalg.norm(velocity)):.3f}"
                )
        elif self.state == "VELOCITY_SYNCHRONIZATION":
            if self.reopen_elapsed <= 0.0:
                self._set_gripper(GRIPPER_OPEN_POSITION)
            relative_speed = float(np.linalg.norm(velocity - self.tool_velocity))
            close_now = in_jaws or (
                pad_error <= close_limit
                and relative_speed
                <= float(CONFIG.controller.close_relative_speed_threshold_mps)
            )
            if close_now:
                self.grasp_position = ball.copy()
                self.close_elapsed = 0.0
                self.force_sensor.reset()
                self._set_gripper(GRIPPER_CLOSED_POSITION)
                self.state = "CLOSE"
                print(
                    f"[{self._tag}] ball in jaws err={pad_error:.3f} "
                    f"rel={relative_speed:.3f} m/s; holding and closing"
                )
            elif pad_error > float(CONFIG.controller.coarse_threshold_m):
                self.state = "WRIST_TRACK_APPROACH"
            else:
                self._command_pose(self._tool_target_for_ball(intercept, 0.0), dt)
        elif self.state == "CLOSE":
            self.close_elapsed += dt
            self._set_gripper(GRIPPER_CLOSED_POSITION)
            if reading["confirmed"] or (in_jaws and self.close_elapsed >= 0.35):
                self.lift_target = tool + np.array([0.0, 0.0, 0.06])
                self.close_elapsed = 0.0
                self.state = "RETRACT_AND_TRANSPORT"
                print(
                    f"[{self._tag}] close held in_jaws={in_jaws} "
                    f"F=({reading['left']:.2f},{reading['right']:.2f})N "
                    f"grip={reading['grip']:.2f}; retracting"
                )
            elif not in_jaws and pad_error > slip_limit and self.close_elapsed > 0.45:
                self._retry_chase("ball left the jaws while closing")
            elif self.close_elapsed > 2.0:
                self._retry_chase("close produced no hold")
        elif self.state == "RETRACT_AND_TRANSPORT":
            self.close_elapsed += dt
            self._set_gripper(GRIPPER_CLOSED_POSITION)
            if not in_jaws and not reading["pinched"] and pad_error > slip_limit:
                self._retry_chase("lost contact during retract")
                return
            self._command_pose(self.lift_target, dt)
            if np.linalg.norm(tool - self.lift_target) < 0.04 or self.close_elapsed > 1.0:
                self.close_elapsed = 0.0
                self.state = "VERIFY"
                print(f"[{self._tag}] transport pose reached; checking held contact")
        elif self.state == "VERIFY":
            self.close_elapsed += dt
            self._set_gripper(GRIPPER_CLOSED_POSITION)
            held = in_jaws or (bool(reading["pinched"]) and pad_error < BALL_RADIUS + 0.04)
            if held and (reading["confirmed"] or self.close_elapsed >= 0.4):
                self.state = "DONE"
                self.terminal_result = "SUCCESS"
                self.terminal_reason = "hold verified"
                print(
                    f"[{self._tag}] grasp verified at {ball.round(3)} "
                    f"hold={pad_error:.3f} m "
                    f"F=({reading['left']:.2f},{reading['right']:.2f})N "
                    f"retries={self.retry_count} t={self.attempt_elapsed:.1f}s"
                )
                return
            if self.close_elapsed >= 1.2:
                self._retry_chase("verify failed; ball is not held")

    def _update_cheats(
        self,
        dt: float,
        ground_truth: np.ndarray | None,
        ball_velocity: np.ndarray | None,
    ) -> None:
        if self.state == "DONE":
            return
        if ground_truth is None:
            return
        ball = np.asarray(ground_truth, dtype=np.float64)
        if position_is_beyond_despawn_boundary(ball):
            self.abort(
                f"ball left workspace at {ball.round(3)} after {self.attempt_elapsed:.1f}s"
            )
            return
        if ball_velocity is not None:
            velocity = np.asarray(ball_velocity, dtype=np.float64)
        elif self.live_ball is not None and dt > 1e-6:
            velocity = (ball - self.live_ball) / dt
        else:
            velocity = np.zeros(3, dtype=np.float64)
        self.live_ball = ball.copy()
        self.live_velocity = velocity
        if self.state == "SEARCH_STEREO":
            self._set_gripper(GRIPPER_OPEN_POSITION)
            self.state = "COARSE_APPROACH"
            print(
                f"[{self._tag}] chasing live ball {ball.round(3)} "
                f"velocity={velocity.round(3)} timeout={ARGS.grasp_timeout:.0f}s"
            )
            return
        self.attempt_elapsed += dt
        if self.attempt_elapsed >= ARGS.grasp_timeout:
            self.abort(f"timeout {ARGS.grasp_timeout:.0f}s retries={self.retry_count}")
            return
        self._update_chase(dt, ball, velocity)

    def _build_estimate(self) -> TargetEstimate:
        covariance = np.asarray(self.filter.covariance, dtype=np.float64)
        position_std = np.sqrt(np.maximum(np.diag(covariance)[:3], 0.0))
        age = self.miss_elapsed
        confidence = 0.0
        if self.filter.initialized:
            age_term = max(0.0, 1.0 - age / 1.5)
            std_term = max(
                0.0,
                1.0
                - 0.15
                * float(position_std.max())
                / max(float(CONFIG.estimator.max_position_std_m), 1e-3),
            )
            confidence = float(np.clip(0.55 * age_term + 0.45 * std_term, 0.0, 1.0))
        estimate = TargetEstimate(
            timestamp=time.monotonic(),
            position=self.filter.position.copy(),
            velocity=self.filter.velocity.copy(),
            covariance=covariance,
            confidence=confidence,
            source="vision",
            valid=self.filter.initialized and age <= 1.20,
            measurement_age=age,
            camera_id="stereo",
            position_std=position_std,
        )
        assert estimate.source != "oracle"
        self.last_estimate = estimate
        return estimate

    def _update_vision(
        self,
        stereo_measurement: np.ndarray | None,
        wrist_observation: dict[str, object] | None,
        dt: float,
        observations: list[BallObservation] | None = None,
    ) -> None:
        if self.state == "DONE":
            return
        new_observations = list(observations or [])
        if wrist_observation is not None and "observation" in wrist_observation:
            wrist_ball = wrist_observation["observation"]
            if isinstance(wrist_ball, BallObservation):
                new_observations.append(wrist_ball)
        if new_observations:
            self.last_observations = new_observations
        visibility = self.visibility.evaluate(self.last_observations)
        if new_observations:
            self.last_visibility_mode = visibility.mode
        self._ingest_vision(stereo_measurement, wrist_observation, dt)
        estimate = self._build_estimate()
        if self.state == "SEARCH_STEREO":
            if not self._target_is_acquired():
                return
            self._set_gripper(GRIPPER_OPEN_POSITION)
            self.state = "COARSE_APPROACH"
            print(
                f"[{self._tag}] target acquired position={estimate.position.round(3)} "
                f"velocity={estimate.velocity.round(3)} "
                f"confidence={estimate.confidence:.2f} "
                f"timeout={ARGS.grasp_timeout:.0f}s"
            )

        holding = self.state in ("CLOSE", "RETRACT_AND_TRANSPORT", "VERIFY")
        if (
            not estimate.valid
            and not holding
            and self.miss_elapsed > 1.20
        ):
            print(
                f"[{self._tag}] lost track for {self.miss_elapsed:.1f}s; returning to search"
            )
            self.recent_stereo.clear()
            self.state = "SEARCH_STEREO"
            self._set_gripper(GRIPPER_OPEN_POSITION)
            return

        ball = estimate.position
        velocity = estimate.velocity
        if position_is_beyond_despawn_boundary(ball):
            self.abort(
                f"estimate left workspace at {ball.round(3)} after {self.attempt_elapsed:.1f}s"
            )
            return
        self.attempt_elapsed += dt
        if self.attempt_elapsed >= ARGS.grasp_timeout:
            self.abort(f"timeout {ARGS.grasp_timeout:.0f}s retries={self.retry_count}")
            return
        if (
            self.last_visibility_mode == "VISIBILITY_RECOVERY"
            and not holding
            and new_observations
            and time.monotonic() - self.last_status_time > 0.8
        ):
            print(
                f"[{self._tag}] FOV risk margin={visibility.margin_px:.1f}px "
                f"camera={visibility.camera_id}; recentering while chasing"
            )
        if self.retry_count > int(CONFIG.episode.max_retries):
            self.abort(f"max retries {CONFIG.episode.max_retries}")
            return
        pads = self._finger_pad_midpoint()
        position_error = (
            None if pads is None else float(np.linalg.norm(pads - ball))
        )
        relative_speed = float(np.linalg.norm(velocity - self.tool_velocity))
        self.structured_log.log_bundle(
            self.last_observations[0] if self.last_observations else None,
            estimate,
            None,
            self.state,
            position_error,
            relative_speed,
            visibility,
            visibility.close_allowed,
        )
        self._update_chase(dt, ball, velocity)

    def update(
        self,
        stereo_measurement: np.ndarray | None,
        wrist_observation: dict[str, object] | None,
        dt: float,
        ground_truth: np.ndarray | None = None,
        cheat_position: np.ndarray | None = None,
        ball_velocity: np.ndarray | None = None,
        observations: list[BallObservation] | None = None,
    ) -> None:
        self.frame_count += 1
        if ARGS.cheats:
            self._update_cheats(dt, ground_truth, ball_velocity)
            return
        self._update_vision(
            stereo_measurement,
            wrist_observation,
            dt,
            observations=observations,
        )


# Measured 2F-85 pad midpoint in the ready pose: about 12.6 cm along tool -X.
DEFAULT_TOOL_TO_PADS = np.array([-0.126, 0.0, 0.002], dtype=np.float64)
PREGRASP_STANDOFF = 0.12
LIFT_HEIGHT = 0.15


def is_reachable_ball_position(position: np.ndarray) -> bool:
    point = np.asarray(position, dtype=np.float32)
    if np.any(point < REACHABLE_MIN) or np.any(point > REACHABLE_MAX):
        return False
    radial = float(np.hypot(point[0], point[1]))
    return REACHABLE_RADIAL[0] <= radial <= REACHABLE_RADIAL[1]


def tool_waypoints_for_ball(ball: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offset = DEFAULT_TOOL_TO_PADS
    direction = offset / np.linalg.norm(offset)
    grasp = np.asarray(ball, dtype=np.float64) - offset
    pregrasp = grasp - direction * PREGRASP_STANDOFF
    lift = grasp + np.array([0.0, 0.0, LIFT_HEIGHT], dtype=np.float64)
    return pregrasp, grasp, lift


def ready_tool_orientation(solver: LulaKinematicsSolver) -> np.ndarray:
    _, rotation = solver.compute_forward_kinematics("tool0", READY_ARM_POSE)
    return rot_matrices_to_quats(np.asarray(rotation, dtype=np.float64))


def ball_is_reachable_with_ik(
    solver: LulaKinematicsSolver,
    ball: np.ndarray,
    seed_joints: np.ndarray | None = None,
    orientation: np.ndarray | None = None,
) -> bool:
    if not is_reachable_ball_position(ball):
        return False
    joints = np.asarray(
        READY_ARM_POSE if seed_joints is None else seed_joints,
        dtype=np.float64,
    )
    if orientation is None:
        orientation = ready_tool_orientation(solver)
    for waypoint in tool_waypoints_for_ball(ball):
        solver.set_default_cspace_seeds(np.asarray([joints], dtype=np.float64))
        solved, success = solver.compute_inverse_kinematics(
            "tool0",
            waypoint,
            orientation,
            warm_start=joints,
            position_tolerance=0.012,
            orientation_tolerance=0.12,
        )
        if not success:
            return False
        joints = np.asarray(solved, dtype=np.float64)
    return True


def sample_reachable_ball_position(
    rng: np.random.Generator,
    solver: LulaKinematicsSolver | None = None,
    seed_joints: np.ndarray | None = None,
    orientation: np.ndarray | None = None,
) -> np.ndarray:
    fallback = CAPTURE_CENTER.copy()
    for _ in range(80):
        candidate = rng.uniform(REACHABLE_MIN, REACHABLE_MAX).astype(np.float32)
        if solver is None:
            if is_reachable_ball_position(candidate):
                return candidate
            continue
        if ball_is_reachable_with_ik(solver, candidate, seed_joints, orientation):
            return candidate
    if solver is None or ball_is_reachable_with_ik(
        solver, fallback, seed_joints, orientation
    ):
        return fallback
    return fallback


def sample_crossing_trajectory(
    rng: np.random.Generator,
    solver: LulaKinematicsSolver | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Launch a target from below the arm toward a reachable capture pose."""
    position = LAUNCH_CENTER.copy()
    position[1] += rng.uniform(-0.07, 0.07)
    position[2] += rng.uniform(-0.035, 0.035)
    if ARGS.speed == 0:
        position = sample_reachable_ball_position(rng, solver=solver)
        zeros = np.zeros(3, dtype=np.float32)
        return position.astype(np.float32), zeros, zeros.copy(), "reachable-workspace"

    aim_point = sample_reachable_ball_position(rng, solver=solver)
    velocity = aim_point - position
    velocity = (velocity / np.linalg.norm(velocity) * ARGS.speed).astype(np.float32)
    angular_velocity = random_unit_vector(rng) * float(rng.uniform(0.18, 0.48))
    return (
        position.astype(np.float32),
        velocity,
        angular_velocity.astype(np.float32),
        "under-arm-launcher",
    )


def create_preview_material(
    stage,
    path: str,
    color: tuple[float, float, float],
    roughness: float = 0.55,
    emissive_color: tuple[float, float, float] | None = None,
):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
    if emissive_color is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*emissive_color)
        )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_material(prim, material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def create_physics_material(
    stage,
    path: str,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
    friction_combine_mode: str = "average",
    restitution_combine_mode: str = "average",
):
    material = UsdShade.Material.Define(stage, path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(static_friction)
    physics_material.CreateDynamicFrictionAttr().Set(dynamic_friction)
    physics_material.CreateRestitutionAttr().Set(restitution)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr().Set(friction_combine_mode)
    physx_material.CreateRestitutionCombineModeAttr().Set(restitution_combine_mode)
    return material


def bind_physics_material(prim, material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def bind_gripper_physics_material(stage, material) -> None:
    link_paths = (
        f"{ROBOT_PATH}/Gripper/Robotiq_2F_85/left_outer_finger",
        f"{ROBOT_PATH}/Gripper/Robotiq_2F_85/right_outer_finger",
        f"{ROBOT_PATH}/Gripper/Robotiq_2F_85/left_inner_finger",
        f"{ROBOT_PATH}/Gripper/Robotiq_2F_85/right_inner_finger",
    )
    for link_path in link_paths:
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim.IsValid():
            raise RuntimeError(f"UR5e gripper link is missing: {link_path}")
        bind_physics_material(link_prim, material)


def configure_gripper_drive(stage) -> None:
    joint_path = f"{ROBOT_PATH}/Gripper/Robotiq_2F_85/Joints/finger_joint"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        raise RuntimeError(f"UR5e gripper drive joint is missing: {joint_path}")
    drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
    drive.CreateStiffnessAttr().Set(180.0)
    drive.CreateDampingAttr().Set(40.0)
    drive.CreateMaxForceAttr().Set(12.0)


def create_grasp_frame(stage) -> None:
    grasp_frame = UsdGeom.Xform.Define(
        stage, f"{ROBOT_TOOL_LINK_PATH}/grasp_frame"
    )
    UsdGeom.Xformable(grasp_frame.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, GRASP_FRAME_OFFSET)
    )


def create_finger_contact_pads(stage, physics_material) -> None:
    """Small contact pads on the inner-finger links. Offsets come from the 2F-85 COM."""
    pad_visual = create_preview_material(
        stage,
        "/World/Materials/FingerPad",
        (0.12, 0.10, 0.09),
        roughness=0.85,
    )
    # Isaac 2F-85 inner-finger COM is about (0, ±0.054, 0.114); the rubber pad
    # sits on the inward face near the fingertip, not a mesh bounding box.
    local_centers = {
        FINGER_LINK_PATHS[0]: (0.0, -0.030, 0.118),
        FINGER_LINK_PATHS[1]: (0.0, 0.030, 0.118),
    }
    for path, center in local_centers.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"UR5e finger link is missing: {path}")
        pad = UsdGeom.Cube.Define(stage, f"{path}/ContactPad")
        pad.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(pad.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(0.022, 0.008, 0.028))
        collision = UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
        collision.CreateCollisionEnabledAttr(True)
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(pad.GetPrim())
        physx_collision.CreateContactOffsetAttr().Set(0.002)
        physx_collision.CreateRestOffsetAttr().Set(0.0)
        PhysxSchema.PhysxContactReportAPI.Apply(pad.GetPrim())
        bind_material(pad.GetPrim(), pad_visual)
        bind_physics_material(pad.GetPrim(), physics_material)
        print(f"[space-grasp] finger pad at {path}/ContactPad local={center}")


def create_physics_scene(stage) -> None:
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(0.0)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateEnableCCDAttr(True)
    physx_scene.CreateEnableStabilizationAttr(True)
    physx_scene.CreateSolverTypeAttr("TGS")
    # The target moves much slower than PhysX's usual bounce threshold.
    physx_scene.CreateBounceThresholdAttr().Set(0.005)


def create_visual_platform(stage, physics_material) -> None:
    material = create_preview_material(stage, "/World/Materials/Platform", (0.10, 0.12, 0.16), 0.35)
    launcher_material = create_preview_material(
        stage, "/World/Materials/Launcher", (0.65, 0.12, 0.035), 0.45
    )
    platform = UsdGeom.Cube.Define(stage, "/World/SatelliteDeck")
    platform.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(platform.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*DECK_CENTER.tolist()))
    xform.AddScaleOp().Set(Gf.Vec3f(*DECK_SIZE.tolist()))
    UsdPhysics.CollisionAPI.Apply(platform.GetPrim())
    bind_material(platform.GetPrim(), material)
    bind_physics_material(platform.GetPrim(), physics_material)

    launcher = UsdGeom.Cylinder.Define(stage, "/World/BallLauncher")
    launcher.CreateAxisAttr("X")
    launcher.CreateRadiusAttr(0.055)
    launcher.CreateHeightAttr(0.12)
    launcher_xform = UsdGeom.Xformable(launcher.GetPrim())
    launcher_xform.AddTranslateOp().Set(Gf.Vec3d(0.14, 0.0, float(LAUNCH_CENTER[2])))
    bind_material(launcher.GetPrim(), launcher_material)


def create_textured_ball(
    stage, rng: np.random.Generator, position: np.ndarray, physics_material
) -> None:
    root = UsdGeom.Xform.Define(stage, TARGET_PATH)
    UsdGeom.Xformable(root.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*position.tolist()))
    root.GetPrim().CreateAttribute("spaceGrasp:valid", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
    root.GetPrim().CreateAttribute("spaceGrasp:episodeId", Sdf.ValueTypeNames.Int, custom=True).Set(0)

    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid_body.CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(BALL_MASS)
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    physx_body.CreateLinearDampingAttr(0.0)
    physx_body.CreateAngularDampingAttr(0.0)
    physx_body.CreateEnableCCDAttr(True)
    physx_body.CreateSleepThresholdAttr(0.0)

    base_material = create_preview_material(
        stage,
        "/World/Materials/BallBase",
        (0.08, 0.82, 0.16),
        0.65,
        emissive_color=(0.01, 0.18, 0.025),
    )
    dark_material = create_preview_material(stage, "/World/Materials/BallDark", (0.025, 0.035, 0.05), 0.8)
    orange_material = create_preview_material(stage, "/World/Materials/BallOrange", (0.95, 0.24, 0.035), 0.5)

    sphere = UsdGeom.Sphere.Define(stage, f"{TARGET_PATH}/CollisionSphere")
    sphere.CreateRadiusAttr(BALL_RADIUS)
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    bind_material(sphere.GetPrim(), base_material)
    bind_physics_material(sphere.GetPrim(), physics_material)

    # Raised visual dots form an asymmetric, rotation-observable texture. They have no colliders,
    # so the physical target remains an exact sphere.
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(28):
        y = 1.0 - (2.0 * (index + 0.5) / 28.0)
        radial = math.sqrt(max(0.0, 1.0 - y * y))
        angle = index * golden_angle + rng.uniform(-0.10, 0.10)
        direction = np.array([math.cos(angle) * radial, math.sin(angle) * radial, y])
        patch = UsdGeom.Sphere.Define(stage, f"{TARGET_PATH}/TexturePatch_{index:02d}")
        patch.CreateRadiusAttr(BALL_RADIUS * (0.133 if index % 4 else 0.178))
        UsdGeom.Xformable(patch.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(*(direction * (BALL_RADIUS * 1.022)).tolist())
        )
        bind_material(patch.GetPrim(), orange_material if index % 5 == 0 else dark_material)


def camera_transform(eye: tuple[float, float, float], target: tuple[float, float, float]) -> Gf.Matrix4d:
    eye_v = Gf.Vec3d(*eye)
    forward = (Gf.Vec3d(*target) - eye_v).GetNormalized()
    world_up = Gf.Vec3d(0.0, 0.0, 1.0)
    right = Gf.Cross(forward, world_up).GetNormalized()
    up = Gf.Cross(right, forward).GetNormalized()
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0.0))
    matrix.SetRow(1, Gf.Vec4d(up[0], up[1], up[2], 0.0))
    matrix.SetRow(2, Gf.Vec4d(-forward[0], -forward[1], -forward[2], 0.0))
    matrix.SetRow(3, Gf.Vec4d(eye_v[0], eye_v[1], eye_v[2], 1.0))
    return matrix


def create_cameras(stage) -> None:
    UsdGeom.Xform.Define(stage, "/World/Cameras")
    target = tuple(float(value) for value in CAPTURE_CENTER)
    eyes = [(1.20, -0.42, 0.88), (1.20, 0.42, 0.88)]
    for path, eye in zip(CAMERA_PATHS, eyes):
        camera = UsdGeom.Camera.Define(stage, path)
        camera.CreateFocalLengthAttr(20.0)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(15.2908)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 100.0))
        UsdGeom.Xformable(camera.GetPrim()).AddTransformOp().Set(camera_transform(eye, target))


def create_wrist_camera(stage) -> str:
    tool_prim = stage.GetPrimAtPath(ROBOT_TOOL_LINK_PATH)
    if not tool_prim.IsValid():
        raise RuntimeError(f"UR5e tool link is missing: {ROBOT_TOOL_LINK_PATH}")

    camera = UsdGeom.Camera.Define(stage, WRIST_CAMERA_PATH)
    camera.CreateFocalLengthAttr(8.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.2908)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.025, 10.0))
    camera.GetPrim().CreateAttribute(
        "spaceGrasp:requestedFps", Sdf.ValueTypeNames.Float, custom=True
    ).Set(10.0)
    camera.GetPrim().CreateRelationship("spaceGrasp:attachedTo", custom=True).SetTargets(
        [tool_prim.GetPath()]
    )
    return WRIST_CAMERA_PATH


def _dock_camera_viewports(titles: list[str]) -> None:
    try:
        import omni.ui as ui
    except Exception:
        return
    APP.update()
    APP.update()
    main = ui.Workspace.get_window("Viewport")
    windows = [ui.Workspace.get_window(title) for title in titles]
    if main is None or any(window is None for window in windows):
        return
    windows[0].dock_in(main, ui.DockPosition.BOTTOM, 0.30)
    APP.update()
    windows[1].dock_in(windows[0], ui.DockPosition.RIGHT, 0.50)
    APP.update()
    windows[2].dock_in(windows[1], ui.DockPosition.RIGHT, 0.50)
    APP.update()


def create_camera_viewports(camera_paths: list[str]):
    if ARGS.headless:
        return []
    titles = [title for path, title in CAMERA_VIEWPORT_SPECS]
    path_to_title = {path: title for path, title in CAMERA_VIEWPORT_SPECS}
    windows = []
    for path in camera_paths:
        window = ViewportManager.create_viewport_window(
            camera=path,
            resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
            title=path_to_title.get(path, path.rsplit("/", 1)[-1]),
        )
        windows.append(window)
        APP.update()
    _dock_camera_viewports(titles)
    print(
        "[space-grasp] camera viewports: "
        + ", ".join(f"{title}" for title in titles)
    )
    return windows


def create_lights(stage) -> None:
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(1400.0)
    key.CreateAngleAttr(1.5)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(38.0, -32.0, 25.0))
    fill = UsdLux.DomeLight.Define(stage, "/World/Lights/Fill")
    fill.CreateIntensityAttr(220.0)
    fill.CreateColorAttr(Gf.Vec3f(0.30, 0.38, 0.55))


def setup_rgb_writer(output_dir: Path, camera_paths: list[str]):
    import omni.replicator.core as rep

    output_dir.mkdir(parents=True, exist_ok=True)
    render_products = [rep.create.render_product(path, (640, 480)) for path in camera_paths]
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(output_dir.resolve()), rgb=True)
    writer.attach(render_products)
    return writer, render_products


def build_scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    stage_utils.create_new_stage()
    APP.update()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    create_physics_scene(stage)
    panel_physics_material = create_physics_material(
        stage,
        "/World/PhysicsMaterials/Panel",
        static_friction=0.15,
        dynamic_friction=0.10,
        restitution=0.72,
        restitution_combine_mode="max",
    )
    ball_physics_material = create_physics_material(
        stage,
        "/World/PhysicsMaterials/GraspableBall",
        static_friction=0.70,
        dynamic_friction=0.50,
        restitution=0.78,
    )
    gripper_physics_material = create_physics_material(
        stage,
        "/World/PhysicsMaterials/GripperFingers",
        static_friction=1.20,
        dynamic_friction=1.00,
        restitution=0.02,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    create_visual_platform(stage, panel_physics_material)
    create_lights(stage)
    create_cameras(stage)

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Isaac Sim assets root is unavailable; check Internet/proxy access.")
    robot_asset = assets_root + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
    robot_prim = stage_utils.add_reference_to_stage(usd_path=robot_asset, path=ROBOT_PATH)
    robot_prim.GetVariantSet("Gripper").SetVariantSelection("Robotiq_2f_85")
    APP.update()
    bind_gripper_physics_material(stage, gripper_physics_material)
    configure_gripper_drive(stage)
    create_grasp_frame(stage)
    create_finger_contact_pads(stage, gripper_physics_material)
    wrist_camera_path = create_wrist_camera(stage)

    initial_position, initial_velocity, initial_angular_velocity, entry_face = sample_crossing_trajectory(rng)
    create_textured_ball(stage, rng, initial_position, ball_physics_material)
    APP.update()
    ViewportManager.set_camera_view(
        "/OmniverseKit_Persp",
        eye=[1.55, 1.35, 1.20],
        target=CAPTURE_CENTER.tolist(),
    )
    APP.update()
    return initial_position, initial_velocity, initial_angular_velocity, entry_face, wrist_camera_path


def position_is_beyond_despawn_boundary(position: np.ndarray) -> bool:
    point = np.asarray(position, dtype=np.float32)
    margin = ARGS.despawn_margin

    if np.any(point[:2] < WORKSPACE_MIN[:2] - margin) or np.any(
        point[:2] > WORKSPACE_MAX[:2] + margin
    ):
        return True
    if point[2] > WORKSPACE_MAX[2] + margin:
        return True
    if point[2] >= WORKSPACE_MIN[2] - margin:
        return False

    # Do not invalidate a downward-moving target before it can reach the deck.
    over_deck = bool(
        DECK_MIN[0] - BALL_RADIUS <= point[0] <= DECK_MAX[0] + BALL_RADIUS
        and DECK_MIN[1] - BALL_RADIUS <= point[1] <= DECK_MAX[1] + BALL_RADIUS
    )
    if over_deck:
        return bool(point[2] < DECK_MIN[2] - margin)
    return True


def target_is_beyond_despawn_boundary(target: RigidPrim) -> bool:
    positions, _ = target.get_world_poses()
    return position_is_beyond_despawn_boundary(positions.numpy()[0])


def set_target_valid(stage, valid: bool, episode_id: int) -> None:
    target_prim = stage.GetPrimAtPath(TARGET_PATH)
    target_prim.GetAttribute("spaceGrasp:valid").Set(valid)
    target_prim.GetAttribute("spaceGrasp:episodeId").Set(episode_id)
    imageable = UsdGeom.Imageable(target_prim)
    imageable.MakeVisible() if valid else imageable.MakeInvisible()
    collision = UsdPhysics.CollisionAPI(stage.GetPrimAtPath(f"{TARGET_PATH}/CollisionSphere"))
    collision.GetCollisionEnabledAttr().Set(valid)


def deactivate_target(stage, target: RigidPrim, episode_id: int) -> None:
    set_target_valid(stage, False, episode_id)
    zeros = np.zeros((1, 3), dtype=np.float32)
    target.set_velocities(zeros, zeros)
    target.set_world_poses(positions=np.asarray([[10.0, 10.0, 10.0]], dtype=np.float32))
    print(f"[space-grasp] target invalid | episode={episode_id}")


def spawn_target(
    stage,
    target: RigidPrim,
    rng: np.random.Generator,
    episode_id: int,
    solver: LulaKinematicsSolver | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    position, velocity, angular_velocity, entry_face = sample_crossing_trajectory(
        rng, solver=solver
    )
    target.set_world_poses(
        positions=np.asarray([position], dtype=np.float32),
        orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    target.set_velocities(
        np.asarray([velocity], dtype=np.float32),
        np.asarray([angular_velocity], dtype=np.float32),
    )
    set_target_valid(stage, True, episode_id)
    print(
        "[space-grasp] target active | "
        f"episode={episode_id} entry={entry_face} "
        f"position={position.round(3)} velocity={velocity.round(3)}"
    )
    return position, velocity


def settle_robot_and_gripper(robot: Articulation, frames: int) -> None:
    robot.set_dof_position_targets(
        np.asarray([[GRIPPER_OPEN_POSITION]], dtype=np.float32),
        dof_indices=robot.get_dof_indices(["finger_joint"]).numpy(),
    )
    for _ in range(max(1, frames)):
        SimulationManager.step()


def main() -> None:
    rng = np.random.default_rng(ARGS.seed)
    (
        initial_position,
        initial_velocity,
        initial_angular_velocity,
        entry_face,
        wrist_camera_path,
    ) = build_scene(rng)
    stage = omni.usd.get_context().get_stage()

    if ARGS.save_usd:
        output_path = ARGS.save_usd.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(output_path))
        print(f"[space-grasp] Exported scene: {output_path}")

    camera_viewports = create_camera_viewports([*CAMERA_PATHS, wrist_camera_path])
    writer_resources = (
        setup_rgb_writer(ARGS.capture_dir, [*CAMERA_PATHS, wrist_camera_path])
        if ARGS.capture_dir
        else None
    )
    target = RigidPrim(TARGET_PATH, masses=[BALL_MASS])
    robot = Articulation(ROBOT_PATH)

    dt = 1.0 / ARGS.physics_hz
    SimulationManager.setup_simulation(dt=dt, device="cuda:0")
    RenderingManager.set_dt(dt)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    APP.update()

    target.set_velocities(
        np.asarray([initial_velocity], dtype=np.float32),
        np.asarray([initial_angular_velocity], dtype=np.float32),
    )
    wrist_camera_follower = WristCameraFollower(stage, robot, wrist_camera_path)
    robot.set_dof_positions(
        np.asarray([READY_ARM_POSE], dtype=np.float32),
        dof_indices=wrist_camera_follower.arm_dof_indices,
    )
    robot.set_dof_position_targets(
        np.asarray([READY_ARM_POSE], dtype=np.float32),
        dof_indices=wrist_camera_follower.arm_dof_indices,
    )
    robot.set_dof_position_targets(
        np.asarray([[GRIPPER_OPEN_POSITION]], dtype=np.float32),
        dof_indices=robot.get_dof_indices(["finger_joint"]).numpy(),
    )
    wrist_camera_follower.update()
    teleop = (
        ManipulatorKeyboardTeleop(
            robot,
            solver=wrist_camera_follower.solver,
            arm_dof_indices=wrist_camera_follower.arm_dof_indices,
        )
        if not ARGS.headless and not ARGS.disable_teleop and not ARGS.auto_grasp and not ARGS.cheats
        else None
    )
    use_vision = ARGS.auto_grasp and not ARGS.cheats
    vision_tracker = (
        MultiCameraBallTracker(stage, CAMERA_PATHS) if use_vision else None
    )
    wrist_tracker = (
        WristImageTracker(stage, wrist_camera_path) if use_vision else None
    )
    visual_grasp = (
        StereoKalmanGraspController(
            stage,
            robot,
            wrist_camera_follower.solver,
            wrist_camera_follower.arm_dof_indices,
        )
        if ARGS.auto_grasp or ARGS.cheats
        else None
    )
    reset_hotkey = SceneResetHotkey() if not ARGS.headless else None
    settle_robot_and_gripper(robot, int(CONFIG.episode.settling_frames))
    if visual_grasp is not None:
        visual_grasp.cheat_offset_locked = False
        visual_grasp.force_sensor.reset()
    episode_recorder = EpisodeRecorder(
        ROOT / "outputs" / str(CONFIG.evaluation.metrics_filename)
    )
    episode_recorder.start(0)
    oracle_logger = StructuredLogger(CONFIG.evaluation.log_interval_s)

    if ARGS.speed == 0:
        solver = wrist_camera_follower.solver
        orientation = ready_tool_orientation(solver)
        if not ball_is_reachable_with_ik(
            solver, initial_position, READY_ARM_POSE, orientation
        ):
            initial_position = sample_reachable_ball_position(
                rng, solver=solver, seed_joints=READY_ARM_POSE, orientation=orientation
            )
            target.set_world_poses(
                positions=np.asarray([initial_position], dtype=np.float32),
                orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
            print(
                f"[space-grasp] initial ball moved into reachable workspace "
                f"{initial_position.round(3)}"
            )
    cheat_position = initial_position.copy() if ARGS.cheats else None
    print(
        "[space-grasp] UR5e zero-g under-arm launch scene ready | "
        f"seed={ARGS.seed} entry={entry_face} position={initial_position.round(3)} "
        f"velocity={initial_velocity.round(3)} wrist_camera={wrist_camera_path}"
        f"{' | CHEATS live-chase grasp' if ARGS.cheats else ''}"
        f"{' | ORACLE baseline' if ARGS.cheats else ''}"
        f"{' | VISION visibility-aware chase' if use_vision else ''}"
    )
    render_interval = max(1, round(ARGS.physics_hz / ARGS.render_hz))
    teleop_interval = max(1, round(ARGS.physics_hz / 60))
    lifecycle_interval = max(1, round(ARGS.physics_hz / 20))
    chase_interval = max(1, round(ARGS.physics_hz / 30))
    respawn_delay_steps = round(ARGS.respawn_delay * ARGS.physics_hz)
    max_steps = round(ARGS.duration * ARGS.physics_hz) if ARGS.duration > 0 else None
    step = 0
    episode_id = 0
    target_valid = True
    respawn_at_step: int | None = None
    pending_stereo: np.ndarray | None = None
    pending_wrist: dict[str, object] | None = None
    pending_observations: list[BallObservation] = []

    try:
        while APP.is_running() and (max_steps is None or step < max_steps):
            SimulationManager.step()
            step += 1
            if reset_hotkey is not None and reset_hotkey.consume():
                episode_id += 1
                spawned_position, _ = spawn_target(
                    stage, target, rng, episode_id, solver=wrist_camera_follower.solver
                )
                if ARGS.cheats:
                    cheat_position = spawned_position.copy()
                robot.set_dof_positions(
                    np.asarray([READY_ARM_POSE], dtype=np.float32),
                    dof_indices=wrist_camera_follower.arm_dof_indices,
                )
                robot.set_dof_position_targets(
                    np.asarray([READY_ARM_POSE], dtype=np.float32),
                    dof_indices=wrist_camera_follower.arm_dof_indices,
                )
                robot.set_dof_position_targets(
                    np.asarray([[GRIPPER_OPEN_POSITION]], dtype=np.float32),
                    dof_indices=robot.get_dof_indices(["finger_joint"]).numpy(),
                )
                if teleop is not None:
                    teleop.reset_requested = True
                if visual_grasp is not None:
                    visual_grasp.reset(full=True)
                settle_robot_and_gripper(robot, int(CONFIG.episode.settling_frames))
                if visual_grasp is not None:
                    visual_grasp.cheat_offset_locked = False
                episode_recorder.start(episode_id)
                target_valid = True
                respawn_at_step = None
                print(f"[space-grasp] full episode reset | episode={episode_id}")
            if teleop is not None and step % teleop_interval == 0:
                teleop.update(teleop_interval * dt)
            if step % lifecycle_interval == 0:
                if target_valid and target_is_beyond_despawn_boundary(target):
                    if visual_grasp is not None:
                        visual_grasp.abort("ball left workspace")
                    deactivate_target(stage, target, episode_id)
                    target_valid = False
                    respawn_at_step = (
                        None
                        if ARGS.single_pass or ARGS.auto_grasp or ARGS.cheats
                        else step + respawn_delay_steps
                    )
                elif (
                    not ARGS.auto_grasp
                    and not ARGS.cheats
                    and not target_valid
                    and respawn_at_step is not None
                    and step >= respawn_at_step
                ):
                    episode_id += 1
                    spawn_target(
                        stage, target, rng, episode_id, solver=wrist_camera_follower.solver
                    )
                    target_valid = True
                    respawn_at_step = None
            if step % render_interval == 0:
                wrist_camera_follower.update()
                RenderingManager.render()
                APP.update()
                if use_vision and vision_tracker is not None:
                    try:
                        vision_result = vision_tracker.estimate()
                        if vision_result is None:
                            pending_stereo = None
                            pending_observations = []
                        else:
                            pending_stereo, pending_observations = vision_result
                        pending_wrist = (
                            wrist_tracker.observe() if wrist_tracker is not None else None
                        )
                    except Exception:
                        traceback.print_exc()
                        print("[space-grasp] visual pipeline exception; window stays open")
            if visual_grasp is not None and target_valid and step % chase_interval == 0:
                try:
                    previous_state = visual_grasp.state
                    if ARGS.cheats:
                        ground_truth = target.get_world_poses()[0].numpy()[0]
                        linear_velocity, _ = target.get_velocities()
                        visual_grasp.update(
                            None,
                            None,
                            chase_interval * dt,
                            ground_truth=ground_truth,
                            ball_velocity=linear_velocity.numpy()[0],
                        )
                    elif use_vision:
                        visual_grasp.update(
                            pending_stereo,
                            pending_wrist,
                            chase_interval * dt,
                            observations=pending_observations,
                        )
                        if bool(CONFIG.evaluation.log_oracle) and visual_grasp.last_estimate is not None:
                            ground_truth = target.get_world_poses()[0].numpy()[0]
                            linear_velocity, _ = target.get_velocities()
                            oracle = OracleMetrics.from_states(
                                ground_truth,
                                linear_velocity.numpy()[0],
                                visual_grasp.last_estimate,
                            )
                            episode_recorder.observe(
                                bool(pending_stereo is not None or pending_wrist is not None),
                                oracle.p_error,
                            )
                            if oracle_logger.should_emit():
                                print(
                                    "[oracle-metrics] "
                                    f"p_gt={np.round(ground_truth, 3)} "
                                    f"p_error={oracle.p_error:.3f} "
                                    f"v_error={oracle.v_error:.3f}"
                                )
                        pending_stereo = None
                        pending_wrist = None
                        pending_observations = []
                    if (
                        visual_grasp.state == "DONE"
                        and previous_state != "DONE"
                        and visual_grasp.terminal_result
                    ):
                        episode_recorder.retry_count = visual_grasp.retry_count
                        episode_recorder.finish(
                            visual_grasp.terminal_result,
                            visual_grasp.terminal_reason,
                        )
                except Exception:
                    traceback.print_exc()
                    print("[space-grasp] grasp pipeline exception; window stays open")
        print(
            f"[space-grasp] main loop exit | step={step} "
            f"app_running={APP.is_running()} max_steps={max_steps}"
        )
    finally:
        timeline.stop()
        if teleop is not None:
            teleop.close()
        if reset_hotkey is not None:
            reset_hotkey.close()
        if vision_tracker is not None:
            vision_tracker.close()
        if wrist_tracker is not None:
            wrist_tracker.close()
        camera_viewports = None
        if writer_resources:
            writer, render_products = writer_resources
            writer.detach()
            render_products.clear()
        APP.update()
        time.sleep(0.05)
        APP.close()


if __name__ == "__main__":
    main()
