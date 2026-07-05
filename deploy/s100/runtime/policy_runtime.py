"""Stateful S100 policy inference and safety handling.

The sensor and motor SDKs are deliberately kept outside this module.  They are
robot-specific; this module owns the model state, observation layout and safety
decisions that must remain identical across integrations.
"""

from dataclasses import dataclass
from enum import Enum
import time
from typing import Callable, Dict, Optional, Protocol, Sequence, Tuple

import numpy as np


NUM_PROP = 53
NUM_SCAN = 132
NUM_PRIV_EXPLICIT = 9
NUM_PRIV_LATENT = 29
NUM_HIST = 10
NUM_ACTIONS = 12
ACTOR_OBS_DIM = 753
DEPTH_HEIGHT = 58
DEPTH_WIDTH = 87
DEPTH_LATENT_DIM = 32
DEPTH_YAW_DIM = 2
GRU_HIDDEN_DIM = 512

# URDF/hardware: FR, FL, RR, RL. Policy: FL, FR, RL, RR.
JOINT_HARDWARE_TO_POLICY = np.asarray(
    [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
    dtype=np.int64,
)
FOOT_HARDWARE_TO_POLICY = np.asarray([1, 0, 3, 2], dtype=np.int64)

DEFAULT_JOINT_ANGLES_POLICY = (
    0.1,
    0.8,
    -1.5,
    -0.1,
    0.8,
    -1.5,
    0.1,
    1.0,
    -1.5,
    -0.1,
    1.0,
    -1.5,
)


class InferenceBackend(Protocol):
    """Backend contract implemented by the S100 HBM adapter and test doubles."""

    def infer_depth(
        self,
        depth_image: np.ndarray,
        proprio: np.ndarray,
        h_in: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...

    def infer_actor(
        self,
        actor_obs: np.ndarray,
        depth_latent: np.ndarray,
    ) -> np.ndarray:
        ...


class ControlMode(str, Enum):
    NORMAL = "normal"
    HOLD = "hold"
    DAMPING = "damping"
    ESTOP = "estop"


@dataclass(frozen=True)
class RuntimeConfig:
    control_frequency_hz: float = 50.0
    depth_update_interval: int = 5
    sensor_timeout_s: float = 0.05
    inference_timeout_s: float = 0.02
    max_consecutive_inference_failures: int = 3
    max_consecutive_depth_errors: int = 3
    roll_limit_rad: float = 1.5
    pitch_limit_rad: float = 1.5
    action_scale: float = 0.25
    # Simulation clips target displacement to +/-1.2 rad, hence 1.2 / 0.25.
    action_clip: float = 4.8
    action_limit_fraction: float = 1.0
    yaw_scale: float = 1.5
    depth_min: float = -0.5
    depth_max: float = 0.5
    depth_range_tolerance: float = 0.05
    default_joint_angles: Tuple[float, ...] = DEFAULT_JOINT_ANGLES_POLICY

    def __post_init__(self) -> None:
        if self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive")
        if self.depth_update_interval <= 0:
            raise ValueError("depth_update_interval must be positive")
        if self.sensor_timeout_s <= 0 or self.inference_timeout_s <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_consecutive_inference_failures <= 0:
            raise ValueError("max_consecutive_inference_failures must be positive")
        if self.max_consecutive_depth_errors <= 0:
            raise ValueError("max_consecutive_depth_errors must be positive")
        if not 0 < self.action_limit_fraction <= 1:
            raise ValueError("action_limit_fraction must be in (0, 1]")
        if len(self.default_joint_angles) != NUM_ACTIONS:
            raise ValueError("default_joint_angles must contain 12 values")


@dataclass(frozen=True)
class PolicyInput:
    """One control-tick input.

    ``proprio`` is the normalized 53-vector in policy order. ``depth_image`` is
    already cropped, resized and normalized to the training range.
    ``timestamp_s`` must use the same monotonic clock as the runtime.
    """

    proprio: np.ndarray
    timestamp_s: float
    depth_image: Optional[np.ndarray] = None
    reset: bool = False


@dataclass(frozen=True)
class ControlCommand:
    mode: ControlMode
    action: np.ndarray
    target_joint_angle: np.ndarray
    reason: str = ""
    inference_time_s: float = 0.0
    clipped_dimensions: int = 0


@dataclass
class _RuntimeStats:
    steps: int = 0
    normal_steps: int = 0
    clipped_dimensions: int = 0
    inference_failures: int = 0
    depth_errors: int = 0
    max_inference_time_s: float = 0.0


def _vector(name: str, value: Sequence[float], size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def hardware_to_policy_order(values: Sequence[float]) -> np.ndarray:
    """Convert 12 joint values from FR/FL/RR/RL to FL/FR/RL/RR."""

    return _vector("joint values", values, NUM_ACTIONS)[JOINT_HARDWARE_TO_POLICY]


def policy_to_hardware_order(values: Sequence[float]) -> np.ndarray:
    """Convert 12 joint values from FL/FR/RL/RR to FR/FL/RR/RL."""

    # This permutation is its own inverse.
    return _vector("joint values", values, NUM_ACTIONS)[JOINT_HARDWARE_TO_POLICY]


def build_proprio(
    base_angular_velocity: Sequence[float],
    roll_pitch: Sequence[float],
    delta_yaw: float,
    delta_next_yaw: float,
    forward_command: float,
    is_parkour: bool,
    joint_position: Sequence[float],
    default_joint_position: Sequence[float],
    joint_velocity: Sequence[float],
    previous_action: Sequence[float],
    foot_contacts: Sequence[float],
    hardware_order: bool = True,
) -> np.ndarray:
    """Build the normalized 53-vector used by ``compute_observations()``.

    Joint arrays are assumed to use URDF order (FR, FL, RR, RL) unless
    ``hardware_order`` is false. ``previous_action`` is always policy order.
    Foot contacts use hardware order FR, FL, RR, RL when ``hardware_order`` is
    true and may be booleans or 0/1 values.
    """

    angular_velocity = _vector("base_angular_velocity", base_angular_velocity, 3)
    attitude = _vector("roll_pitch", roll_pitch, 2)
    positions = _vector("joint_position", joint_position, NUM_ACTIONS)
    defaults = _vector("default_joint_position", default_joint_position, NUM_ACTIONS)
    velocities = _vector("joint_velocity", joint_velocity, NUM_ACTIONS)
    last_action = _vector("previous_action", previous_action, NUM_ACTIONS)
    contacts = _vector("foot_contacts", foot_contacts, 4)

    if hardware_order:
        positions = positions[JOINT_HARDWARE_TO_POLICY]
        defaults = defaults[JOINT_HARDWARE_TO_POLICY]
        velocities = velocities[JOINT_HARDWARE_TO_POLICY]
        contacts = contacts[FOOT_HARDWARE_TO_POLICY]

    proprio = np.concatenate(
        (
            angular_velocity * np.float32(0.25),
            attitude,
            np.asarray([0.0, delta_yaw, delta_next_yaw], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.asarray(
                [forward_command, float(not is_parkour), float(is_parkour)],
                dtype=np.float32,
            ),
            positions - defaults,
            velocities * np.float32(0.05),
            last_action,
            contacts - np.float32(0.5),
        )
    )
    if proprio.shape != (NUM_PROP,):
        raise AssertionError(f"internal proprio size error: {proprio.shape}")
    if not np.all(np.isfinite(proprio)):
        raise ValueError("proprio contains NaN or Inf")
    return np.ascontiguousarray(proprio, dtype=np.float32)


class S100PolicyRuntime:
    """Run both deployment models and own their recurrent state."""

    def __init__(
        self,
        backend: InferenceBackend,
        config: RuntimeConfig = RuntimeConfig(),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.config = config
        self._clock = clock
        self._default_joint_angles = np.asarray(
            config.default_joint_angles,
            dtype=np.float32,
        )
        self._stats = _RuntimeStats()
        self.reset()

    def reset(self) -> None:
        self._hidden = np.zeros((1, 1, GRU_HIDDEN_DIM), dtype=np.float32)
        self._history = np.zeros((NUM_HIST, NUM_PROP), dtype=np.float32)
        self._depth_latent: Optional[np.ndarray] = None
        self._yaw = np.zeros((1, DEPTH_YAW_DIM), dtype=np.float32)
        self._last_safe_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self._last_safe_target = self._default_joint_angles.copy()
        self._step_count = 0
        self._consecutive_inference_failures = 0
        self._consecutive_depth_errors = 0
        self._estop_latched = False

    @property
    def stats(self) -> Dict[str, float]:
        steps = max(self._stats.steps, 1)
        return {
            "steps": float(self._stats.steps),
            "normal_steps": float(self._stats.normal_steps),
            "inference_failures": float(self._stats.inference_failures),
            "depth_errors": float(self._stats.depth_errors),
            "action_saturation_rate": (
                self._stats.clipped_dimensions / (steps * NUM_ACTIONS)
            ),
            "max_inference_time_s": self._stats.max_inference_time_s,
        }

    def step(self, policy_input: PolicyInput) -> ControlCommand:
        if policy_input.reset:
            self.reset()
        self._stats.steps += 1
        tick = self._step_count
        self._step_count += 1

        if self._estop_latched:
            return self._safe_command(ControlMode.ESTOP, "estop_latched")

        now = self._clock()
        if not np.isfinite(policy_input.timestamp_s):
            return self._safe_command(ControlMode.DAMPING, "invalid_sensor_timestamp")
        sensor_age = now - policy_input.timestamp_s
        if sensor_age > self.config.sensor_timeout_s:
            return self._safe_command(
                ControlMode.DAMPING,
                f"sensor_timeout:{sensor_age:.6f}s",
            )

        try:
            proprio = _vector("proprio", policy_input.proprio, NUM_PROP)
        except ValueError as exc:
            return self._latch_estop(str(exc))

        roll = float(proprio[3])
        pitch = float(proprio[4])
        if abs(roll) > self.config.roll_limit_rad:
            return self._latch_estop(f"roll_limit:{roll:.6f}")
        if abs(pitch) > self.config.pitch_limit_rad:
            return self._latch_estop(f"pitch_limit:{pitch:.6f}")

        depth_reason = ""
        if tick % self.config.depth_update_interval == 0:
            if not self._valid_depth(policy_input.depth_image):
                self._consecutive_depth_errors += 1
                self._stats.depth_errors += 1
                depth_reason = "stale_depth"
                if (
                    self._depth_latent is None
                    or self._consecutive_depth_errors
                    >= self.config.max_consecutive_depth_errors
                ):
                    return self._safe_command(
                        ControlMode.DAMPING,
                        "depth_invalid",
                    )
            else:
                depth_proprio = proprio.copy()
                depth_proprio[6:8] = 0.0
                started = self._clock()
                try:
                    depth_latent, yaw, h_out = self.backend.infer_depth(
                        np.ascontiguousarray(
                            policy_input.depth_image[None, ...],
                            dtype=np.float32,
                        ),
                        depth_proprio[None, ...],
                        self._hidden,
                    )
                except Exception as exc:  # Board runtime errors are backend-specific.
                    return self._inference_failure(f"depth_error:{exc}", proprio)
                elapsed = self._clock() - started
                if elapsed > self.config.inference_timeout_s:
                    return self._inference_failure(
                        f"depth_timeout:{elapsed:.6f}s",
                        proprio,
                        elapsed,
                    )
                try:
                    new_latent = self._model_output(
                        "depth_latent",
                        depth_latent,
                        (1, DEPTH_LATENT_DIM),
                    )
                    new_yaw = self._model_output(
                        "yaw_correction",
                        yaw,
                        (1, DEPTH_YAW_DIM),
                    )
                    new_hidden = self._model_output(
                        "h_out",
                        h_out,
                        (1, 1, GRU_HIDDEN_DIM),
                    )
                except ValueError as exc:
                    return self._latch_estop(str(exc))
                self._depth_latent = new_latent
                self._yaw = new_yaw
                self._hidden = new_hidden
                self._consecutive_depth_errors = 0
                self._record_inference_time(elapsed)

        if (
            self._consecutive_depth_errors
            >= self.config.max_consecutive_depth_errors
        ):
            return self._safe_command(ControlMode.DAMPING, "depth_unavailable")
        if self._depth_latent is None:
            return self._safe_command(ControlMode.DAMPING, "depth_not_initialized")

        actor_proprio = proprio.copy()
        actor_proprio[6:8] = self.config.yaw_scale * self._yaw[0]
        actor_obs = self._build_actor_obs(actor_proprio)

        started = self._clock()
        try:
            action_output = self.backend.infer_actor(
                actor_obs[None, ...],
                self._depth_latent,
            )
        except Exception as exc:
            return self._inference_failure(f"actor_error:{exc}", proprio)
        elapsed = self._clock() - started
        if elapsed > self.config.inference_timeout_s:
            return self._inference_failure(
                f"actor_timeout:{elapsed:.6f}s",
                proprio,
                elapsed,
            )
        try:
            action = self._model_output(
                "action",
                action_output,
                (1, NUM_ACTIONS),
            )[0]
        except ValueError as exc:
            return self._latch_estop(str(exc))

        self._append_history(proprio)
        self._consecutive_inference_failures = 0
        self._record_inference_time(elapsed)

        action_limit = self.config.action_clip * self.config.action_limit_fraction
        clipped_action = np.clip(action, -action_limit, action_limit)
        clipped_dimensions = int(np.count_nonzero(clipped_action != action))
        target = (
            self._default_joint_angles
            + np.float32(self.config.action_scale) * clipped_action
        )
        self._last_safe_action = clipped_action.copy()
        self._last_safe_target = target.copy()
        self._stats.normal_steps += 1
        self._stats.clipped_dimensions += clipped_dimensions
        return ControlCommand(
            mode=ControlMode.NORMAL,
            action=clipped_action,
            target_joint_angle=target,
            reason=depth_reason,
            inference_time_s=elapsed,
            clipped_dimensions=clipped_dimensions,
        )

    def _valid_depth(self, depth_image: Optional[np.ndarray]) -> bool:
        if depth_image is None:
            return False
        depth = np.asarray(depth_image)
        if depth.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
            return False
        if not np.all(np.isfinite(depth)):
            return False
        low = self.config.depth_min - self.config.depth_range_tolerance
        high = self.config.depth_max + self.config.depth_range_tolerance
        return bool(np.min(depth) >= low and np.max(depth) <= high)

    @staticmethod
    def _model_output(
        name: str,
        value: np.ndarray,
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        output = np.asarray(value, dtype=np.float32)
        if output.shape != shape:
            raise ValueError(f"{name} has shape {output.shape}, expected {shape}")
        if not np.all(np.isfinite(output)):
            raise ValueError(f"{name} contains NaN or Inf")
        return np.ascontiguousarray(output)

    def _build_actor_obs(self, proprio: np.ndarray) -> np.ndarray:
        actor_obs = np.zeros(ACTOR_OBS_DIM, dtype=np.float32)
        actor_obs[:NUM_PROP] = proprio
        history_start = (
            NUM_PROP + NUM_SCAN + NUM_PRIV_EXPLICIT + NUM_PRIV_LATENT
        )
        actor_obs[history_start:] = self._history.reshape(-1)
        return actor_obs

    def _append_history(self, proprio: np.ndarray) -> None:
        history_proprio = proprio.copy()
        history_proprio[6:8] = 0.0
        self._history[:-1] = self._history[1:]
        self._history[-1] = history_proprio

    def _record_inference_time(self, elapsed: float) -> None:
        self._stats.max_inference_time_s = max(
            self._stats.max_inference_time_s,
            elapsed,
        )

    def _inference_failure(
        self,
        reason: str,
        proprio: np.ndarray,
        elapsed: float = 0.0,
    ) -> ControlCommand:
        self._append_history(proprio)
        self._consecutive_inference_failures += 1
        self._stats.inference_failures += 1
        self._record_inference_time(elapsed)
        if (
            self._consecutive_inference_failures
            >= self.config.max_consecutive_inference_failures
        ):
            return self._latch_estop(reason)
        return self._safe_command(
            ControlMode.HOLD,
            reason,
            action=self._last_safe_action,
            target=self._last_safe_target,
            inference_time_s=elapsed,
        )

    def _latch_estop(self, reason: str) -> ControlCommand:
        self._estop_latched = True
        return self._safe_command(ControlMode.ESTOP, reason)

    def _safe_command(
        self,
        mode: ControlMode,
        reason: str,
        action: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        inference_time_s: float = 0.0,
    ) -> ControlCommand:
        if action is None:
            action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if target is None:
            target = self._default_joint_angles
        return ControlCommand(
            mode=mode,
            action=np.asarray(action, dtype=np.float32).copy(),
            target_joint_angle=np.asarray(target, dtype=np.float32).copy(),
            reason=reason,
            inference_time_s=inference_time_s,
        )
