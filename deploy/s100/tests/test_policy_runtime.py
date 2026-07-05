import unittest

import numpy as np

from deploy.s100.runtime.policy_runtime import (
    ControlMode,
    PolicyInput,
    RuntimeConfig,
    S100PolicyRuntime,
    build_proprio,
    policy_to_hardware_order,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeBackend:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.delay_s = 0.0
        self.nan_action = False
        self.depth_calls = []
        self.actor_calls = []

    def infer_depth(self, depth_image, proprio, h_in):
        self.depth_calls.append((depth_image.copy(), proprio.copy(), h_in.copy()))
        self.clock.now += self.delay_s
        return (
            np.full((1, 32), 0.25, dtype=np.float32),
            np.asarray([[0.2, -0.1]], dtype=np.float32),
            h_in + 1.0,
        )

    def infer_actor(self, actor_obs, depth_latent):
        self.actor_calls.append((actor_obs.copy(), depth_latent.copy()))
        self.clock.now += self.delay_s
        action = np.arange(12, dtype=np.float32)[None, :] / 10.0
        if self.nan_action:
            action[0, 0] = np.nan
        return action


class PolicyRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.backend = FakeBackend(self.clock)
        self.runtime = S100PolicyRuntime(
            self.backend,
            RuntimeConfig(depth_update_interval=5),
            clock=self.clock,
        )
        self.depth = np.zeros((58, 87), dtype=np.float32)
        self.proprio = np.arange(53, dtype=np.float32) / 100.0
        self.proprio[3:5] = 0.0

    def policy_input(self, depth=True):
        return PolicyInput(
            proprio=self.proprio.copy(),
            depth_image=self.depth.copy() if depth else None,
            timestamp_s=self.clock.now,
        )

    def test_state_and_observation_layout(self):
        first = self.runtime.step(self.policy_input())
        self.assertEqual(first.mode, ControlMode.NORMAL)
        self.assertEqual(len(self.backend.depth_calls), 1)
        depth_proprio = self.backend.depth_calls[0][1]
        np.testing.assert_array_equal(depth_proprio[0, 6:8], np.zeros(2))

        first_actor_obs = self.backend.actor_calls[0][0][0]
        np.testing.assert_allclose(first_actor_obs[6:8], [0.3, -0.15])
        np.testing.assert_array_equal(first_actor_obs[-530:], np.zeros(530))

        second = self.runtime.step(self.policy_input(depth=False))
        self.assertEqual(second.mode, ControlMode.NORMAL)
        self.assertEqual(len(self.backend.depth_calls), 1)
        second_actor_obs = self.backend.actor_calls[1][0][0]
        expected_history_tail = self.proprio.copy()
        expected_history_tail[6:8] = 0.0
        np.testing.assert_allclose(second_actor_obs[-53:], expected_history_tail)

    def test_stale_depth_then_damping(self):
        self.runtime = S100PolicyRuntime(
            self.backend,
            RuntimeConfig(
                depth_update_interval=1,
                max_consecutive_depth_errors=2,
            ),
            clock=self.clock,
        )
        self.assertEqual(self.runtime.step(self.policy_input()).mode, ControlMode.NORMAL)
        stale = self.runtime.step(self.policy_input(depth=False))
        self.assertEqual(stale.mode, ControlMode.NORMAL)
        self.assertEqual(stale.reason, "stale_depth")
        stopped = self.runtime.step(self.policy_input(depth=False))
        self.assertEqual(stopped.mode, ControlMode.DAMPING)
        self.assertEqual(stopped.reason, "depth_invalid")
        still_stopped = self.runtime.step(self.policy_input(depth=False))
        self.assertEqual(still_stopped.mode, ControlMode.DAMPING)

    def test_repeated_inference_timeout_latches_estop(self):
        self.backend.delay_s = 0.02
        runtime = S100PolicyRuntime(
            self.backend,
            RuntimeConfig(
                depth_update_interval=1,
                inference_timeout_s=0.01,
                max_consecutive_inference_failures=2,
            ),
            clock=self.clock,
        )
        first = runtime.step(self.policy_input())
        self.assertEqual(first.mode, ControlMode.HOLD)
        second = runtime.step(self.policy_input())
        self.assertEqual(second.mode, ControlMode.ESTOP)
        self.backend.delay_s = 0.0
        latched = runtime.step(self.policy_input())
        self.assertEqual(latched.mode, ControlMode.ESTOP)
        self.assertEqual(latched.reason, "estop_latched")

    def test_nan_action_immediately_latches_estop(self):
        self.backend.nan_action = True
        command = self.runtime.step(self.policy_input())
        self.assertEqual(command.mode, ControlMode.ESTOP)
        self.assertIn("NaN or Inf", command.reason)

    def test_sensor_timeout_uses_damping_mode(self):
        policy_input = self.policy_input()
        self.clock.now += 0.1
        command = self.runtime.step(policy_input)
        self.assertEqual(command.mode, ControlMode.DAMPING)
        self.assertIn("sensor_timeout", command.reason)

    def test_build_proprio_reorders_hardware_arrays(self):
        hardware_values = np.arange(12, dtype=np.float32)
        proprio = build_proprio(
            base_angular_velocity=[4.0, 8.0, 12.0],
            roll_pitch=[0.1, -0.2],
            delta_yaw=0.3,
            delta_next_yaw=-0.4,
            forward_command=0.5,
            is_parkour=True,
            joint_position=hardware_values,
            default_joint_position=np.zeros(12),
            joint_velocity=hardware_values,
            previous_action=np.zeros(12),
            foot_contacts=[0, 1, 0, 1],
        )
        np.testing.assert_allclose(proprio[:3], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            proprio[13:25],
            [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
        )
        np.testing.assert_allclose(proprio[-4:], [0.5, -0.5, 0.5, -0.5])
        np.testing.assert_array_equal(
            policy_to_hardware_order(proprio[13:25]),
            hardware_values,
        )


if __name__ == "__main__":
    unittest.main()
