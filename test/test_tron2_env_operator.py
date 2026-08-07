# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hardware-free tests for the tron2_env-backed control-mode adapter."""

import threading
import time

import numpy as np
import pytest

from fluxvla.engines.operators.tron2_env_operator import Tron2EnvOperator
from fluxvla.engines.operators.tron2_native_env_operator import (
    Tron2NativeEnvOperator, )
from fluxvla.engines.runners.tron2_inference_runner import (
    Tron2InferenceRunner, )


class FakeTransport:

    def __init__(self):
        self.connected = True
        self.disconnected = False
        self.state = np.zeros(18, dtype=np.float64)
        self.movej_calls = []
        self.gripper_calls = []
        self.head_calls = []

    def is_connected(self):
        return self.connected

    def disconnect(self):
        self.connected = False
        self.disconnected = True

    def get_joint_state(self, timeout=1.0):
        del timeout
        return {'states': self.state.copy().tolist()}

    def set_gripper(self, left_opening, right_opening):
        self.gripper_calls.append((left_opening, right_opening))

    def movej(self, target, move_time):
        self.movej_calls.append((np.asarray(target).copy(), move_time))

    def wait_until_reached(self, target, tolerance, timeout):
        del target, tolerance, timeout
        return True

    def move_head(self, target, move_time):
        self.head_calls.append((np.asarray(target).copy(), move_time))


class FakeMotionController:

    def __init__(self, transport, publish_rate, eta_default):
        self.transport = transport
        self.publish_rate = publish_rate
        self.eta_default = eta_default
        self.started = False
        self.commands = []
        self.gripper_calls = []

    def start(self):
        self.started = True

    def disconnect(self):
        self.transport.disconnect()

    def get_joint_states(self, timeout=1.0):
        return self.transport.get_joint_state(timeout)

    def get_head_position(self):
        return self.transport.state[16:18].copy()

    def is_connected(self):
        return self.transport.is_connected()

    def set_gripper(self, left_opening, right_opening):
        self.gripper_calls.append((left_opening, right_opening))

    def command_joints(self, target, eta=None):
        self.commands.append((np.asarray(target).copy(), eta))


class FakeNativeConfig:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeNativeRobot:

    def __init__(self, events):
        self.events = events
        self.gripper_calls = []

    def set_gripper(self, left_opening, right_opening):
        call = (float(left_opening), float(right_opening))
        self.gripper_calls.append(call)
        self.events.append(('gripper', call))


class FakeNativeEnv:

    instances = []
    events = []

    def __init__(self, config):
        self.config = config
        self.closed = False
        self.steps = []
        self.last_action = None
        self.robot = FakeNativeRobot(type(self).events)
        type(self).instances.append(self)
        type(self).events.append(('create', self))

    def get_obs(self):
        return make_bridge_observation()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.steps.append(action.copy())
        if len(action) == 16:
            self.last_action = np.concatenate(
                [action[:7], action[8:15], [0.4, -0.3]])
        else:
            self.last_action = np.concatenate(
                [action[:7], action[8:15], action[16:18]])

    def close(self):
        self.closed = True
        type(self).events.append(('close', self))


def make_operator(transports):
    operator = Tron2EnvOperator.__new__(Tron2EnvOperator)
    operator.servoj_publish_rate = 300.0
    operator.recovery_blend_frames = 6
    operator.chunk_boundary_blend_enabled = False
    operator.chunk_boundary_blend_frames = 6
    operator.chunk_boundary_blend_scope = 'arm'
    operator.state_polling_rate = 200.0
    operator.connection_timeout = 1.0
    operator.state_timeout = 1.0
    operator.movej_duration = 2.0
    operator.movej_tolerance = 0.05
    operator.movej_timeout = 10.0
    operator.max_servoj_step_rad = 0.2
    operator.max_state_source_mismatch_rad = None
    operator.lock_head = True
    operator.max_head_hold_error_rad = 0.05
    operator.servoj_joint_lower_limits = None
    operator.servoj_joint_upper_limits = None
    operator.robot_ip = 'fake'
    operator.ws_port = 5000
    operator._control_lock = threading.RLock()
    operator._transport = None
    operator._motion_controller = None
    operator._bridge_provider = None
    operator._pending_bridge_observation = None
    operator._latest_bridge_state = np.zeros(18, dtype=np.float64)
    operator._head_hold_position = None
    operator._closed = False
    operator._trajectory_error = None
    operator._traj_thread = None
    operator._traj_stop_event = threading.Event()
    operator._last_servoj_waypoint = None
    operator._last_left_gripper = None
    operator._last_right_gripper = None
    operator._recovery_blend_pending = False
    operator._active_servoj_waypoints = None
    operator._active_left_gripper = None
    operator._active_right_gripper = None
    operator._active_next_index = 0
    operator._issued_action_callback = None
    operator._runtime_types = lambda: (FakeMotionController, None, None)
    operator._create_transport = lambda: transports.pop(0)
    return operator


def make_bridge_observation():
    return {
        'images': {
            'cam_high': np.zeros((4, 6, 3), dtype=np.uint8),
            'cam_left_wrist': np.ones((4, 6, 3), dtype=np.uint8),
            'cam_right_wrist': np.full((4, 6, 3), 2, dtype=np.uint8),
        },
        'state': np.arange(18, dtype=np.float32),
        'metadata': {
            'observation_source': 'bridge'
        },
    }


def make_native_operator(monkeypatch):
    FakeNativeEnv.instances = []
    FakeNativeEnv.events = []
    runtime_types = (FakeNativeConfig, FakeNativeConfig, FakeNativeConfig,
                     FakeNativeConfig, FakeNativeEnv)
    monkeypatch.setattr(
        Tron2NativeEnvOperator,
        '_runtime_types',
        staticmethod(lambda: runtime_types),
    )
    return Tron2NativeEnvOperator(
        robot_ip='10.192.1.2',
        bridge_host='wss://10.192.1.4',
        bridge_state_source='legacy',
        state_dim=18,
        fps=30.0,
        publish_rate=300.0,
        init_joints=np.zeros(14),
        init_head=None,
        reset_gripper_open_wait_s=0.0,
    )


def test_native_operator_delegates_observation_and_step(monkeypatch):
    operator = make_native_operator(monkeypatch)
    env = FakeNativeEnv.instances[-1]
    issued = []
    operator.set_issued_action_callback(lambda **record: issued.append(record))

    observation = operator.get_observation()
    operator.execute_waypoint(
        left_arm=np.arange(7, dtype=np.float64) / 10.0,
        right_arm=-np.arange(7, dtype=np.float64) / 10.0,
        left_gripper=-0.2,
        right_gripper=1.4,
        trajectory_index=3,
    )

    assert observation['state'].shape == (18, )
    assert env.config.observation_source == 'bridge'
    assert env.config.bridge_state_source == 'legacy'
    assert env.config.fps == 30.0
    assert env.config.publish_rate == 300.0
    assert len(env.steps) == 1
    np.testing.assert_allclose(
        env.steps[0],
        np.concatenate([
            np.arange(7, dtype=np.float64) / 10.0,
            [-0.2],
            -np.arange(7, dtype=np.float64) / 10.0,
            [1.4],
        ]),
    )
    assert issued[0]['trajectory_index'] == 3
    assert issued[0]['left_gripper'] == pytest.approx(0.0)
    assert issued[0]['right_gripper'] == pytest.approx(1.0)
    np.testing.assert_allclose(issued[0]['waypoint'][-2:], [0.4, -0.3])


def test_native_operator_executes_non_rtc_trajectory_at_policy_rate(
        monkeypatch):
    operator = make_native_operator(monkeypatch)
    env = FakeNativeEnv.instances[-1]
    issued = []
    operator.set_issued_action_callback(lambda **record: issued.append(record))
    clock = {'now': 10.0}
    sleeps = []

    monkeypatch.setattr(time, 'perf_counter', lambda: clock['now'])

    def advance_clock(delay):
        sleeps.append(delay)
        clock['now'] += delay

    monkeypatch.setattr(time, 'sleep', advance_clock)
    left = np.arange(21, dtype=np.float64).reshape(3, 7) / 100.0
    right = -left

    operator.execute_trajectory(
        left_arm_trajectory=left,
        right_arm_trajectory=right,
        left_gripper_trajectory=[0.1, 0.2, 0.3],
        right_gripper_trajectory=[0.9, 0.8, 0.7],
        dt=1.0 / 30.0,
        action_metadata={
            'trajectory_id': 8,
            'rtc_enabled': False,
        },
    )

    assert len(env.steps) == 3
    np.testing.assert_allclose(env.steps[0][:7], left[0])
    np.testing.assert_allclose(env.steps[-1][8:15], right[-1])
    assert all(len(action) == 16 for action in env.steps)
    assert [record['trajectory_index'] for record in issued] == [0, 1, 2]
    assert all(record['action_metadata']['rtc_enabled'] is False
               for record in issued)
    assert sleeps == pytest.approx([1.0 / 30.0] * 3)


def test_native_operator_rejects_invalid_trajectory_before_execution(
        monkeypatch):
    operator = make_native_operator(monkeypatch)
    env = FakeNativeEnv.instances[-1]

    with pytest.raises(ValueError, match='same number of frames'):
        operator.execute_trajectory(
            left_arm_trajectory=np.zeros((2, 7)),
            right_arm_trajectory=np.zeros((1, 7)),
            left_gripper_trajectory=np.zeros(2),
            right_gripper_trajectory=np.zeros(2),
        )
    with pytest.raises(ValueError, match='synchronous'):
        operator.execute_trajectory(
            left_arm_trajectory=np.zeros((1, 7)),
            right_arm_trajectory=np.zeros((1, 7)),
            left_gripper_trajectory=np.zeros(1),
            right_gripper_trajectory=np.zeros(1),
            async_exec=True,
        )

    assert env.steps == []


def test_native_reset_reconstructs_upstream_environment(monkeypatch):
    operator = make_native_operator(monkeypatch)
    first = FakeNativeEnv.instances[-1]

    operator.reset_native_env()

    second = FakeNativeEnv.instances[-1]
    assert first.closed is True
    assert second is not first
    assert operator.native_env is second
    assert first.robot.gripper_calls == [(100.0, 100.0)]
    assert FakeNativeEnv.events == [
        ('create', first),
        ('gripper', (100.0, 100.0)),
        ('close', first),
        ('create', second),
    ]


def test_native_reset_does_not_move_when_gripper_open_fails(monkeypatch):
    operator = make_native_operator(monkeypatch)
    first = FakeNativeEnv.instances[-1]

    def fail_to_open(**kwargs):
        del kwargs
        raise RuntimeError('gripper unavailable')

    first.robot.set_gripper = fail_to_open
    with pytest.raises(RuntimeError, match='gripper unavailable'):
        operator.reset_native_env()

    assert first.closed is False
    assert operator.native_env is first
    assert FakeNativeEnv.instances == [first]


def test_plain_runner_delegates_prepare_pose_to_native_reset():
    reset_calls = []
    runner = Tron2InferenceRunner.__new__(Tron2InferenceRunner)
    runner.ros_operator = type(
        'NativeResetOperator', (),
        {'reset_native_env': lambda self: reset_calls.append(True)})()
    runner.dry_run = False
    runner.last_actions = np.ones((2, 16))

    runner._move_to_prepare_pose()

    assert reset_calls == [True]
    assert runner.last_actions is None


class FakeBridgeProvider:

    def __init__(self, observation):
        self.observation = observation
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def get_obs(self, timeout):
        assert timeout > 0
        return self.observation


def test_operator_uses_bridge_without_ros(monkeypatch):
    provider = FakeBridgeProvider(make_bridge_observation())
    monkeypatch.setattr(Tron2EnvOperator, '_create_bridge_provider',
                        lambda self: provider)

    operator = Tron2EnvOperator(
        bridge_host='wss://bridge.test',
        connect_websocket=False,
    )

    assert provider.started is True
    observation = operator.get_observation()
    assert observation['state'].shape == (18, )
    assert set(observation['images']) == {
        'cam_high', 'cam_left_wrist', 'cam_right_wrist'
    }
    operator.close()
    assert provider.stopped is True


@pytest.mark.parametrize(
    ('layout', 'expected'),
    [
        ('tron2_16', np.arange(16, dtype=np.float32)),
        ('tron2_18',
         np.array(
             [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 7, 15],
             dtype=np.float32)),
    ],
)
def test_runner_maps_bridge_state_to_training_layout(layout, expected):
    runner = Tron2InferenceRunner.__new__(Tron2InferenceRunner)
    runner.action_layout = layout
    runner.include_head_in_state = False
    runner.camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    runner.observation_window = None
    runner.ros_operator = type(
        'FakeObservationOperator', (),
        {'get_observation': lambda self: make_bridge_observation()})()
    runner._apply_jpeg_compression = lambda image: image

    observation = runner.update_observation_window()

    np.testing.assert_array_equal(observation['qpos'], expected)
    assert observation['cam_high'].shape == (4, 6, 3)


def test_runner_keeps_measured_head_in_tron2_16_model_state():
    runner = Tron2InferenceRunner.__new__(Tron2InferenceRunner)
    runner.action_layout = 'tron2_16'
    runner.include_head_in_state = True
    runner.camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    runner.observation_window = None
    runner.ros_operator = type(
        'FakeObservationOperator', (),
        {'get_observation': lambda self: make_bridge_observation()})()
    runner._apply_jpeg_compression = lambda image: image

    observation = runner.update_observation_window()

    np.testing.assert_array_equal(observation['qpos'],
                                  np.arange(18, dtype=np.float32))
    np.testing.assert_array_equal(runner._latest_head_position, [16.0, 17.0])


def test_prepare_uses_movej_then_policy_uses_servoj():
    transport = FakeTransport()
    operator = make_operator([transport])

    operator.move_to_targets(
        np.full(7, 0.1),
        np.full(7, -0.1),
        left_gripper=1.0,
        right_gripper=0.5,
    )

    assert len(transport.movej_calls) == 1
    np.testing.assert_allclose(
        transport.movej_calls[0][0],
        np.concatenate([np.full(7, 0.1), np.full(7, -0.1)]),
    )
    assert transport.gripper_calls == [(100.0, 50.0)]
    assert operator._motion_controller is None

    left = np.array([[0.01] * 7, [0.02] * 7])
    right = -left
    operator.execute_trajectory(
        left,
        right,
        np.array([1.0, 0.9]),
        np.array([0.5, 0.4]),
        dt=0.001,
    )

    controller = operator._motion_controller
    assert controller.started is True
    assert controller.publish_rate == 300.0
    assert len(controller.commands) == 2
    np.testing.assert_allclose(
        controller.commands[0][0],
        np.concatenate([left[0], right[0], np.zeros(2)]),
    )
    assert controller.commands[0][1] == pytest.approx(0.001)
    assert controller.gripper_calls == [(100.0, 50.0), (90.0, 40.0)]


def test_operator_reports_only_policy_waypoints_issued_to_servoj():
    transport = FakeTransport()
    operator = make_operator([transport])
    issued = []
    operator.set_issued_action_callback(lambda **record: issued.append(record))
    left = np.array([[0.01] * 7, [0.02] * 7])
    right = -left

    operator.execute_trajectory(
        left,
        right,
        np.array([0.25, 0.5]),
        np.array([0.75, 1.0]),
        dt=0.001,
        action_metadata={
            'trajectory_id': 7,
            'rtc_enabled': True,
        },
    )

    assert len(issued) == 2
    np.testing.assert_allclose(issued[0]['waypoint'][:7], left[0])
    np.testing.assert_allclose(issued[1]['waypoint'][7:14], right[1])
    assert issued[0]['left_gripper'] == pytest.approx(0.25)
    assert issued[1]['right_gripper'] == pytest.approx(1.0)
    assert issued[0]['trajectory_index'] == 0
    assert issued[1]['trajectory_index'] == 1
    assert issued[0]['action_metadata'] == {
        'trajectory_id': 7,
        'rtc_enabled': True,
    }
    assert issued[0]['issued_at_unix'] > 0
    assert issued[0]['issued_at_monotonic'] > 0


def test_persistent_waypoint_stream_keeps_one_motion_controller():
    transport = FakeTransport()
    operator = make_operator([transport])
    issued = []
    operator.set_issued_action_callback(lambda **record: issued.append(record))

    operator.begin_waypoint_stream()
    operator.execute_waypoint(
        left_arm=np.full(7, 0.01),
        right_arm=np.full(7, -0.01),
        left_gripper=0.2,
        right_gripper=0.8,
        dt=1.0 / 30.0,
        trajectory_index=3,
    )
    controller = operator._motion_controller
    operator.execute_waypoint(
        left_arm=np.full(7, 0.02),
        right_arm=np.full(7, -0.02),
        left_gripper=0.3,
        right_gripper=0.7,
        dt=1.0 / 30.0,
        trajectory_index=4,
    )

    assert operator._motion_controller is controller
    assert controller.started is True
    assert len(controller.commands) == 2
    assert controller.commands[0][1] is None
    assert controller.commands[1][1] is None
    assert controller.gripper_calls == pytest.approx([(20.0, 80.0),
                                                      (30.0, 70.0)])
    assert [record['trajectory_index'] for record in issued] == [3, 4]
    assert operator._traj_thread is None


def test_native_waypoint_matches_tron2_env_step_head_and_gripper_behavior():
    transport = FakeTransport()
    transport.state[16:18] = [0.4, -0.3]
    operator = make_operator([transport])

    operator.execute_waypoint(
        left_arm=np.arange(7, dtype=np.float64) / 10.0,
        right_arm=-np.arange(7, dtype=np.float64) / 10.0,
        left_gripper=-0.2,
        right_gripper=1.4,
        dt=1.0 / 30.0,
    )

    controller = operator._motion_controller
    expected = np.concatenate([
        np.arange(7, dtype=np.float64) / 10.0,
        -np.arange(7, dtype=np.float64) / 10.0,
        [0.4, -0.3],
    ])
    np.testing.assert_allclose(controller.commands[0][0], expected)
    assert controller.commands[0][1] is None
    assert controller.gripper_calls == [(0.0, 100.0)]


def test_movej_disconnects_active_servoj_controller():
    first_transport = FakeTransport()
    second_transport = FakeTransport()
    operator = make_operator([first_transport, second_transport])

    operator.execute_trajectory(
        np.zeros((1, 7)),
        np.zeros((1, 7)),
        np.zeros(1),
        np.zeros(1),
        dt=0.001,
    )
    controller = operator._motion_controller
    assert len(controller.commands) == 1

    operator.move_to_targets(np.zeros(7), np.zeros(7))

    assert first_transport.disconnected is True
    assert len(second_transport.movej_calls) == 1
    assert operator._motion_controller is None
    assert operator._transport is second_transport


def test_movej_clears_drained_trajectory_recovery_origin():
    first_transport = FakeTransport()
    second_transport = FakeTransport()
    operator = make_operator([first_transport, second_transport])

    operator.execute_trajectory(
        np.zeros((1, 7)),
        np.zeros((1, 7)),
        np.zeros(1),
        np.zeros(1),
        dt=0.001,
    )
    assert operator._recovery_blend_pending is True

    operator.move_to_targets(np.zeros(7), np.zeros(7))
    assert operator._recovery_blend_pending is False
    assert operator._last_servoj_waypoint is None

    target = np.full((1, 7), 0.12)
    operator.execute_trajectory(
        target,
        -target,
        np.ones(1),
        np.full(1, 0.6),
        dt=0.001,
    )

    controller = operator._motion_controller
    np.testing.assert_allclose(controller.commands[0][0][:7], 0.12)
    np.testing.assert_allclose(controller.commands[0][0][7:14], -0.12)
    assert controller.gripper_calls[0] == pytest.approx((100.0, 60.0))


def test_servoj_rejects_large_first_or_inter_waypoint_delta():
    operator = make_operator([])
    left = np.zeros((2, 7))
    right = np.zeros((2, 7))
    left[1, 0] = 0.21

    with pytest.raises(ValueError, match='delta exceeds safety limit'):
        operator._prepare_servoj_trajectory(
            left,
            right,
            np.zeros(2),
            np.zeros(2),
            None,
            np.zeros(18),
        )


def test_invalid_first_chunk_never_starts_servoj_publisher():
    transport = FakeTransport()
    operator = make_operator([transport])
    left = np.zeros((1, 7))
    left[0, 3] = 1.0

    with pytest.raises(ValueError, match='delta exceeds safety limit'):
        operator.execute_trajectory(
            left,
            np.zeros((1, 7)),
            np.zeros(1),
            np.zeros(1),
            dt=0.001,
        )

    assert operator._motion_controller is None
    assert transport.disconnected is False


def test_head_is_locked_and_policy_head_commands_are_rejected():
    operator = make_operator([])
    state = np.zeros(18, dtype=np.float64)
    state[16:18] = [0.4, -0.2]
    operator._latest_bridge_state = state.copy()

    waypoints, _, _ = operator._prepare_servoj_trajectory(
        np.zeros((1, 7)),
        np.zeros((1, 7)),
        np.zeros(1),
        np.zeros(1),
        None,
        state,
    )
    np.testing.assert_allclose(waypoints[0, -2:], [0.4, -0.2])

    with pytest.raises(ValueError, match='Head movement is locked'):
        operator._prepare_servoj_trajectory(
            np.zeros((1, 7)),
            np.zeros((1, 7)),
            np.zeros(1),
            np.zeros(1),
            np.zeros((1, 2)),
            state,
        )


def test_bridge_control_state_mismatch_check_is_disabled():
    operator = make_operator([])
    control_state = np.zeros(18, dtype=np.float64)
    control_state[3] = 0.51
    left = np.zeros((1, 7), dtype=np.float64)
    left[0, 3] = 0.51

    waypoints, _, _ = operator._prepare_servoj_trajectory(
        left,
        np.zeros((1, 7)),
        np.zeros(1),
        np.zeros(1),
        None,
        control_state,
    )

    assert waypoints[0, 3] == pytest.approx(0.51)


def test_rejected_active_trajectory_keeps_last_servoj_target():
    transport = FakeTransport()
    operator = make_operator([transport])
    trajectory = np.zeros((1, 7))

    operator.execute_trajectory(
        trajectory,
        trajectory,
        np.zeros(1),
        np.zeros(1),
        dt=0.001,
    )
    controller = operator._motion_controller
    bad_left = trajectory.copy()
    bad_left[0, 3] = 1.0

    with pytest.raises(ValueError, match='delta exceeds safety limit'):
        operator.execute_trajectory(
            bad_left,
            trajectory,
            np.zeros(1),
            np.zeros(1),
            dt=0.001,
        )

    assert operator._motion_controller is controller
    assert transport.disconnected is False
    assert len(controller.commands) == 1


def test_active_servoj_ignores_expected_stale_bridge_feedback():
    transport = FakeTransport()
    operator = make_operator([transport])
    trajectory = np.zeros((1, 7))

    operator.execute_trajectory(
        trajectory,
        trajectory,
        np.zeros(1),
        np.zeros(1),
        dt=0.001,
    )
    controller = operator._motion_controller
    operator._latest_bridge_state[11] = 0.55

    operator.execute_trajectory(
        trajectory,
        trajectory,
        np.zeros(1),
        np.zeros(1),
        dt=0.001,
    )

    assert operator._motion_controller is controller
    assert transport.disconnected is False
    assert len(controller.commands) == 2


def test_drained_servoj_trajectory_blends_six_recovery_frames():
    transport = FakeTransport()
    operator = make_operator([transport])

    operator.execute_trajectory(
        np.zeros((1, 7)),
        np.zeros((1, 7)),
        np.zeros(1),
        np.zeros(1),
        dt=0.001,
    )
    controller = operator._motion_controller
    assert operator._recovery_blend_pending is True

    left = np.full((6, 7), 0.12)
    right = np.full((6, 7), -0.12)
    operator.execute_trajectory(
        left,
        right,
        np.ones(6),
        np.full(6, 0.6),
        dt=0.001,
    )

    recovered_commands = [command[0] for command in controller.commands[1:]]
    assert len(recovered_commands) == 6
    for index, command in enumerate(recovered_commands, start=1):
        alpha = index / 6.0
        np.testing.assert_allclose(command[:7], 0.12 * alpha)
        np.testing.assert_allclose(command[7:14], -0.12 * alpha)
        np.testing.assert_allclose(command[14:], 0.0)
    assert controller.gripper_calls[1] == pytest.approx((100.0 / 6.0, 10.0))
    assert controller.gripper_calls[-1] == pytest.approx((100.0, 60.0))


def test_active_servoj_preemption_does_not_trigger_recovery_blend():
    transport = FakeTransport()
    operator = make_operator([transport])
    long_trajectory = np.zeros((20, 7))

    operator.execute_trajectory(
        long_trajectory,
        long_trajectory,
        np.zeros(20),
        np.zeros(20),
        dt=0.01,
        async_exec=True,
    )
    deadline = time.monotonic() + 1.0
    while (operator._motion_controller is None
           and time.monotonic() < deadline):
        time.sleep(0.001)
    controller = operator._motion_controller
    assert controller is not None
    while not controller.commands and time.monotonic() < deadline:
        time.sleep(0.001)
    assert controller.commands

    replacement = np.full((1, 7), 0.12)
    operator.execute_trajectory(
        replacement,
        -replacement,
        np.ones(1),
        np.full(1, 0.6),
        dt=0.001,
        async_exec=True,
    )
    operator.wait_for_trajectory()

    np.testing.assert_allclose(controller.commands[-1][0][:7], 0.12)
    np.testing.assert_allclose(controller.commands[-1][0][7:14], -0.12)
    assert controller.gripper_calls[-1] == pytest.approx((100.0, 60.0))


def test_active_chunk_replacement_smoothstep_blends_arm_boundary():
    transport = FakeTransport()
    operator = make_operator([transport])
    operator.chunk_boundary_blend_enabled = True
    operator.chunk_boundary_blend_frames = 4

    old_steps = np.arange(20, dtype=np.float64)[:, None] * 0.01
    old_left = np.repeat(old_steps, 7, axis=1)
    old_right = -old_left
    operator.execute_trajectory(
        old_left,
        old_right,
        np.linspace(0.0, 0.95, 20),
        np.linspace(0.0, 0.95, 20),
        dt=1.0,
        async_exec=True,
    )
    deadline = time.monotonic() + 1.0
    while operator._motion_controller is None and time.monotonic() < deadline:
        time.sleep(0.001)
    controller = operator._motion_controller
    assert controller is not None
    while len(controller.commands) < 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(controller.commands) == 1

    new_left = np.full((4, 7), 0.12)
    new_right = -new_left
    operator.execute_trajectory(
        new_left,
        new_right,
        np.ones(4),
        np.full(4, 0.6),
        dt=0.001,
    )

    recovered_commands = [command[0] for command in controller.commands[1:]]
    assert len(recovered_commands) == 4
    for offset, command in enumerate(recovered_commands):
        u = (offset + 1) / 5.0
        alpha = u * u * (3.0 - 2.0 * u)
        old_value = (offset + 1) * 0.01
        expected_left = (1.0 - alpha) * old_value + alpha * 0.12
        np.testing.assert_allclose(command[:7], expected_left)
        np.testing.assert_allclose(command[7:14], -expected_left)
        np.testing.assert_allclose(command[14:], 0.0)
    # The configured arm-only scope must not delay gripper commands.
    assert controller.gripper_calls[1] == pytest.approx((100.0, 60.0))
    assert controller.gripper_calls[-1] == pytest.approx((100.0, 60.0))


def test_wait_for_trajectory_does_not_cancel_accepted_feeder():
    operator = make_operator([])
    feeder_finished = threading.Event()

    def feed():
        time.sleep(0.02)
        feeder_finished.set()

    operator._traj_thread = threading.Thread(target=feed)
    operator._traj_thread.start()

    operator.wait_for_trajectory()

    assert feeder_finished.is_set()
    assert operator._traj_thread is None
    assert not operator._traj_stop_event.is_set()


def test_servoj_rejects_out_of_range_gripper():
    operator = make_operator([])

    with pytest.raises(ValueError, match=r'within \[0, 1\]'):
        operator._prepare_servoj_trajectory(
            np.zeros((1, 7)),
            np.zeros((1, 7)),
            np.array([1.1]),
            np.array([0.0]),
            None,
            np.zeros(18),
        )
