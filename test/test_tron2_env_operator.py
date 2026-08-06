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

    def set_gripper(self, left, right):
        self.gripper_calls.append((left, right))

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

    def is_connected(self):
        return self.transport.is_connected()

    def set_gripper(self, left, right):
        self.gripper_calls.append((left, right))

    def command_joints(self, target, eta=None):
        self.commands.append((np.asarray(target).copy(), eta))


def make_operator(transports):
    operator = Tron2EnvOperator.__new__(Tron2EnvOperator)
    operator.servoj_publish_rate = 300.0
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
    runner.camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    runner.observation_window = None
    runner.ros_operator = type(
        'FakeObservationOperator', (),
        {'get_observation': lambda self: make_bridge_observation()})()
    runner._apply_jpeg_compression = lambda image: image

    observation = runner.update_observation_window()

    np.testing.assert_array_equal(observation['qpos'], expected)
    assert observation['cam_high'].shape == (4, 6, 3)


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
