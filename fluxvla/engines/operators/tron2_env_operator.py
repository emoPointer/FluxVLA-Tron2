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
"""TRON2 WebSocket observation and control operator.

The public :mod:`tron2_env` Bridge provider supplies synchronized camera,
joint, and gripper observations over WebSocket.  A separate robot WebSocket
transport performs MoveJ initialization and measured-state-seeded ServoJ
policy execution.  This operator does not initialize ROS or subscribe to ROS
topics.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from fluxvla.engines.utils.root import OPERATORS


@OPERATORS.register_module()
class Tron2EnvOperator:
    """Use ``tron2_env`` Bridge observations and robot control WebSockets.

    The control modes are deliberately exclusive.  Entering the MoveJ path
    disconnects any active MotionController so its ServoJ publish thread
    cannot compete with initialization.  The next policy trajectory creates a
    fresh controller, which seeds interpolation from measured robot state
    before publishing its first ServoJ setpoint.

    Args:
        bridge_host: TRON2 Bridge WebSocket origin, for example
            ``wss://10.192.1.4``.
        bridge_ws_path: Bridge subscription endpoint.
        bridge_image_topics: Mapping with ``camera_left``, ``camera_right``,
            and ``camera_top`` topics. The upstream defaults are used when
            omitted.
        bridge_joint_topics: Mapping with ``joint_states`` and ``gripper``
            topics. The upstream defaults are used when omitted.
        bridge_observation_timeout: Seconds to wait for each aligned Bridge
            observation.
        bridge_startup_timeout: Seconds to wait for the first complete Bridge
            observation during construction.
        servoj_publish_rate: Background ServoJ publication rate in Hz.
        state_polling_rate: Robot-state polling rate used by ``tron2_env``.
        connection_timeout: Seconds to wait for the WebSocket connection.
        state_timeout: Seconds to wait for the measured state used to seed and
            validate a policy trajectory.
        movej_duration: Controller-side duration for each MoveJ prepare pose.
        movej_tolerance: Maximum joint error accepted after MoveJ.
        movej_timeout: Maximum time to wait for each MoveJ target.
        max_servoj_step_rad: Maximum absolute change of any ServoJ joint from
            measured state to the first waypoint or between adjacent policy
            waypoints. Set to ``None`` only after an external safety layer has
            been verified.
        max_state_source_mismatch_rad: Maximum allowed joint difference
            between the latest Bridge observation and robot-control WebSocket
            state before ServoJ may start.
        lock_head: Reject policy head trajectories and hold the measured head
            position captured before the first valid ServoJ trajectory.
        max_head_hold_error_rad: If the head has moved farther than this from
            its locked position, reject instead of commanding it back.
        servoj_joint_lower_limits: Optional 16-dim lower bounds ordered as
            left arm, right arm, head. Must be supplied with upper bounds.
        servoj_joint_upper_limits: Optional 16-dim upper bounds ordered as
            left arm, right arm, head. Must be supplied with lower bounds.
    """

    _ARM_DIM = 7
    _HEAD_DIM = 2
    _SERVOJ_DIM = 16
    _STATE_DIM = 18
    _BRIDGE_IMAGE_KEYS = frozenset(
        {'camera_left', 'camera_right', 'camera_top'})
    _BRIDGE_JOINT_KEYS = frozenset({'joint_states', 'gripper'})
    _POLICY_IMAGE_KEYS = (
        'cam_high',
        'cam_left_wrist',
        'cam_right_wrist',
    )

    def __init__(self,
                 bridge_host: str = 'wss://10.192.1.4',
                 bridge_ws_path: str = '/bridge/ws',
                 bridge_image_topics: Optional[Dict[str, str]] = None,
                 bridge_joint_topics: Optional[Dict[str, str]] = None,
                 bridge_image_max_fps: int = 0,
                 bridge_align_max_delay_ms: int = 200,
                 bridge_verify_tls: bool = False,
                 bridge_observation_timeout: float = 2.0,
                 bridge_startup_timeout: float = 10.0,
                 servoj_publish_rate: float = 300.0,
                 state_polling_rate: float = 200.0,
                 connection_timeout: float = 5.0,
                 state_timeout: float = 2.0,
                 movej_duration: float = 2.0,
                 movej_tolerance: float = 0.05,
                 movej_timeout: float = 10.0,
                 max_servoj_step_rad: Optional[float] = 0.2,
                 max_state_source_mismatch_rad: Optional[float] = None,
                 lock_head: bool = True,
                 max_head_hold_error_rad: float = 0.05,
                 servoj_joint_lower_limits: Optional[List[float]] = None,
                 servoj_joint_upper_limits: Optional[List[float]] = None,
                 robot_ip: str = '10.192.1.2',
                 ws_port: int = 5000,
                 ws_accid: Optional[str] = None,
                 trajectory_exec_mode: str = 'servoj',
                 connect_websocket: bool = True,
                 connect_observation: bool = True):
        if trajectory_exec_mode != 'servoj':
            raise ValueError(
                'Tron2EnvOperator only supports ServoJ policy execution; '
                f'got trajectory_exec_mode={trajectory_exec_mode!r}')
        if ws_accid is not None:
            raise ValueError(
                'Tron2EnvOperator uses tron2_env account auto-detection; '
                'ws_accid must be None.')

        self._validate_positive('servoj_publish_rate', servoj_publish_rate)
        self._validate_positive('state_polling_rate', state_polling_rate)
        self._validate_positive('connection_timeout', connection_timeout)
        self._validate_positive('state_timeout', state_timeout)
        self._validate_positive('bridge_observation_timeout',
                                bridge_observation_timeout)
        self._validate_positive('bridge_startup_timeout',
                                bridge_startup_timeout)
        self._validate_positive('movej_duration', movej_duration)
        self._validate_positive('movej_tolerance', movej_tolerance)
        self._validate_positive('movej_timeout', movej_timeout)
        if max_servoj_step_rad is not None:
            self._validate_positive('max_servoj_step_rad', max_servoj_step_rad)
        if max_state_source_mismatch_rad is not None:
            self._validate_positive('max_state_source_mismatch_rad',
                                    max_state_source_mismatch_rad)
        self._validate_positive('max_head_hold_error_rad',
                                max_head_hold_error_rad)

        parsed_bridge_host = urlparse(bridge_host)
        if (parsed_bridge_host.scheme not in {'ws', 'wss'}
                or not parsed_bridge_host.netloc
                or 'BRIDGE_HOST' in bridge_host):
            raise ValueError('bridge_host must be a concrete ws:// or wss:// '
                             f'origin; got {bridge_host!r}.')
        if not bridge_ws_path.startswith('/'):
            raise ValueError('bridge_ws_path must start with "/"; got '
                             f'{bridge_ws_path!r}.')
        if bridge_image_max_fps < 0:
            raise ValueError('bridge_image_max_fps must be >= 0; got '
                             f'{bridge_image_max_fps!r}.')
        if bridge_align_max_delay_ms <= 0:
            raise ValueError('bridge_align_max_delay_ms must be > 0; got '
                             f'{bridge_align_max_delay_ms!r}.')
        self._validate_topic_mapping(bridge_image_topics,
                                     self._BRIDGE_IMAGE_KEYS,
                                     'bridge_image_topics')
        self._validate_topic_mapping(bridge_joint_topics,
                                     self._BRIDGE_JOINT_KEYS,
                                     'bridge_joint_topics')

        self.bridge_host = bridge_host.rstrip('/')
        self.bridge_ws_path = bridge_ws_path
        self.bridge_image_topics = (None if bridge_image_topics is None else
                                    dict(bridge_image_topics))
        self.bridge_joint_topics = (None if bridge_joint_topics is None else
                                    dict(bridge_joint_topics))
        self.bridge_image_max_fps = int(bridge_image_max_fps)
        self.bridge_align_max_delay_ms = int(bridge_align_max_delay_ms)
        self.bridge_verify_tls = bool(bridge_verify_tls)
        self.bridge_observation_timeout = float(bridge_observation_timeout)
        self.bridge_startup_timeout = float(bridge_startup_timeout)
        self.servoj_publish_rate = float(servoj_publish_rate)
        self.state_polling_rate = float(state_polling_rate)
        self.connection_timeout = float(connection_timeout)
        self.state_timeout = float(state_timeout)
        self.movej_duration = float(movej_duration)
        self.movej_tolerance = float(movej_tolerance)
        self.movej_timeout = float(movej_timeout)
        self.max_servoj_step_rad = (None if max_servoj_step_rad is None else
                                    float(max_servoj_step_rad))
        self.max_state_source_mismatch_rad = (
            None if max_state_source_mismatch_rad is None else
            float(max_state_source_mismatch_rad))
        self.lock_head = bool(lock_head)
        self.max_head_hold_error_rad = float(max_head_hold_error_rad)
        (self.servoj_joint_lower_limits,
         self.servoj_joint_upper_limits) = self._validate_joint_limits(
             servoj_joint_lower_limits, servoj_joint_upper_limits)

        self._control_lock = threading.RLock()
        self.robot_ip = robot_ip
        self.ws_port = int(ws_port)
        self.connect_websocket = bool(connect_websocket)
        self.connect_observation = bool(connect_observation)
        self._transport = None
        self._motion_controller = None
        self._bridge_provider = None
        self._pending_bridge_observation = None
        self._latest_bridge_state = None
        self._head_hold_position = None
        self._closed = False
        self._trajectory_error = None
        self._traj_thread = None
        self._traj_stop_event = threading.Event()

        if self.connect_observation:
            self._start_bridge_observation()
        else:
            print('Tron2EnvOperator Bridge observation disabled '
                  '(connect_observation=False)')

        try:
            if self.connect_websocket:
                with self._control_lock:
                    self._ensure_transport_locked()
            else:
                print('Tron2EnvOperator control transport disabled '
                      '(connect_websocket=False)')
        except Exception:
            self._stop_bridge_observation()
            raise

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and > 0, got {value!r}')

    @staticmethod
    def _validate_topic_mapping(topics: Optional[Dict[str, str]],
                                expected_keys, name: str) -> None:
        if topics is None:
            return
        if set(topics) != set(expected_keys):
            raise ValueError(f'{name} must contain exactly '
                             f'{sorted(expected_keys)}; got '
                             f'{sorted(topics)}.')
        invalid = [
            key for key, topic in topics.items()
            if not isinstance(topic, str) or not topic.startswith('/')
        ]
        if invalid:
            raise ValueError(f'{name} contains invalid Bridge topic paths for '
                             f'keys {invalid}.')

    @classmethod
    def _validate_joint_limits(
        cls,
        lower: Optional[List[float]],
        upper: Optional[List[float]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if (lower is None) != (upper is None):
            raise ValueError('ServoJ lower and upper limits must be supplied '
                             'together.')
        if lower is None:
            return None, None

        lower_arr = np.asarray(lower, dtype=np.float64)
        upper_arr = np.asarray(upper, dtype=np.float64)
        expected = (cls._SERVOJ_DIM, )
        if lower_arr.shape != expected or upper_arr.shape != expected:
            raise ValueError('ServoJ joint limits must both have shape '
                             f'{expected}; got {lower_arr.shape} and '
                             f'{upper_arr.shape}.')
        if not np.all(np.isfinite(lower_arr)) or not np.all(
                np.isfinite(upper_arr)):
            raise ValueError('ServoJ joint limits must be finite.')
        if np.any(lower_arr >= upper_arr):
            raise ValueError('Every ServoJ lower limit must be less than its '
                             'upper limit.')
        return lower_arr, upper_arr

    @staticmethod
    def _runtime_types():
        try:
            from tron2_env import (MotionController, Tron2Config,
                                   WebsocketTransport)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'Tron2EnvOperator requires the public tron2_env runtime. '
                'Install the project requirements in the FluxVLA '
                'environment.') from exc
        return MotionController, Tron2Config, WebsocketTransport

    @staticmethod
    def _observation_types():
        try:
            import websockets  # noqa: F401
            from tron2_env import BridgeConfig, BridgeObservationProvider
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'Tron2EnvOperator Bridge observations require the '
                'tron2_env bridge extra (websockets>=12).') from exc
        return BridgeConfig, BridgeObservationProvider

    def _create_bridge_provider(self):
        config_type, provider_type = self._observation_types()
        config_kwargs: Dict[str, Any] = {
            'host': self.bridge_host,
            'ws_path': self.bridge_ws_path,
            'image_max_fps': self.bridge_image_max_fps,
            'align_max_delay_ms': self.bridge_align_max_delay_ms,
            'verify_tls': self.bridge_verify_tls,
            'save_debug_images': False,
        }
        if self.bridge_image_topics is not None:
            config_kwargs['image_topics'] = self.bridge_image_topics
        if self.bridge_joint_topics is not None:
            config_kwargs['joint_topics'] = self.bridge_joint_topics
        return provider_type(config_type(**config_kwargs))

    def _start_bridge_observation(self) -> None:
        provider = self._create_bridge_provider()
        provider.start()
        try:
            observation = provider.get_obs(timeout=self.bridge_startup_timeout)
            self._pending_bridge_observation = (
                self._validate_bridge_observation(observation))
        except Exception as exc:
            provider.stop()
            raise RuntimeError(
                'TRON2 Bridge did not provide one complete camera/state '
                f'observation within {self.bridge_startup_timeout:.1f}s '
                f'from {self.bridge_host}{self.bridge_ws_path}.') from exc
        self._bridge_provider = provider

    def _stop_bridge_observation(self) -> None:
        provider = self._bridge_provider
        self._bridge_provider = None
        self._pending_bridge_observation = None
        self._latest_bridge_state = None
        if provider is not None:
            provider.stop()

    @classmethod
    def _validate_bridge_observation(cls, observation) -> Dict[str, Any]:
        if not isinstance(observation, dict):
            raise RuntimeError('tron2_env Bridge observation must be a dict; '
                               f'got {type(observation).__name__}.')
        images = observation.get('images')
        if not isinstance(images, dict):
            raise RuntimeError('tron2_env Bridge observation has no image '
                               'mapping.')
        missing = [key for key in cls._POLICY_IMAGE_KEYS if key not in images]
        if missing:
            raise RuntimeError('tron2_env Bridge observation is missing '
                               f'policy images: {missing}.')
        validated_images = {}
        for key in cls._POLICY_IMAGE_KEYS:
            image = np.asarray(images[key])
            if (image.ndim != 3 or image.shape[2] != 3 or image.size == 0
                    or image.dtype != np.uint8):
                raise RuntimeError(
                    f'Bridge image {key!r} must be a non-empty HxWx3 uint8 '
                    f'array; got shape={image.shape}, dtype={image.dtype}.')
            validated_images[key] = image

        state = np.asarray(observation.get('state'), dtype=np.float32)
        if state.shape != (cls._STATE_DIM, ) or not np.all(np.isfinite(state)):
            raise RuntimeError('tron2_env Bridge must provide one finite '
                               f'{cls._STATE_DIM}-dim state; got '
                               f'{state.shape}.')
        return {
            'images': validated_images,
            'state': state,
            'metadata': dict(observation.get('metadata', {})),
        }

    def get_observation(self) -> Dict[str, Any]:
        """Return the freshest aligned Bridge camera/joint/gripper sample."""
        if self._closed:
            raise RuntimeError('Tron2EnvOperator is closed.')
        if self._bridge_provider is None:
            raise RuntimeError('TRON2 Bridge observation is not connected.')
        if self._pending_bridge_observation is not None:
            observation = self._pending_bridge_observation
            self._pending_bridge_observation = None
        else:
            observation = self._bridge_provider.get_obs(
                timeout=self.bridge_observation_timeout)
            observation = self._validate_bridge_observation(observation)
        with self._control_lock:
            self._latest_bridge_state = observation['state'].astype(
                np.float64, copy=True)
        return observation

    def _create_transport(self):
        _, config_type, transport_type = self._runtime_types()
        config = config_type(
            robot_ip=self.robot_ip,
            port=self.ws_port,
            init_joints=None,
            init_head=None,
            polling_rate=self.state_polling_rate,
            connection_timeout=self.connection_timeout,
        )
        transport = transport_type(config)

        deadline = time.monotonic() + self.connection_timeout
        while not transport.is_connected() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not transport.is_connected():
            transport.disconnect()
            raise ConnectionError('tron2_env WebSocket connection timed out: '
                                  f'ws://{self.robot_ip}:{self.ws_port}')
        return transport

    def _ensure_transport_locked(self):
        if self._closed:
            raise RuntimeError('Tron2EnvOperator is closed.')
        if self._transport is None:
            self._transport = self._create_transport()
        if not self._transport.is_connected():
            self._disconnect_control_locked()
            self._transport = self._create_transport()
        return self._transport

    def _start_motion_controller_locked(self, dt: float):
        """Start ServoJ only after the caller has validated a policy chunk."""
        if self._motion_controller is not None:
            return self._motion_controller

        motion_controller_type, _, _ = self._runtime_types()
        transport = self._ensure_transport_locked()
        controller = motion_controller_type(
            transport=transport,
            publish_rate=self.servoj_publish_rate,
            eta_default=dt,
        )
        try:
            controller.start()
        except Exception:
            transport.disconnect()
            self._transport = None
            raise
        self._motion_controller = controller
        return controller

    def _disconnect_control_locked(self) -> None:
        controller = self._motion_controller
        transport = self._transport
        self._motion_controller = None
        self._transport = None

        if controller is not None:
            controller.disconnect()
        elif transport is not None:
            transport.disconnect()

    @staticmethod
    def _validate_arm_target(left, right) -> np.ndarray:
        left_arr = np.asarray(left, dtype=np.float64)
        right_arr = np.asarray(right, dtype=np.float64)
        if left_arr.shape != (7, ) or right_arr.shape != (7, ):
            raise ValueError('MoveJ targets must each contain 7 joints; got '
                             f'{left_arr.shape} and {right_arr.shape}.')
        target = np.concatenate([left_arr, right_arr])
        if not np.all(np.isfinite(target)):
            raise ValueError('MoveJ target contains non-finite values.')
        return target

    @staticmethod
    def _validate_gripper_value(value, name: str) -> Optional[float]:
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f'{name} must be finite and within [0, 1], '
                             f'got {value!r}.')
        return value

    def move_to_targets(self,
                        left,
                        right,
                        control_rate: int = 30,
                        left_gripper: float = None,
                        right_gripper: float = None,
                        head: List[float] = None):
        """Execute one prepare-pose target with controller-side MoveJ."""
        del control_rate  # Kept for compatibility with Tron2InferenceRunner.
        target = self._validate_arm_target(left, right)
        left_gripper = self._validate_gripper_value(left_gripper,
                                                    'left_gripper')
        right_gripper = self._validate_gripper_value(right_gripper,
                                                     'right_gripper')
        if head is not None:
            if self.lock_head:
                raise ValueError(
                    'Head movement is locked for this deployment; '
                    'MoveJ prepare poses must not include head.')
            head = np.asarray(head, dtype=np.float64)
            if head.shape != (self._HEAD_DIM, ) or not np.all(
                    np.isfinite(head)):
                raise ValueError('MoveJ head target must contain 2 finite '
                                 f'values; got shape {head.shape}.')

        self.stop_trajectory()
        with self._control_lock:
            # A ServoJ publisher must never run concurrently with MoveJ.
            if self._motion_controller is not None:
                self._disconnect_control_locked()
            transport = self._ensure_transport_locked()
            try:
                if left_gripper is not None or right_gripper is not None:
                    state = transport.get_joint_state(
                        timeout=self.state_timeout)['states']
                    if len(state) != self._STATE_DIM:
                        raise RuntimeError('tron2_env returned an invalid '
                                           f'{len(state)}-dim state.')
                    current_left = float(state[7]) * 100.0
                    current_right = float(state[15]) * 100.0
                    transport.set_gripper(
                        current_left
                        if left_gripper is None else left_gripper * 100.0,
                        current_right
                        if right_gripper is None else right_gripper * 100.0,
                    )

                transport.movej(target, move_time=self.movej_duration)
                reached = transport.wait_until_reached(
                    target,
                    tolerance=self.movej_tolerance,
                    timeout=self.movej_timeout,
                )
                if not reached:
                    raise TimeoutError(
                        'MoveJ prepare target was not reached within '
                        f'{self.movej_timeout:.1f}s.')

                if head is not None:
                    transport.move_head(head, move_time=self.movej_duration)
                    time.sleep(self.movej_duration)
            except Exception:
                self._disconnect_control_locked()
                raise

    @classmethod
    def _as_trajectory(cls, values, name: str, width: int) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f'{name} must have shape (N, {width}); got '
                             f'{array.shape}.')
        if not np.all(np.isfinite(array)):
            raise ValueError(f'{name} contains non-finite values.')
        return array

    def _prepare_servoj_trajectory(self,
                                   left_arm_trajectory,
                                   right_arm_trajectory,
                                   left_gripper_trajectory,
                                   right_gripper_trajectory,
                                   head_trajectory,
                                   current_state,
                                   check_state_source: bool = True):
        left = self._as_trajectory(left_arm_trajectory, 'left arm trajectory',
                                   self._ARM_DIM)
        right = self._as_trajectory(right_arm_trajectory,
                                    'right arm trajectory', self._ARM_DIM)
        if left.shape[0] != right.shape[0]:
            raise ValueError('Left and right trajectories must have the same '
                             'number of waypoints.')
        n = left.shape[0]
        if n == 0:
            return np.empty((0, self._SERVOJ_DIM)), np.empty((0, )), np.empty(
                (0, ))

        left_gripper = np.asarray(
            left_gripper_trajectory, dtype=np.float64).reshape(-1)
        right_gripper = np.asarray(
            right_gripper_trajectory, dtype=np.float64).reshape(-1)
        if left_gripper.shape != (n, ) or right_gripper.shape != (n, ):
            raise ValueError('Gripper trajectories must each contain one '
                             f'value per waypoint ({n}).')
        if (not np.all(np.isfinite(left_gripper))
                or not np.all(np.isfinite(right_gripper))):
            raise ValueError('Gripper trajectory contains non-finite values.')
        if (np.any((left_gripper < 0.0) | (left_gripper > 1.0))
                or np.any((right_gripper < 0.0)
                          | (right_gripper > 1.0))):
            raise ValueError('Gripper trajectory values must be within '
                             '[0, 1].')

        current = np.asarray(current_state, dtype=np.float64)
        if current.shape != (self._STATE_DIM, ) or not np.all(
                np.isfinite(current)):
            raise ValueError('tron2_env must provide one finite 18-dim '
                             f'state; got {current.shape}.')
        current_servoj = np.concatenate(
            [current[:7], current[8:15], current[16:18]])

        if (check_state_source
                and self.max_state_source_mismatch_rad is not None):
            if self._latest_bridge_state is None:
                raise RuntimeError('No Bridge state is available for ServoJ '
                                   'cross-validation. Acquire an observation '
                                   'before executing policy actions.')
            bridge_state = np.asarray(
                self._latest_bridge_state, dtype=np.float64)
            if bridge_state.shape != (self._STATE_DIM, ) or not np.all(
                    np.isfinite(bridge_state)):
                raise RuntimeError('Latest Bridge state is invalid for ServoJ '
                                   'cross-validation: '
                                   f'{bridge_state.shape}.')
            bridge_servoj = np.concatenate(
                [bridge_state[:7], bridge_state[8:15], bridge_state[16:18]])
            source_deltas = np.abs(current_servoj - bridge_servoj)
            source_joint = int(np.argmax(source_deltas))
            source_delta = float(source_deltas[source_joint])
            if source_delta > self.max_state_source_mismatch_rad:
                raise ValueError(
                    'Bridge/control state mismatch blocks ServoJ: '
                    f'{source_delta:.6f} rad at joint {source_joint} > '
                    f'{self.max_state_source_mismatch_rad:.6f} rad '
                    f'(bridge={bridge_servoj[source_joint]:.6f}, '
                    f'control={current_servoj[source_joint]:.6f}).')

        if self.lock_head:
            if head_trajectory is not None:
                raise ValueError(
                    'Head movement is locked for this deployment; '
                    'policy head trajectories are not allowed.')
            if self._head_hold_position is None:
                # Lock exactly to the robot-control feedback used to seed the
                # first ServoJ publisher. Bridge feedback is independently
                # checked above, but is not used as a corrective head target.
                self._head_hold_position = current_servoj[-self.
                                                          _HEAD_DIM:].copy()
            head_error = np.abs(current_servoj[-self._HEAD_DIM:] -
                                self._head_hold_position)
            head_joint = int(np.argmax(head_error))
            if float(head_error[head_joint]) > self.max_head_hold_error_rad:
                raise ValueError(
                    'Measured head moved away from its locked position; '
                    'ServoJ is blocked instead of commanding the head back: '
                    f'{float(head_error[head_joint]):.6f} rad at head joint '
                    f'{head_joint} > {self.max_head_hold_error_rad:.6f} rad.')
            head = np.repeat(self._head_hold_position[None], n, axis=0)
        elif head_trajectory is None:
            head = np.repeat(current_servoj[-self._HEAD_DIM:][None], n, axis=0)
        else:
            head = self._as_trajectory(head_trajectory, 'head trajectory',
                                       self._HEAD_DIM)
            if head.shape[0] != n:
                raise ValueError('Head trajectory must contain one value per '
                                 f'arm waypoint ({n}).')

        waypoints = np.concatenate([left, right, head], axis=1)
        if self.servoj_joint_lower_limits is not None:
            below = waypoints < self.servoj_joint_lower_limits
            above = waypoints > self.servoj_joint_upper_limits
            if np.any(below | above):
                waypoint, joint = np.argwhere(below | above)[0]
                raise ValueError('ServoJ joint limit violation at waypoint '
                                 f'{int(waypoint)}, joint {int(joint)}: '
                                 f'{waypoints[waypoint, joint]:.6f}.')

        if self.max_servoj_step_rad is not None:
            points = np.concatenate([current_servoj[None], waypoints], axis=0)
            deltas = np.abs(np.diff(points, axis=0))
            max_index = np.unravel_index(np.argmax(deltas), deltas.shape)
            max_delta = float(deltas[max_index])
            if max_delta > self.max_servoj_step_rad:
                current_value = float(points[max_index[0], max_index[1]])
                target_value = float(points[max_index[0] + 1, max_index[1]])
                raise ValueError(
                    'ServoJ waypoint delta exceeds safety limit: '
                    f'{max_delta:.6f} rad at transition {max_index[0]}, '
                    f'joint {max_index[1]} > '
                    f'{self.max_servoj_step_rad:.6f} rad '
                    f'(current={current_value:.6f}, '
                    f'target={target_value:.6f}).')

        return waypoints, left_gripper, right_gripper

    def execute_trajectory(self,
                           left_arm_trajectory,
                           right_arm_trajectory,
                           left_gripper_trajectory,
                           right_gripper_trajectory,
                           head_trajectory=None,
                           dt: float = 0.1,
                           async_exec: bool = False):
        """Feed policy waypoints to the ``tron2_env`` ServoJ controller."""
        self._validate_positive('dt', dt)
        self.stop_trajectory()
        self._traj_stop_event = threading.Event()
        self._trajectory_error = None
        args = (left_arm_trajectory, right_arm_trajectory,
                left_gripper_trajectory, right_gripper_trajectory,
                head_trajectory, float(dt), self._traj_stop_event)

        if async_exec:
            self._traj_thread = threading.Thread(
                target=self._run_servoj_trajectory,
                args=args,
                daemon=True,
                name='Tron2EnvOperator-policy-feeder')
            self._traj_thread.start()
        else:
            self._traj_thread = None
            self._run_servoj_trajectory(*args)

    def execute_waypoint(self,
                         left_arm,
                         right_arm,
                         left_gripper,
                         right_gripper,
                         head=None,
                         dt: float = 0.1):
        """Validate and submit one policy waypoint without blocking for ``dt``.

        The overlap inference runner owns the 30 Hz action clock, while the
        persistent ``tron2_env`` MotionController interpolates and publishes
        ServoJ commands at its configured high rate. Every waypoint still
        passes the same live-feedback, joint-limit, gripper, and head-lock
        checks as a complete trajectory.
        """
        self._validate_positive('dt', dt)
        left = np.asarray(left_arm, dtype=np.float64).reshape(1, -1)
        right = np.asarray(right_arm, dtype=np.float64).reshape(1, -1)
        left_gripper = np.asarray([left_gripper], dtype=np.float64)
        right_gripper = np.asarray([right_gripper], dtype=np.float64)
        head_trajectory = None
        if head is not None:
            head_trajectory = np.asarray(head, dtype=np.float64).reshape(1, -1)

        self.stop_trajectory()
        self._traj_stop_event = threading.Event()
        self._trajectory_error = None
        self._traj_thread = None
        self._run_servoj_trajectory(
            left,
            right,
            left_gripper,
            right_gripper,
            head_trajectory,
            float(dt),
            self._traj_stop_event,
            pace=False,
        )

    def _run_servoj_trajectory(self,
                               left_arm_trajectory,
                               right_arm_trajectory,
                               left_gripper_trajectory,
                               right_gripper_trajectory,
                               head_trajectory,
                               dt,
                               stop_event,
                               pace=True):
        try:
            with self._control_lock:
                controller = self._motion_controller
                starting_controller = controller is None
                if starting_controller:
                    # Validate against measured state before MotionController
                    # starts its 300 Hz publisher. A rejected first chunk must
                    # result in zero ServoJ command frames.
                    transport = self._ensure_transport_locked()
                    state = transport.get_joint_state(
                        timeout=self.state_timeout)['states']
                    self._prepare_servoj_trajectory(
                        left_arm_trajectory,
                        right_arm_trajectory,
                        left_gripper_trajectory,
                        right_gripper_trajectory,
                        head_trajectory,
                        state,
                        check_state_source=True)
                    controller = self._start_motion_controller_locked(dt)

                # On first startup, cross-check Bridge and control feedback
                # again against the exact post-start state. For later chunks,
                # Bridge feedback predates inference and naturally lags the
                # moving control state, so only the live control-state delta,
                # joint-limit, gripper, and head-lock guards apply.
                state = controller.get_joint_states(
                    timeout=self.state_timeout)['states']
                waypoints, left_gripper, right_gripper = (
                    self._prepare_servoj_trajectory(
                        left_arm_trajectory,
                        right_arm_trajectory,
                        left_gripper_trajectory,
                        right_gripper_trajectory,
                        head_trajectory,
                        state,
                        check_state_source=(starting_controller)))

            start = time.perf_counter()
            for index, waypoint in enumerate(waypoints):
                if stop_event.is_set():
                    return
                if not controller.is_connected():
                    raise ConnectionError(
                        'tron2_env WebSocket disconnected during ServoJ '
                        'trajectory execution.')

                controller.set_gripper(
                    float(left_gripper[index] * 100.0),
                    float(right_gripper[index] * 100.0),
                )
                controller.command_joints(waypoint, eta=dt)

                if pace:
                    deadline = start + (index + 1) * dt
                    remaining = deadline - time.perf_counter()
                    if remaining > 0:
                        stop_event.wait(remaining)
        except ValueError as exc:
            # A rejected waypoint is recoverable. Do not disconnect an active
            # MotionController: issuing no replacement command keeps its last
            # accepted target unchanged, and the runner can continue with the
            # next policy waypoint.
            self._trajectory_error = exc
            stop_event.set()
            raise
        except Exception as exc:
            self._trajectory_error = exc
            stop_event.set()
            with self._control_lock:
                self._disconnect_control_locked()
            raise

    def stop_trajectory(self):
        """Stop policy waypoint feeding while holding the latest target."""
        self._traj_stop_event.set()
        thread = self._traj_thread
        if (thread is not None and thread.is_alive()
                and thread is not threading.current_thread()):
            thread.join(timeout=max(2.0, self.state_timeout))
            if thread.is_alive():
                raise RuntimeError('ServoJ trajectory feeder did not stop.')
        self._traj_thread = None

    def close(self):
        """Stop observation/control threads and disconnect both WebSockets."""
        if self._closed:
            return
        try:
            self.stop_trajectory()
        finally:
            try:
                self._stop_bridge_observation()
            finally:
                with self._control_lock:
                    self._disconnect_control_locked()
                    self._closed = True
