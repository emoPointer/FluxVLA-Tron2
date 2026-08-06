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

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from fluxvla.engines.utils.root import OPERATORS

logger = logging.getLogger(__name__)


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
        recovery_blend_frames: Number of policy frames used to blend from the
            held ServoJ target after the previous trajectory has drained. Zero
            disables recovery blending. Normal asynchronous preemption does
            not trigger this path.
        chunk_boundary_blend_enabled: Smooth normal asynchronous chunk
            replacements with a smoothstep-weighted blend of the old unissued
            trajectory and the new trajectory.
        chunk_boundary_blend_frames: Maximum number of replacement-boundary
            policy frames to blend.
        chunk_boundary_blend_scope: Components to blend: ``arm``, ``gripper``,
            or ``all``.
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
                 recovery_blend_frames: int = 6,
                 chunk_boundary_blend_enabled: bool = False,
                 chunk_boundary_blend_frames: int = 6,
                 chunk_boundary_blend_scope: str = 'arm',
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
        if (isinstance(recovery_blend_frames, bool)
                or not isinstance(recovery_blend_frames, int)
                or recovery_blend_frames < 0):
            raise ValueError('recovery_blend_frames must be a non-negative '
                             f'integer; got {recovery_blend_frames!r}.')
        if not isinstance(chunk_boundary_blend_enabled, bool):
            raise ValueError('chunk_boundary_blend_enabled must be a bool; '
                             f'got {chunk_boundary_blend_enabled!r}.')
        if (isinstance(chunk_boundary_blend_frames, bool)
                or not isinstance(chunk_boundary_blend_frames, int)
                or chunk_boundary_blend_frames < 0):
            raise ValueError(
                'chunk_boundary_blend_frames must be a non-negative integer; '
                f'got {chunk_boundary_blend_frames!r}.')
        if chunk_boundary_blend_scope not in {'arm', 'gripper', 'all'}:
            raise ValueError(
                "chunk_boundary_blend_scope must be 'arm', 'gripper', or "
                f"'all'; got {chunk_boundary_blend_scope!r}.")
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
        self.recovery_blend_frames = recovery_blend_frames
        self.chunk_boundary_blend_enabled = chunk_boundary_blend_enabled
        self.chunk_boundary_blend_frames = chunk_boundary_blend_frames
        self.chunk_boundary_blend_scope = chunk_boundary_blend_scope
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
        self._last_servoj_waypoint = None
        self._last_left_gripper = None
        self._last_right_gripper = None
        self._recovery_blend_pending = False
        self._active_servoj_waypoints = None
        self._active_left_gripper = None
        self._active_right_gripper = None
        self._active_next_index = 0
        self._issued_action_callback = None

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
        self._clear_recovery_state_locked()

        if controller is not None:
            controller.disconnect()
        elif transport is not None:
            transport.disconnect()

    def _clear_recovery_state_locked(self) -> None:
        """Forget ServoJ hold state after MoveJ or a transport restart."""
        self._last_servoj_waypoint = None
        self._last_left_gripper = None
        self._last_right_gripper = None
        self._recovery_blend_pending = False
        self._clear_active_trajectory_locked()

    def _clear_active_trajectory_locked(self) -> None:
        self._active_servoj_waypoints = None
        self._active_left_gripper = None
        self._active_right_gripper = None
        self._active_next_index = 0

    def set_issued_action_callback(self, callback) -> None:
        """Set a non-blocking observer for issued policy-rate waypoints."""
        if callback is not None and not callable(callback):
            raise ValueError('Issued-action callback must be callable or '
                             f'None; got {callback!r}.')
        with self._control_lock:
            self._issued_action_callback = callback

    def _notify_issued_action(self, *, waypoint, left_gripper, right_gripper,
                              trajectory_index, issued_at_unix,
                              issued_at_monotonic, action_metadata) -> None:
        with self._control_lock:
            callback = self._issued_action_callback
        if callback is None:
            return
        try:
            callback(
                waypoint=waypoint.copy(),
                left_gripper=float(left_gripper),
                right_gripper=float(right_gripper),
                trajectory_index=int(trajectory_index),
                issued_at_unix=float(issued_at_unix),
                issued_at_monotonic=float(issued_at_monotonic),
                action_metadata=dict(action_metadata or {}),
            )
        except Exception:
            logger.exception('Issued-action recorder callback failed; action '
                             'execution continues and recording is disabled.')
            with self._control_lock:
                if self._issued_action_callback is callback:
                    self._issued_action_callback = None

    def _snapshot_active_leftover_locked(self):
        if (self._active_servoj_waypoints is None
                or self._active_left_gripper is None
                or self._active_right_gripper is None):
            return None
        index = min(self._active_next_index,
                    len(self._active_servoj_waypoints))
        # A feeder that was stopped before issuing its first waypoint has no
        # executable history, so its unsent plan must not influence the next
        # chunk boundary.
        if index <= 0 or index >= len(self._active_servoj_waypoints):
            return None
        return (
            self._active_servoj_waypoints[index:].copy(),
            self._active_left_gripper[index:].copy(),
            self._active_right_gripper[index:].copy(),
        )

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
            # MoveJ establishes a new measured starting point. A future ServoJ
            # trajectory must seed from that state rather than an old hold
            # target captured before reset.
            self._clear_recovery_state_locked()
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
        current_servoj = self._state_to_servoj(current)

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

        self._validate_servoj_waypoint_deltas(current_servoj, waypoints)

        return waypoints, left_gripper, right_gripper

    @staticmethod
    def _state_to_servoj(state: np.ndarray) -> np.ndarray:
        return np.concatenate([state[:7], state[8:15], state[16:18]])

    def _validate_servoj_waypoint_deltas(self, current_servoj: np.ndarray,
                                         waypoints: np.ndarray) -> None:
        if self.max_servoj_step_rad is None or len(waypoints) == 0:
            return
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

    def _apply_recovery_blend(self, waypoints: np.ndarray,
                              left_gripper: np.ndarray,
                              right_gripper: np.ndarray):
        """Blend a recovered trajectory from the last held command."""
        if (not self._recovery_blend_pending or self.recovery_blend_frames == 0
                or len(waypoints) == 0 or self._last_servoj_waypoint is None
                or self._last_left_gripper is None
                or self._last_right_gripper is None):
            return waypoints, left_gripper, right_gripper, 0

        count = min(self.recovery_blend_frames, len(waypoints))
        alpha = (
            np.arange(1, count + 1, dtype=np.float64) /
            self.recovery_blend_frames)
        blended_waypoints = waypoints.copy()
        blended_left = left_gripper.copy()
        blended_right = right_gripper.copy()
        blended_waypoints[:count] = (
            (1.0 - alpha[:, None]) * self._last_servoj_waypoint[None] +
            alpha[:, None] * blended_waypoints[:count])
        blended_left[:count] = ((1.0 - alpha) * self._last_left_gripper +
                                alpha * blended_left[:count])
        blended_right[:count] = ((1.0 - alpha) * self._last_right_gripper +
                                 alpha * blended_right[:count])
        return blended_waypoints, blended_left, blended_right, count

    def _apply_chunk_boundary_blend(self, waypoints: np.ndarray,
                                    left_gripper: np.ndarray,
                                    right_gripper: np.ndarray, old_leftover):
        """Smooth an active chunk replacement with old unissued actions."""
        if (not self.chunk_boundary_blend_enabled
                or self.chunk_boundary_blend_frames == 0
                or old_leftover is None or len(waypoints) == 0):
            return waypoints, left_gripper, right_gripper, 0

        old_waypoints, old_left, old_right = old_leftover
        count = min(self.chunk_boundary_blend_frames, len(waypoints),
                    len(old_waypoints))
        if count == 0:
            return waypoints, left_gripper, right_gripper, 0

        u = np.arange(1, count + 1, dtype=np.float64) / (count + 1)
        alpha = u * u * (3.0 - 2.0 * u)
        blended_waypoints = waypoints.copy()
        blended_left = left_gripper.copy()
        blended_right = right_gripper.copy()
        if self.chunk_boundary_blend_scope in {'arm', 'all'}:
            waypoint_dims = (
                slice(None) if self.chunk_boundary_blend_scope == 'all' else
                slice(0, 2 * self._ARM_DIM))
            blended_waypoints[:count, waypoint_dims] = (
                (1.0 - alpha[:, None]) * old_waypoints[:count, waypoint_dims] +
                alpha[:, None] * blended_waypoints[:count, waypoint_dims])
        if self.chunk_boundary_blend_scope in {'gripper', 'all'}:
            blended_left[:count] = ((1.0 - alpha) * old_left[:count] +
                                    alpha * blended_left[:count])
            blended_right[:count] = ((1.0 - alpha) * old_right[:count] +
                                     alpha * blended_right[:count])
        return blended_waypoints, blended_left, blended_right, count

    def begin_waypoint_stream(self) -> None:
        """Prepare for a persistent policy-rate ActionQueue consumer.

        A queue consumer calls :meth:`execute_waypoint` at a stable policy
        rate.  It must not compete with an older trajectory feeder, but it
        deliberately keeps the existing MotionController alive so its 300 Hz
        interpolation/publish phase is not restarted at chunk boundaries.
        """
        self.stop_trajectory()
        with self._control_lock:
            self._recovery_blend_pending = False
            self._clear_active_trajectory_locked()

    def execute_waypoint(self,
                         left_arm,
                         right_arm,
                         left_gripper: float,
                         right_gripper: float,
                         head=None,
                         dt: float = 1.0 / 30.0,
                         action_metadata=None,
                         trajectory_index: int = 0) -> None:
        """Execute one action using the upstream ``Tron2Env.step`` path.

        The action extraction, gripper clipping, current-head passthrough, and
        no-argument ``command_joints`` call below intentionally mirror
        ``tron2_env.env.Tron2Env.step`` at commit
        ``5b7b145229416f3731f61657e6fa71c89c37bc9d``. In particular, this path
        does not poll measured state or run the custom trajectory guards at
        every 30 Hz policy tick; ``MotionController`` owns interpolation and
        publishes the latest target at 300 Hz.
        """
        self._validate_positive('dt', dt)
        left = np.asarray(left_arm, dtype=np.float64).reshape(-1)
        right = np.asarray(right_arm, dtype=np.float64).reshape(-1)
        action = np.concatenate(
            [left, [float(left_gripper)], right, [float(right_gripper)]])
        if head is not None:
            action = np.concatenate(
                [action,
                 np.asarray(head, dtype=np.float64).reshape(-1)])
        if len(action) not in (16, 18):
            raise ValueError(
                f'TRON2 action must contain 16 or 18 values; got {len(action)}.'
            )

        with self._control_lock:
            if self._traj_thread is not None and self._traj_thread.is_alive():
                raise RuntimeError('A trajectory feeder is active while the '
                                   'RTC waypoint stream is running.')
            controller = self._start_motion_controller_locked(dt)

            arm_action = np.concatenate([action[:7], action[8:15]])
            if len(action) >= 18:
                head_action = action[16:18]
            else:
                head_action = controller.get_head_position()
            full_servo_action = np.concatenate([arm_action, head_action])

            gripper_action = np.clip(
                np.asarray([action[7], action[15]]) * 100.0, 0.0, 100.0)
            controller.set_gripper(
                left_opening=float(gripper_action[0]),
                right_opening=float(gripper_action[1]),
            )
            controller.command_joints(full_servo_action)
            issued_at_unix = time.time()
            issued_at_monotonic = time.monotonic()
            self._last_servoj_waypoint = full_servo_action.copy()
            self._last_left_gripper = float(gripper_action[0] / 100.0)
            self._last_right_gripper = float(gripper_action[1] / 100.0)
            self._recovery_blend_pending = False

        self._notify_issued_action(
            waypoint=full_servo_action,
            left_gripper=float(gripper_action[0] / 100.0),
            right_gripper=float(gripper_action[1] / 100.0),
            trajectory_index=trajectory_index,
            issued_at_unix=issued_at_unix,
            issued_at_monotonic=issued_at_monotonic,
            action_metadata=dict(action_metadata or {}),
        )

    def execute_trajectory(self,
                           left_arm_trajectory,
                           right_arm_trajectory,
                           left_gripper_trajectory,
                           right_gripper_trajectory,
                           head_trajectory=None,
                           dt: float = 0.1,
                           async_exec: bool = False,
                           action_metadata=None):
        """Feed policy waypoints to the ``tron2_env`` ServoJ controller."""
        self._validate_positive('dt', dt)
        old_leftover = self._stop_trajectory_feeder(
            capture_leftover=self.chunk_boundary_blend_enabled)
        self._traj_stop_event = threading.Event()
        self._trajectory_error = None
        args = (left_arm_trajectory, right_arm_trajectory,
                left_gripper_trajectory, right_gripper_trajectory,
                head_trajectory, float(dt), self._traj_stop_event,
                old_leftover, dict(action_metadata or {}))

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

    def _run_servoj_trajectory(self,
                               left_arm_trajectory,
                               right_arm_trajectory,
                               left_gripper_trajectory,
                               right_gripper_trajectory,
                               head_trajectory,
                               dt,
                               stop_event,
                               old_leftover=None,
                               action_metadata=None):
        controller = None
        issued_any = False
        completed = False
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

                waypoints, left_gripper, right_gripper, boundary_blend_count = (
                    self._apply_chunk_boundary_blend(waypoints, left_gripper,
                                                     right_gripper,
                                                     old_leftover))
                recovery_count = 0
                if boundary_blend_count:
                    self._recovery_blend_pending = False
                    logger.info(
                        'ServoJ chunk boundary: smoothstep-blended %d/%d '
                        'policy frames (scope=%s).', boundary_blend_count,
                        self.chunk_boundary_blend_frames,
                        self.chunk_boundary_blend_scope)
                else:
                    waypoints, left_gripper, right_gripper, recovery_count = (
                        self._apply_recovery_blend(waypoints, left_gripper,
                                                   right_gripper))
                    if recovery_count:
                        logger.warning(
                            'ServoJ action stream recovered after holding the '
                            'last target; blending the next %d/%d policy '
                            'frames.', recovery_count,
                            self.recovery_blend_frames)
                if boundary_blend_count or recovery_count:
                    # The transformed trajectory is checked again because the
                    # blend changes the executable waypoint sequence.
                    self._validate_servoj_waypoint_deltas(
                        self._state_to_servoj(
                            np.asarray(state, dtype=np.float64)), waypoints)
                # Any pending transition has now been consumed. If this feeder
                # drains, fails, or is later stopped, state is updated below.
                self._recovery_blend_pending = False
                self._active_servoj_waypoints = waypoints.copy()
                self._active_left_gripper = left_gripper.copy()
                self._active_right_gripper = right_gripper.copy()
                self._active_next_index = 0

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
                issued_at_unix = time.time()
                issued_at_monotonic = time.monotonic()
                with self._control_lock:
                    self._last_servoj_waypoint = waypoint.copy()
                    self._last_left_gripper = float(left_gripper[index])
                    self._last_right_gripper = float(right_gripper[index])
                    self._active_next_index = index + 1
                self._notify_issued_action(
                    waypoint=waypoint,
                    left_gripper=left_gripper[index],
                    right_gripper=right_gripper[index],
                    trajectory_index=index,
                    issued_at_unix=issued_at_unix,
                    issued_at_monotonic=issued_at_monotonic,
                    action_metadata=action_metadata,
                )
                issued_any = True

                deadline = start + (index + 1) * dt
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    stop_event.wait(remaining)
            completed = issued_any and not stop_event.is_set()
        except ValueError as exc:
            # A rejected trajectory is recoverable. If MotionController has
            # already started, issuing no replacement command keeps its last
            # accepted ServoJ target unchanged for the next inference cycle.
            self._trajectory_error = exc
            stop_event.set()
            with self._control_lock:
                self._recovery_blend_pending = (
                    self._motion_controller is not None
                    and self._last_servoj_waypoint is not None)
            raise
        except Exception as exc:
            self._trajectory_error = exc
            stop_event.set()
            with self._control_lock:
                self._disconnect_control_locked()
            raise
        finally:
            if completed:
                with self._control_lock:
                    # With no queued waypoint left, MotionController keeps
                    # publishing this final target. The next valid trajectory
                    # should recover gradually from that held command.
                    if self._motion_controller is controller:
                        self._recovery_blend_pending = True

    def _stop_trajectory_feeder(self, capture_leftover: bool = False):
        self._traj_stop_event.set()
        thread = self._traj_thread
        was_active = thread is not None and thread.is_alive()
        if (thread is not None and thread.is_alive()
                and thread is not threading.current_thread()):
            thread.join(timeout=max(2.0, self.state_timeout))
            if thread.is_alive():
                raise RuntimeError('ServoJ trajectory feeder did not stop.')
        self._traj_thread = None
        with self._control_lock:
            leftover = (
                self._snapshot_active_leftover_locked()
                if capture_leftover and was_active else None)
            self._clear_active_trajectory_locked()
        return leftover

    def stop_trajectory(self):
        """Stop policy waypoint feeding while holding the latest target."""
        self._stop_trajectory_feeder(capture_leftover=False)

    def wait_for_trajectory(self):
        """Wait until the accepted policy trajectory has finished feeding.

        Unlike :meth:`stop_trajectory`, this does not set the stop event. It is
        used by the keyboard state machine so idle-only MoveJ reset cannot race
        an accepted asynchronous ServoJ trajectory.
        """
        thread = self._traj_thread
        if (thread is not None and thread.is_alive()
                and thread is not threading.current_thread()):
            thread.join()
        self._traj_thread = None
        with self._control_lock:
            self._clear_active_trajectory_locked()

        error = self._trajectory_error
        self._trajectory_error = None
        if error is not None:
            raise error

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
