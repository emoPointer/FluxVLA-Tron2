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
"""Thin FluxVLA adapter around the public :class:`tron2_env.Tron2Env`.

This operator deliberately does not reproduce the TRON2 execution stack.
The upstream environment owns Bridge observations, robot WebSocket state,
ServoJ interpolation/publication, gripper commands, and reset bring-up.  The
adapter only translates the FluxVLA runner's method calls into ``get_obs()``,
``step()`` and environment reconstruction for the client-local reset command.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from fluxvla.engines.utils.root import OPERATORS


@OPERATORS.register_module()
class Tron2NativeEnvOperator:
    """Delegate real-robot observation and execution to upstream Tron2Env."""

    def __init__(
        self,
        robot_ip: str,
        ws_port: int = 5000,
        bridge_host: str = 'wss://10.192.1.4',
        bridge_ws_path: str = '/bridge/ws',
        bridge_image_topics: Optional[Dict[str, str]] = None,
        bridge_joint_topics: Optional[Dict[str, str]] = None,
        bridge_image_max_fps: int = 0,
        bridge_align_max_delay_ms: int = 200,
        bridge_verify_tls: bool = False,
        bridge_state_source: str = 'legacy',
        state_dim: int = 18,
        fps: float = 30.0,
        publish_rate: float = 300.0,
        init_joints=None,
        init_head=None,
        init_ee_z_min: Optional[float] = -0.6,
        init_gripper_opening: float = 1.0,
        reset_gripper_open_wait_s: float = 0.5,
        state_queue_maxlen: int = 7,
        state_polling_rate: float = 200.0,
        connection_timeout: float = 5.0,
        connect_websocket: bool = True,
    ) -> None:
        if bridge_state_source not in {'bridge', 'legacy'}:
            raise ValueError(
                "bridge_state_source must be 'bridge' or 'legacy', got "
                f'{bridge_state_source!r}.')
        if state_dim not in {16, 18}:
            raise ValueError(f'state_dim must be 16 or 18, got {state_dim}.')
        if (isinstance(reset_gripper_open_wait_s, bool)
                or not isinstance(reset_gripper_open_wait_s, (int, float))
                or not np.isfinite(reset_gripper_open_wait_s)
                or reset_gripper_open_wait_s < 0):
            raise ValueError('reset_gripper_open_wait_s must be finite and '
                             '>= 0, got '
                             f'{reset_gripper_open_wait_s!r}.')

        self._env_kwargs = dict(
            robot_ip=str(robot_ip),
            ws_port=int(ws_port),
            bridge_host=str(bridge_host),
            bridge_ws_path=str(bridge_ws_path),
            bridge_image_topics=(None if bridge_image_topics is None else
                                 dict(bridge_image_topics)),
            bridge_joint_topics=(None if bridge_joint_topics is None else
                                 dict(bridge_joint_topics)),
            bridge_image_max_fps=int(bridge_image_max_fps),
            bridge_align_max_delay_ms=int(bridge_align_max_delay_ms),
            bridge_verify_tls=bool(bridge_verify_tls),
            bridge_state_source=bridge_state_source,
            state_dim=int(state_dim),
            fps=float(fps),
            publish_rate=float(publish_rate),
            init_joints=(None if init_joints is None else list(init_joints)),
            init_head=None if init_head is None else list(init_head),
            init_ee_z_min=init_ee_z_min,
            init_gripper_opening=float(init_gripper_opening),
            state_queue_maxlen=int(state_queue_maxlen),
            state_polling_rate=float(state_polling_rate),
            connection_timeout=float(connection_timeout),
        )
        self._reset_gripper_open_wait_s = float(reset_gripper_open_wait_s)
        self._lock = threading.RLock()
        self._issued_action_callback = None
        self._env = None
        self._closed = False
        if connect_websocket:
            self._env = self._create_env()

    @staticmethod
    def _runtime_types():
        try:
            from tron2_env import (BridgeConfig, CameraConfig, EnvConfig,
                                   Tron2Config, Tron2Env)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'Tron2NativeEnvOperator requires the pinned public '
                'tron2-env[bridge] package.') from exc
        return BridgeConfig, CameraConfig, EnvConfig, Tron2Config, Tron2Env

    def _create_env(self):
        (bridge_config_type, camera_config_type, env_config_type,
         robot_config_type, env_type) = self._runtime_types()
        cfg = self._env_kwargs
        bridge_kwargs: Dict[str, Any] = dict(
            host=cfg['bridge_host'],
            ws_path=cfg['bridge_ws_path'],
            image_max_fps=cfg['bridge_image_max_fps'],
            align_max_delay_ms=cfg['bridge_align_max_delay_ms'],
            verify_tls=cfg['bridge_verify_tls'],
            save_debug_images=False,
        )
        if cfg['bridge_image_topics'] is not None:
            bridge_kwargs['image_topics'] = cfg['bridge_image_topics']
        if cfg['bridge_joint_topics'] is not None:
            bridge_kwargs['joint_topics'] = cfg['bridge_joint_topics']

        robot_config = robot_config_type(
            robot_ip=cfg['robot_ip'],
            port=cfg['ws_port'],
            init_joints=cfg['init_joints'],
            init_head=cfg['init_head'],
            init_ee_z_min=cfg['init_ee_z_min'],
            state_queue_maxlen=cfg['state_queue_maxlen'],
            polling_rate=cfg['state_polling_rate'],
            connection_timeout=cfg['connection_timeout'],
        )
        env_config = env_config_type(
            robot_config=robot_config,
            camera_config=camera_config_type(
                camera_names=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                save_debug_images=False,
            ),
            control_backend='websocket',
            publish_rate=cfg['publish_rate'],
            fps=cfg['fps'],
            init_gripper_opening=cfg['init_gripper_opening'],
            observation_source='bridge',
            state_dim=cfg['state_dim'],
            bridge_state_source=cfg['bridge_state_source'],
            bridge_config=bridge_config_type(**bridge_kwargs),
        )
        return env_type(env_config)

    def _require_env(self):
        if self._closed:
            raise RuntimeError('Tron2NativeEnvOperator is closed.')
        if self._env is None:
            raise RuntimeError(
                'Native Tron2Env is not connected. This operator cannot '
                'supply observations or execute actions in offline mode.')
        return self._env

    @property
    def native_env(self):
        """Return the actual public Tron2Env instance used for deployment."""
        with self._lock:
            return self._require_env()

    def get_observation(self) -> Dict[str, Any]:
        """Call upstream ``Tron2Env.get_obs`` without local reconstruction."""
        with self._lock:
            return self._require_env().get_obs()

    def begin_waypoint_stream(self) -> None:
        """Compatibility hook; upstream Tron2Env is already streaming."""

    def execute_waypoint(self,
                         left_arm,
                         right_arm,
                         left_gripper: float,
                         right_gripper: float,
                         head=None,
                         dt: float = 1.0 / 30.0,
                         action_metadata=None,
                         trajectory_index: int = 0) -> None:
        """Build the public 16/18-D layout and call ``Tron2Env.step``."""
        del dt
        action = np.concatenate([
            np.asarray(left_arm).reshape(-1),
            [float(left_gripper)],
            np.asarray(right_arm).reshape(-1),
            [float(right_gripper)],
        ])
        if head is not None:
            action = np.concatenate([action, np.asarray(head).reshape(-1)])

        with self._lock:
            env = self._require_env()
            env.step(action)
            waypoint = np.asarray(env.last_action, dtype=np.float64).copy()
            callback = self._issued_action_callback
            issued_at_unix = time.time()
            issued_at_monotonic = time.monotonic()

        if callback is not None:
            callback(
                waypoint=waypoint,
                left_gripper=float(np.clip(left_gripper, 0.0, 1.0)),
                right_gripper=float(np.clip(right_gripper, 0.0, 1.0)),
                trajectory_index=int(trajectory_index),
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
                           dt: float = 1.0 / 30.0,
                           async_exec: bool = False,
                           action_metadata=None) -> None:
        """Feed a complete non-RTC chunk through upstream ``step`` calls.

        ``Tron2InferenceRunner`` submits a trajectory, whereas the RTC runner
        submits individual waypoints.  Both paths deliberately converge on
        :meth:`execute_waypoint` so action layout, head passthrough, gripper
        handling and issued-action recording remain identical.
        """
        if async_exec:
            raise ValueError(
                'Tron2NativeEnvOperator only supports synchronous trajectory '
                'execution. Set inference.async_execution=False.')
        if (isinstance(dt, bool) or not isinstance(dt, (int, float))
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError(f'dt must be finite and > 0, got {dt!r}.')

        left = np.asarray(left_arm_trajectory, dtype=np.float64)
        right = np.asarray(right_arm_trajectory, dtype=np.float64)
        left_gripper = np.asarray(
            left_gripper_trajectory, dtype=np.float64).reshape(-1)
        right_gripper = np.asarray(
            right_gripper_trajectory, dtype=np.float64).reshape(-1)
        if left.ndim != 2 or left.shape[1] != 7:
            raise ValueError('left_arm_trajectory must have shape [T, 7], '
                             f'got {left.shape}.')
        if right.ndim != 2 or right.shape[1] != 7:
            raise ValueError('right_arm_trajectory must have shape [T, 7], '
                             f'got {right.shape}.')

        count = left.shape[0]
        expected_vector_shape = (count, )
        if right.shape[0] != count:
            raise ValueError(
                'Arm trajectories must contain the same number of frames; '
                f'got left={count}, right={right.shape[0]}.')
        if left_gripper.shape != expected_vector_shape:
            raise ValueError(
                'left_gripper_trajectory must contain one value per frame; '
                f'expected {expected_vector_shape}, got '
                f'{left_gripper.shape}.')
        if right_gripper.shape != expected_vector_shape:
            raise ValueError(
                'right_gripper_trajectory must contain one value per frame; '
                f'expected {expected_vector_shape}, got '
                f'{right_gripper.shape}.')

        head = None
        if head_trajectory is not None:
            head = np.asarray(head_trajectory, dtype=np.float64)
            if head.shape != (count, 2):
                raise ValueError('head_trajectory must have shape [T, 2]; '
                                 f'expected {(count, 2)}, got {head.shape}.')

        named_arrays = {
            'left_arm_trajectory': left,
            'right_arm_trajectory': right,
            'left_gripper_trajectory': left_gripper,
            'right_gripper_trajectory': right_gripper,
        }
        if head is not None:
            named_arrays['head_trajectory'] = head
        for name, values in named_arrays.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f'{name} contains non-finite values.')

        start = time.perf_counter()
        for index in range(count):
            self.execute_waypoint(
                left_arm=left[index],
                right_arm=right[index],
                left_gripper=float(left_gripper[index]),
                right_gripper=float(right_gripper[index]),
                head=None if head is None else head[index],
                dt=float(dt),
                action_metadata=action_metadata,
                trajectory_index=index,
            )
            deadline = start + (index + 1) * float(dt)
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

    def reset_native_env(self) -> None:
        """Open both grippers, then recreate the upstream environment.

        Upstream environment construction performs the configured MoveJ
        bring-up and opens the grippers only afterwards.  The idle ``r`` path
        first opens them through the still-connected old environment so the
        arms do not carry a grasped object through the reset trajectory.
        """
        with self._lock:
            old_env = self._require_env()
            robot = getattr(old_env, 'robot', None)
            set_gripper = getattr(robot, 'set_gripper', None)
            if not callable(set_gripper):
                raise RuntimeError(
                    'Native Tron2Env does not expose robot.set_gripper; '
                    'refusing to MoveJ before opening both grippers.')
            opening = float(
                np.clip(self._env_kwargs['init_gripper_opening'], 0.0, 1.0) *
                100.0)
            set_gripper(
                left_opening=opening,
                right_opening=opening,
            )
            if self._reset_gripper_open_wait_s > 0:
                time.sleep(self._reset_gripper_open_wait_s)
            old_env.close()
            self._env = None
            self._env = self._create_env()

    def set_issued_action_callback(self, callback) -> None:
        """Attach the existing FluxVLA action recorder without changing step."""
        if callback is not None and not callable(callback):
            raise TypeError('Issued action callback must be callable or None.')
        with self._lock:
            self._issued_action_callback = callback

    def stop_trajectory(self) -> None:
        """Compatibility hook; upstream RTC owns no trajectory feeder."""

    def wait_for_trajectory(self) -> None:
        """Compatibility hook; each upstream ``step`` call is non-blocking."""

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            env = self._env
            self._env = None
        if env is not None:
            env.close()
