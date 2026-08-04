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

import contextlib
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

from ..utils.root import RUNNERS
from ..utils.trajectory_utils import resample_remaining
from .base_inference_runner import BaseInferenceRunner


class _TerminalKeyReader:
    """Read single robot-client keys and restore terminal state on exit."""

    def __init__(self, input_stream=None):
        self._input_stream = sys.stdin if input_stream is None else input_stream
        self._fd = None
        self._saved_attributes = None
        self._keys = queue.Queue()
        self._stop_event = threading.Event()
        self._reader_error = None
        self._reader_thread = None

    def __enter__(self):
        try:
            self._fd = self._input_stream.fileno()
        except (AttributeError, OSError) as exc:
            raise RuntimeError('TRON2 keyboard control requires a terminal '
                               'stdin with a file descriptor.') from exc
        if not os.isatty(self._fd):
            raise RuntimeError(
                'TRON2 keyboard control requires an interactive TTY. Run the '
                'remote client in a foreground terminal.')

        self._saved_attributes = termios.tcgetattr(self._fd)
        try:
            tty.setcbreak(self._fd)
            self._reader_thread = threading.Thread(
                target=self._read_keys,
                daemon=True,
                name='Tron2-client-key-reader',
            )
            self._reader_thread.start()
        except BaseException:
            termios.tcsetattr(self._fd, termios.TCSADRAIN,
                              self._saved_attributes)
            raise
        return self

    def _read_keys(self):
        try:
            while not self._stop_event.is_set():
                readable, _, _ = select.select([self._fd], [], [], 0.1)
                if not readable:
                    continue
                data = os.read(self._fd, 1)
                if not data:
                    raise EOFError('TRON2 client terminal input closed.')
                self._keys.put(data.decode('utf-8', errors='ignore'))
        except BaseException as exc:
            if not self._stop_event.is_set():
                self._reader_error = exc
                self._keys.put(None)

    def get_key(self, timeout=None):
        try:
            key = self._keys.get(timeout=timeout)
        except queue.Empty:
            return None
        if key is None and self._reader_error is not None:
            raise RuntimeError('TRON2 client key reader failed.') from \
                self._reader_error
        return key

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        if self._saved_attributes is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN,
                              self._saved_attributes)


@RUNNERS.register_module()
class Tron2InferenceRunner(BaseInferenceRunner):
    """Runner for Tron2 dual-arm robot inference tasks.

    This runner handles real-time inference tasks for dual-arm robotic
    manipulation using Vision-Language-Action (VLA) models. With
    ``Tron2EnvOperator`` it receives aligned observations from the TRON2
    Bridge WebSocket and controls both arms through the robot WebSocket.

    The runner supports various camera configurations, action chunking,
    and provides a complete inference pipeline from sensor data to
    dual-arm robot actuation.

    Args:
        gripper_threshold (float, optional): Gripper 0-1; below -> 0 (closed).
            Defaults to 0.1.

        prepare_pose (List[float], optional): Prepare pose for the robot.
            Defaults to None.

        enable_head_control (bool, optional): Whether the runner should send
            head commands during prepare pose execution and trajectory
            execution. Defaults to False.

    """

    def __init__(self,
                 gripper_threshold: float = 0.1,
                 prepare_pose: List[float] = None,
                 enable_head_control: bool = False,
                 async_execution: bool = False,
                 execute_horizon: int = None,
                 action_layout: str = 'tron2_18',
                 dry_run: bool = False,
                 *args,
                 **kwargs):
        self.gripper_threshold = gripper_threshold
        self.enable_head_control = enable_head_control
        self.async_execution = async_execution
        self.execute_horizon = execute_horizon
        self.action_layout = action_layout
        self.dry_run = dry_run
        if self.action_layout not in ('tron2_18', 'tron2_16'):
            raise ValueError(
                f'Unsupported action_layout: {self.action_layout}. '
                "Expected one of {'tron2_18', 'tron2_16'}")
        if self.action_layout == 'tron2_16' and self.enable_head_control:
            raise ValueError(
                'action_layout=tron2_16 does not include head actions. '
                'Set enable_head_control=False.')
        # Set Tron2-specific defaults
        if 'camera_names' not in kwargs or kwargs['camera_names'] is None:
            kwargs['camera_names'] = [
                'cam_high', 'cam_left_wrist', 'cam_right_wrist'
            ]

        if 'operator' not in kwargs or kwargs['operator'] is None:
            kwargs['operator'] = {
                'type': 'Tron2EnvOperator',
                'bridge_host': 'wss://10.192.1.4',
                'bridge_ws_path': '/bridge/ws',
                'connect_websocket': not self.dry_run,
            }
        else:
            kwargs['operator'] = dict(kwargs['operator'])
            kwargs['operator'].setdefault('connect_websocket',
                                          not self.dry_run)

        # Initialize Tron2-specific task descriptions
        if 'task_descriptions' not in kwargs or kwargs[
                'task_descriptions'] is None:
            kwargs['task_descriptions'] = {'1': 'Complete the task.'}

        # Call parent constructor
        super().__init__(*args, **kwargs)

        self.dt = 1.0 / self.publish_rate
        self._chunk_accept_lock = threading.Lock()

        if prepare_pose is None:
            # Initialize Tron2-specific prepare poses
            # [left(7), right(7), head_pitch, head_yaw,
            #  left_gripper(0-1), right_gripper(0-1)]
            self.prepare_pose = [
                [
                    1.2, 0, 0, -2.5, 0, 0, 0, 1.2, 0, 0, -2.5, 0, 0, 0, 0, 0,
                    1, 1
                ],
                [
                    0, 0.24, 0, -2.5, 0.24, 0, 0, 0, -0.24, 0, -2.5, -0.24, 0,
                    0, 0, 0, 1, 1
                ],
                [
                    0, 0.24, 0, -1.56, 0.24, 0, 0, 0, -0.24, 0, -1.56, -0.24,
                    0, 0, 0, 0, 1, 1
                ],
            ]
        else:
            self.prepare_pose = prepare_pose

    def run_setup(self):
        """Verify inference connectivity and adopt checkpoint task metadata."""
        super().run_setup()
        if not self._use_remote:
            return

        import msgpack

        request = msgpack.packb({'endpoint': 'get_deployment_metadata'})
        with self._zmq_lock:
            self._zmq_socket.send(request)
            response = msgpack.unpackb(self._zmq_socket.recv(), raw=False)
        if response.get('error'):
            raise RuntimeError(
                'Inference server does not provide checkpoint deployment '
                'metadata. Restart it with the synchronized server code: '
                f"{response['error']}")

        server_layout = response.get('action_layout')
        if server_layout != self.action_layout:
            raise ValueError('Client/server action layout mismatch: '
                             f'client={self.action_layout!r}, '
                             f'server={server_layout!r}.')
        task_descriptions = response.get('task_descriptions')
        if not isinstance(task_descriptions, dict) or not task_descriptions:
            raise ValueError('Inference checkpoint has no task descriptions; '
                             'robot execution is blocked.')
        invalid = {
            str(task_id): description
            for task_id, description in task_descriptions.items()
            if not isinstance(description, str) or not description.strip()
        }
        if invalid:
            raise ValueError('Inference checkpoint contains invalid task '
                             f'descriptions: {invalid}.')
        self.task_descriptions = {
            str(task_id): description.strip()
            for task_id, description in task_descriptions.items()
        }

        from ..utils import initialize_overwatch
        overwatch = initialize_overwatch(__name__)
        overwatch.info(
            'Checkpoint deployment metadata loaded: '
            f"work_dir={response.get('checkpoint_work_dir', 'unknown')}, "
            f'task_ids={sorted(self.task_descriptions)}')
        for task_id, description in sorted(self.task_descriptions.items()):
            overwatch.info(f'  task {task_id}: {description}')

    def _get_task_description(self, task_id: str) -> str:
        """Require a task ID advertised by the active checkpoint."""
        if task_id not in self.task_descriptions:
            available = ', '.join(sorted(self.task_descriptions)) or 'none'
            raise ValueError(f'Unknown task ID {task_id!r}; active checkpoint '
                             f'only advertises: {available}.')
        return self.task_descriptions[task_id]

    def _wait_for_idle_command(self, key_reader):
        """Select a checkpoint task, start it, or request the prepare pose."""
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        task_buffer = ''
        selected_task_id = None
        print('\n[TRON2 client idle] Type task ID and press Enter. '
              'b=start, r=prepare pose, Ctrl+C=exit.')
        print('Task ID: ', end='', flush=True)

        while True:
            key = key_reader.get_key(timeout=0.1)
            if key is None:
                continue
            command = key.lower()

            if key.isdigit():
                if selected_task_id is not None:
                    selected_task_id = None
                    task_buffer = ''
                    print('\nTask ID: ', end='', flush=True)
                task_buffer += key
                print(key, end='', flush=True)
                continue
            if key in {'\x7f', '\b'}:
                if task_buffer:
                    task_buffer = task_buffer[:-1]
                    print('\b \b', end='', flush=True)
                continue
            if key in {'\r', '\n'}:
                print()
                if not task_buffer:
                    print('Task ID: ', end='', flush=True)
                    continue
                if task_buffer == '0':
                    overwatch.warning(
                        'Task ID 0 is the prepare-pose command; press r while '
                        'the client is idle.')
                    task_buffer = ''
                    print('Task ID: ', end='', flush=True)
                    continue
                try:
                    description = self._get_task_description(task_buffer)
                except ValueError as exc:
                    overwatch.warning('%s', exc)
                    task_buffer = ''
                    print('Task ID: ', end='', flush=True)
                    continue
                selected_task_id = task_buffer
                task_buffer = ''
                overwatch.info('Selected task %s: %s', selected_task_id,
                               description)
                print('Press b to start, or type another task ID and press '
                      'Enter.')
                continue
            if command == 'b':
                if task_buffer:
                    overwatch.warning(
                        'Press Enter to confirm task ID %s before starting.',
                        task_buffer)
                    continue
                if selected_task_id is None:
                    overwatch.warning('Select a task ID before pressing b.')
                    continue
                return 'start', selected_task_id
            if command == 'r':
                return 'prepare', None
            if command == 's':
                overwatch.info('Inference is already stopped; no action is '
                               'being generated or sent.')
                continue
            if key not in {' ', '\t'}:
                overwatch.warning(
                    'Unknown idle key %r. Use task ID + Enter, b, or r.', key)

    def _monitor_active_keys(self, key_reader, stop_requested: threading.Event,
                             monitor_done: threading.Event,
                             monitor_errors: list[BaseException]):
        """Accept only ``s`` while inference or a chunk is executing."""
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        try:
            while not monitor_done.is_set():
                key = key_reader.get_key(timeout=0.1)
                if key is None:
                    continue
                command = key.lower()
                if command == 's':
                    with self._chunk_accept_lock:
                        already_stopping = stop_requested.is_set()
                        stop_requested.set()
                    if already_stopping:
                        overwatch.info('Stop is already pending; waiting for '
                                       'the accepted action chunk to finish.')
                    else:
                        overwatch.info(
                            'Stop requested: no further inference result will '
                            'be accepted; the current accepted chunk will '
                            'finish.')
                elif command == 'r':
                    overwatch.warning(
                        'r is ignored while running. Press s, wait for the '
                        'client to report idle, then press r.')
                elif command == 'b':
                    overwatch.info('Inference is already running. Press s to '
                                   'stop after the accepted chunk finishes.')
                elif key not in {'\r', '\n', ' ', '\t'}:
                    overwatch.warning(
                        'Key %r is ignored while inference is running; only s '
                        'is active.', key)
        except BaseException as exc:
            monitor_errors.append(exc)
            with self._chunk_accept_lock:
                stop_requested.set()

    def _run_continuous_task(self, instruction: str,
                             stop_requested: threading.Event):
        """Run sequential non-RTC chunks until ``s`` requests a stop."""
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        self._prev_ctx = None
        chunk_index = 0
        while not stop_requested.is_set():
            self._action_ctx = SimpleNamespace(instruction=instruction)
            inputs = self._preprocess(instruction)
            if stop_requested.is_set():
                break

            if self._use_remote:
                autocast_context = contextlib.nullcontext()
            else:
                import torch
                autocast_context = torch.autocast(
                    'cuda',
                    dtype=self.mixed_precision_dtype,
                    enabled=self.enable_mixed_precision,
                )
            with autocast_context:
                raw_action = self._predict_action(inputs)

            try:
                actions = self._postprocess_actions(raw_action)
            except ValueError as exc:
                overwatch.warning(
                    '[Hold] action chunk postprocessing failed (%s); keeping '
                    'the previous ServoJ target unchanged.', exc)
                continue

            with self._chunk_accept_lock:
                accepted = not stop_requested.is_set()
            if not accepted:
                overwatch.info('Discarding the inference result because s was '
                               'pressed before the chunk was accepted.')
                break

            try:
                self._execute_actions(actions, rate=None)
            except ValueError as exc:
                overwatch.warning(
                    '[Hold] action chunk rejected (%s); keeping the previous '
                    'ServoJ target unchanged.', exc)
                continue

            self._prev_ctx = self._action_ctx
            chunk_index += 1
            overwatch.info('Completed non-RTC action chunk %d (%d frames).',
                           chunk_index, len(actions))

    def _run_selected_task(self, key_reader, task_id: str):
        """Run a selected task while the PCM key monitor owns stdin."""
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        instruction = self._get_task_description(task_id)
        stop_requested = threading.Event()
        self._chunk_accept_lock = threading.Lock()
        monitor_done = threading.Event()
        monitor_errors: list[BaseException] = []
        monitor_thread = threading.Thread(
            target=self._monitor_active_keys,
            args=(key_reader, stop_requested, monitor_done, monitor_errors),
            daemon=True,
            name='Tron2-client-active-key-monitor',
        )
        monitor_thread.start()
        try:
            if task_id in self.task_pose_sequences:
                self.execute_task_pose(task_id)
            if not stop_requested.is_set():
                self._run_continuous_task(instruction, stop_requested)
        finally:
            monitor_done.set()
            monitor_thread.join(timeout=1.0)
            if monitor_thread.is_alive():
                raise RuntimeError('TRON2 active key monitor did not stop.')
        if monitor_errors:
            raise RuntimeError('TRON2 active key monitor failed.') from \
                monitor_errors[0]
        overwatch.info(
            'Task %s stopped. Select a task ID before pressing b again.',
            task_id)

    def run(self,
            initial_instruction:
            str = 'place it in the brown paper bag with right arm'):
        """Run the PCM-local non-RTC keyboard state machine until Ctrl+C."""
        from ..utils import initialize_overwatch

        del initial_instruction
        overwatch = initialize_overwatch(__name__)
        overwatch.info('Starting non-RTC TRON2 client keyboard control. All '
                       'b/s/r handling runs on this robot computer.')
        if self._use_remote:
            inference_context = contextlib.nullcontext()
        else:
            import torch
            inference_context = torch.inference_mode()

        with inference_context:
            with _TerminalKeyReader() as key_reader:
                while True:
                    command, task_id = self._wait_for_idle_command(key_reader)
                    if command == 'prepare':
                        try:
                            self._move_to_prepare_pose()
                            overwatch.info(
                                'Prepare-pose sequence completed. Select a '
                                'task ID, then press b.')
                        except Exception as exc:
                            overwatch.error(
                                'Prepare-pose command failed; client remains '
                                'idle: %s', exc)
                        continue
                    self._run_selected_task(key_reader, task_id)

    def get_robot_observation(self) -> Dict:
        """Get one synchronized observation from the configured operator.

        ``Tron2EnvOperator`` returns the official Bridge/OpenPI structure:
        ``images`` plus an 18-dimensional state ordered as left arm, left
        gripper, right arm, right gripper, and head. A legacy tuple fallback
        remains for custom configurations that still use ``Tron2Operator``.

        Returns:
            Official Bridge observation dictionary, or the legacy synchronized
            tuple returned by ``Tron2Operator``.
        """
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)

        if hasattr(self.ros_operator, 'get_observation'):
            return self.ros_operator.get_observation()

        rate_period = 1.0 / self.publish_rate
        print_flag = True
        time.sleep(rate_period)

        while True:
            result = self.ros_operator.get_frame()
            if not result:
                if print_flag:
                    overwatch.info(
                        'Synchronization failed in get_ros_observation')
                    print_flag = False
                time.sleep(rate_period)
                continue

            print_flag = True
            (img_top, img_left, img_right, img_top_depth, img_left_depth,
             img_right_depth, arm_left, arm_right, head,
             robot_gripper) = result

            return (img_top, img_left, img_right, arm_left, arm_right, head,
                    robot_gripper)

    def get_ros_observation(self):
        """Backward-compatible alias for older external callers."""
        return self.get_robot_observation()

    def update_observation_window(self) -> Dict:
        """Update the observation window with latest sensor data.

        Maintains a sliding window of observations for temporal context.
        The window includes robot joint positions from both arms and
        camera images from three viewpoints.

        Returns:
            Dict: Latest observation containing camera images keyed by
                camera_names and a qpos vector matching action_layout:
                - tron2_18: left7 + right7 + head2 + left_gripper
                    + right_gripper
                - tron2_16: left7 + left_gripper + right7 + right_gripper

        Note:
            The first observation in a new window is a dummy placeholder
            to maintain consistent window size.
        """
        from collections import deque

        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)

            # Add dummy observation for initialization
            dummy_obs = {'qpos': None}
            for camera_name in self.camera_names:
                dummy_obs[camera_name] = None
            self.observation_window.append(dummy_obs)

        sensor_observation = self.get_robot_observation()
        if isinstance(sensor_observation, dict):
            images = sensor_observation.get('images', {})
            state = np.asarray(
                sensor_observation.get('state'), dtype=np.float32)
            if state.shape != (18, ) or not np.all(np.isfinite(state)):
                raise RuntimeError('TRON2 Bridge observation must provide a '
                                   f'finite 18-dim state; got {state.shape}.')
            try:
                img_top = images['cam_high']
                img_left = images['cam_left_wrist']
                img_right = images['cam_right_wrist']
            except KeyError as exc:
                raise RuntimeError('TRON2 Bridge observation is missing '
                                   f'image {exc.args[0]!r}.') from exc

            # tron2_env Bridge/OpenPI state layout:
            # [left7, left_gripper, right7, right_gripper, head2].
            if self.action_layout == 'tron2_16':
                qpos = state[:16].copy()
            else:
                # Preserve FluxVLA's historical tron2_18 training layout:
                # [left7, right7, head2, left_gripper, right_gripper].
                qpos = np.concatenate([
                    state[:7], state[8:15], state[16:18], state[7:8],
                    state[15:16]
                ])
        else:
            (img_top, img_left, img_right, arm_left, arm_right, head,
             robot_gripper) = sensor_observation

            gripper_pos = robot_gripper.position
            left_gripper = np.array(gripper_pos[0:1])
            right_gripper = np.array(gripper_pos[1:2])
            left_arm = np.array(arm_left.position)
            right_arm = np.array(arm_right.position)
            head_joints = np.array(head.position)
            if self.action_layout == 'tron2_16':
                qpos = np.concatenate(
                    (left_arm, left_gripper, right_arm, right_gripper), axis=0)
            else:
                qpos = np.concatenate(
                    (left_arm, right_arm, head_joints, left_gripper,
                     right_gripper),
                    axis=0,
                )

        # Apply JPEG compression to match training conditions
        img_top = self._apply_jpeg_compression(img_top)
        img_left = self._apply_jpeg_compression(img_left)
        img_right = self._apply_jpeg_compression(img_right)

        # Create observation dictionary
        observation = {
            'qpos': qpos,
            self.camera_names[0]: img_top,  # cam_high
            self.camera_names[1]: img_left,  # cam_left_wrist
            self.camera_names[2]: img_right,  # cam_right_wrist
        }

        self.observation_window.append(observation)
        return self.observation_window[-1]

    def _move_to_prepare_pose(self):
        """Move robot to predefined preparation pose.

        Supports prepare_pose as:
        - 18-dim: left7 + right7 + head2 + left_gripper + right_gripper
        - 16-dim: left7 + left_gripper + right7 + right_gripper
        - List of 18-dim lists: execute each pose sequentially
        """
        if self.dry_run:
            print('[Tron2InferenceRunner] dry_run=True, skip prepare pose.')
            return

        if self.prepare_pose is None:
            return

        # Check if it's a list of poses or single pose
        if isinstance(self.prepare_pose[0], (list, tuple, np.ndarray)):
            # Multiple poses - execute sequentially
            poses = self.prepare_pose
        else:
            # Single pose
            poses = [self.prepare_pose]

        for pose in poses:
            pose = np.array(pose)
            if len(pose) == 16:
                left_joints = pose[:7]
                left_gripper = pose[7]
                right_joints = pose[8:15]
                right_gripper = pose[15]
                head_joints = None
            elif len(pose) >= 18:
                left_joints = pose[:7]
                right_joints = pose[7:14]
                head_joints = (
                    list(pose[14:16]) if self.enable_head_control else None)
                left_gripper = pose[16]
                right_gripper = pose[17]
            else:
                raise ValueError(
                    f'Unsupported prepare_pose length: {len(pose)}. '
                    'Expected 16 or 18.')

            self.ros_operator.move_to_targets(
                left_joints,
                right_joints,
                head=head_joints,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
                control_rate=30)

        self.last_actions = None

    def _predict_action(self, inputs: dict):
        self._action_ctx.inference_start = time.time()
        return super()._predict_action(inputs)

    # Layouts:
    # - tron2_18: left7 + right7 + head2 + left_gripper + right_gripper
    # - tron2_16: left7 + left_gripper + right7 + right_gripper
    GRIPPER_CLOSED = 0.0

    def _action_parts(self, actions: np.ndarray):
        """Split denormalized action chunks into robot command arrays."""
        if actions.ndim != 2:
            raise ValueError(
                f'Tron2 actions must be 2-D [T, D], got {actions.shape}')
        if self.action_layout == 'tron2_16':
            if actions.shape[1] < 16:
                raise ValueError(
                    f'tron2_16 expects action dim >= 16, got {actions.shape}')
            return dict(
                left_arm=actions[:, :7],
                right_arm=actions[:, 8:15],
                left_gripper=actions[:, 7],
                right_gripper=actions[:, 15],
                head=None)
        if actions.shape[1] < 18:
            raise ValueError(
                f'tron2_18 expects action dim >= 18, got {actions.shape}')
        return dict(
            left_arm=actions[:, :7],
            right_arm=actions[:, 7:14],
            left_gripper=actions[:, 16],
            right_gripper=actions[:, 17],
            head=actions[:, 14:16] if self.enable_head_control else None)

    def _postprocess_actions(self, raw_action):
        """Denormalize and snap near-closed grippers to fully closed."""
        actions = super()._postprocess_actions(raw_action)
        gripper_cols = (7, 15) if self.action_layout == 'tron2_16' else (16,
                                                                         17)
        for col in gripper_cols:
            if actions.shape[1] <= col:
                raise ValueError(
                    f'{self.action_layout} expects gripper column {col}, '
                    f'but action shape is {actions.shape}')
            actions[:,
                    col] = np.where(actions[:, col] < self.gripper_threshold,
                                    self.GRIPPER_CLOSED, actions[:, col])
        return actions

    def _execute_actions(self, actions: np.ndarray, rate):
        """Execute a chunk of dual-arm robot actions.

        In async mode, skips elapsed steps and executes in background thread.
        """
        if self.disable_puppet_arm:
            return

        ctx = self._action_ctx

        if self.async_execution and self._prev_ctx is not None:
            ctx.action_timestamp = ctx.inference_start
            offset = (time.time() - ctx.action_timestamp) / self.dt
            actions = resample_remaining(actions, offset)
        else:
            ctx.action_timestamp = time.time()
            if self.execute_horizon is not None:
                actions = actions[:self.execute_horizon]

        parts = self._action_parts(actions)

        if self.dry_run:
            print('[Tron2InferenceRunner] dry_run=True, skip execution. '
                  f'action_shape={actions.shape}, '
                  f'layout={self.action_layout}, '
                  f'first_action={actions[0].tolist()}')
            return

        self.ros_operator.execute_trajectory(
            left_arm_trajectory=parts['left_arm'],
            right_arm_trajectory=parts['right_arm'],
            left_gripper_trajectory=parts['left_gripper'],
            right_gripper_trajectory=parts['right_gripper'],
            head_trajectory=parts['head'],
            dt=self.dt,
            async_exec=self.async_execution)

        if self.async_execution and self.execute_horizon is not None:
            time.sleep(self.execute_horizon * self.dt)

    def cleanup(self):
        """Clean up resources."""
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        overwatch.info('Cleaning up Tron2InferenceRunner')

        try:
            try:
                if hasattr(self.ros_operator, 'stop_trajectory'):
                    self.ros_operator.stop_trajectory()
            finally:
                if hasattr(self.ros_operator, 'close'):
                    self.ros_operator.close()
        finally:
            super().cleanup()

        overwatch.info('Tron2InferenceRunner cleanup completed')
