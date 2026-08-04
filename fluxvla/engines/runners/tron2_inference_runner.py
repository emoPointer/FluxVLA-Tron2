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
import time
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

from ..utils.root import RUNNERS
from ..utils.trajectory_utils import resample_remaining
from .base_inference_runner import BaseInferenceRunner


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
        self.deployment_metadata = dict(response)

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

    def _get_user_task_instruction(self, default_instruction: str):
        """Select only an explicit task advertised by the active checkpoint."""
        del default_instruction
        task_id = input('Enter task ID (0 = prepare pose): ').strip()
        if task_id == '0':
            self._move_to_prepare_pose()
            task_id = input('Enter task ID after prepare pose: ').strip()

        task_description = self._get_task_description(task_id)
        if task_id in self.task_pose_sequences:
            self.execute_task_pose(task_id)

        repeat_text = input('Number of times to repeat the task: ').strip()
        try:
            repeat_count = int(repeat_text)
        except ValueError as exc:
            raise ValueError('Repeat count must be a positive integer; '
                             f'got {repeat_text!r}.') from exc
        if repeat_count <= 0:
            raise ValueError('Repeat count must be a positive integer; '
                             f'got {repeat_count}.')
        return [task_description] * repeat_count

    def run(self,
            initial_instruction:
            str = 'place it in the brown paper bag with right arm'):
        """Run until ``Ctrl+C`` without requiring a local ROS client."""
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        overwatch.info('Starting Tron2 WebSocket inference runner')
        if self._use_remote:
            inference_context = contextlib.nullcontext()
        else:
            import torch
            inference_context = torch.inference_mode()

        with inference_context:
            while True:
                self._run_episode(initial_instruction)

    def _run_episode(self, default_instruction: str):
        """Run one interactive episode without ROS shutdown/rate objects."""
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        t = 0
        while t < self.max_publish_step:
            instructions = self._get_user_task_instruction(
                default_instruction)
            self._prev_ctx = None
            for instruction in instructions:
                self._action_ctx = SimpleNamespace(instruction=instruction)
                inputs = self._preprocess(instruction)
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

                actions = self._postprocess_actions(raw_action)
                self._execute_actions(actions, rate=None)
                self._prev_ctx = self._action_ctx
                t += self.action_chunk
                overwatch.info(f'Published Step {t}')

    def get_robot_observation(
        self
    ) -> Dict:
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
            state = np.asarray(sensor_observation.get('state'),
                               dtype=np.float32)
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
