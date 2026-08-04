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
"""TRON2 remote inference with latency-aligned overlapping action chunks.

The robot-side client remains model-free. A consumer thread executes actions
at the policy rate while the main thread obtains a fresh Bridge observation
and requests the next chunk from the remote GPU server. The overlapping arm
segments are cross-faded from the old plan to the new plan, then the queue is
replaced after discarding the number of frames actually consumed in flight.
"""

from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace

import numpy as np
from tron2_env.rtc import ActionQueue

from ..utils import initialize_overwatch
from ..utils.root import RUNNERS
from .tron2_inference_runner import Tron2InferenceRunner


@RUNNERS.register_module()
class Tron2OverlapInferenceRunner(Tron2InferenceRunner):
    """Cross-fade overlapping remote action chunks during continuous control.

    ``execute_horizon`` is the number of action frames between inference
    starts. For example, a 50-frame chunk and ``execute_horizon=25`` request a
    fresh observation after half of the current plan has been consumed.

    The blend applies only to arm joints. Gripper/head dimensions select the
    old or new discrete plan according to whether the cross-fade weight is
    below or above 0.5.
    """

    def __init__(self,
                 execute_horizon: int = 25,
                 blend_start_weight: float = 0.0,
                 blend_end_weight: float = 1.0,
                 queue_poll_interval_s: float = 0.005,
                 hold_warning_interval_s: float = 1.0,
                 async_execution: bool = False,
                 *args,
                 **kwargs):
        if async_execution:
            raise ValueError(
                'Tron2OverlapInferenceRunner owns its action consumer thread; '
                'set async_execution=False.')
        super().__init__(
            execute_horizon=execute_horizon,
            async_execution=False,
            *args,
            **kwargs,
        )
        if not 0 < self.execute_horizon < self.action_chunk:
            raise ValueError(
                'Overlap execution requires 0 < execute_horizon < '
                f'action_chunk; got {self.execute_horizon} and '
                f'{self.action_chunk}.')
        if not 0.0 <= blend_start_weight <= blend_end_weight <= 1.0:
            raise ValueError(
                'Blend weights must satisfy 0 <= start <= end <= 1; '
                f'got {blend_start_weight} and {blend_end_weight}.')
        if queue_poll_interval_s <= 0:
            raise ValueError('queue_poll_interval_s must be positive.')
        if hold_warning_interval_s <= 0:
            raise ValueError('hold_warning_interval_s must be positive.')

        self.blend_start_weight = float(blend_start_weight)
        self.blend_end_weight = float(blend_end_weight)
        self.queue_poll_interval_s = float(queue_poll_interval_s)
        self.hold_warning_interval_s = float(hold_warning_interval_s)

    def _arm_action_indices(self, action_dim: int) -> np.ndarray:
        if self.action_layout == 'tron2_16':
            indices = tuple(range(7)) + tuple(range(8, 15))
        else:
            indices = tuple(range(14))
        if not indices or max(indices) >= action_dim:
            raise ValueError(
                f'{self.action_layout} arm indices do not fit action dim '
                f'{action_dim}.')
        return np.asarray(indices, dtype=np.int64)

    def _validate_action_chunk(self, actions, name: str) -> np.ndarray:
        chunk = np.asarray(actions)
        if chunk.ndim != 2:
            raise ValueError(f'{name} must have shape [T, D], got '
                             f'{chunk.shape}.')
        if chunk.shape[0] != self.action_chunk:
            raise ValueError(
                f'{name} must contain the complete {self.action_chunk}-frame '
                f'chunk; got {chunk.shape[0]} frames.')
        if not np.all(np.isfinite(chunk)):
            raise ValueError(f'{name} contains non-finite values.')
        self._arm_action_indices(chunk.shape[1])
        return chunk.copy()

    def _blend_overlapping_plan(self, old_left_over,
                                new_actions) -> tuple[np.ndarray, dict]:
        """Build a plan at inference-start time from old and new chunks."""
        new_chunk = self._validate_action_chunk(new_actions, 'new actions')
        if old_left_over is None:
            return new_chunk, {
                'overlap_frames': 0,
                'max_arm_disagreement_rad': 0.0,
            }

        old_chunk = np.asarray(old_left_over)
        if old_chunk.ndim != 2 or old_chunk.shape[1] != new_chunk.shape[1]:
            raise ValueError('Old/new overlap action shapes are incompatible: '
                             f'{old_chunk.shape} and {new_chunk.shape}.')
        if not np.all(np.isfinite(old_chunk)):
            raise ValueError('Old overlap actions contain non-finite values.')

        overlap = min(old_chunk.shape[0], new_chunk.shape[0])
        if overlap <= 0:
            return new_chunk, {
                'overlap_frames': 0,
                'max_arm_disagreement_rad': 0.0,
            }

        if overlap == 1:
            weights = np.asarray(
                [(self.blend_start_weight + self.blend_end_weight) / 2.0],
                dtype=np.float64)
        else:
            weights = np.linspace(
                self.blend_start_weight,
                self.blend_end_weight,
                overlap,
                dtype=np.float64,
            )

        plan = new_chunk.copy()
        choose_new = weights >= 0.5
        plan[:overlap] = np.where(choose_new[:, None], new_chunk[:overlap],
                                  old_chunk[:overlap])

        arm_indices = self._arm_action_indices(new_chunk.shape[1])
        old_arms = old_chunk[:overlap, arm_indices]
        new_arms = new_chunk[:overlap, arm_indices]
        plan[:overlap, arm_indices] = ((1.0 - weights[:, None]) * old_arms +
                                       weights[:, None] * new_arms)
        max_disagreement = float(np.max(np.abs(new_arms - old_arms)))

        return plan.astype(
            new_chunk.dtype, copy=False), {
                'overlap_frames': overlap,
                'max_arm_disagreement_rad': max_disagreement,
            }

    def _predict_processed_chunk(self, instruction: str) -> np.ndarray:
        self._action_ctx = SimpleNamespace(instruction=instruction)
        inputs = self._preprocess(instruction)
        raw_action = self._predict_action(inputs)
        actions = self._postprocess_actions(raw_action)
        return self._validate_action_chunk(actions, 'predicted actions')

    def _execute_waypoint(self, action: np.ndarray) -> None:
        if self.disable_puppet_arm or self.dry_run:
            return
        if not hasattr(self.ros_operator, 'execute_waypoint'):
            raise TypeError('Overlap execution requires an operator with '
                            'execute_waypoint(), such as Tron2EnvOperator.')

        parts = self._action_parts(np.asarray(action)[None])
        head = None if parts['head'] is None else parts['head'][0]
        self.ros_operator.execute_waypoint(
            left_arm=parts['left_arm'][0],
            right_arm=parts['right_arm'][0],
            left_gripper=parts['left_gripper'][0],
            right_gripper=parts['right_gripper'][0],
            head=head,
            dt=self.dt,
        )

    def _consume_actions(self, action_queue: ActionQueue,
                         producer_done: threading.Event,
                         shutdown_event: threading.Event,
                         errors: list[BaseException]) -> None:
        overwatch = initialize_overwatch(__name__)
        next_deadline = time.perf_counter()
        last_warning_at = float('-inf')
        last_warning_key = None
        suppressed_warnings = 0
        try:
            while not shutdown_event.is_set():
                action = action_queue.get()
                if action is None:
                    if producer_done.is_set():
                        return
                    hold_reason = (
                        'action queue is empty; keeping the previous ServoJ '
                        'target unchanged')
                    hold_key = 'queue-empty'
                else:
                    try:
                        self._execute_waypoint(action)
                        hold_reason = None
                        hold_key = None
                    except ValueError as exc:
                        hold_reason = (
                            f'action waypoint rejected ({exc}); keeping the '
                            'previous ServoJ target unchanged')
                        hold_key = 'waypoint-rejected'

                if hold_reason is None:
                    last_warning_key = None
                    suppressed_warnings = 0
                else:
                    now = time.perf_counter()
                    if (hold_key != last_warning_key or now - last_warning_at
                            >= self.hold_warning_interval_s):
                        suppressed = ('' if suppressed_warnings == 0 else
                                      f'; suppressed={suppressed_warnings}')
                        overwatch.warning('[Hold] %s%s', hold_reason,
                                          suppressed)
                        last_warning_at = now
                        last_warning_key = hold_key
                        suppressed_warnings = 0
                    else:
                        suppressed_warnings += 1

                next_deadline += self.dt
                remaining = next_deadline - time.perf_counter()
                if remaining > 0:
                    shutdown_event.wait(remaining)
                else:
                    # Never emit catch-up bursts after a slow control call.
                    next_deadline = time.perf_counter()
        except BaseException as exc:
            errors.append(exc)
            shutdown_event.set()

    @staticmethod
    def _raise_consumer_error(errors: list[BaseException]) -> None:
        if errors:
            raise RuntimeError('TRON2 overlap action consumer failed.') from \
                errors[0]

    def _wait_for_inference_trigger(self, action_queue: ActionQueue,
                                    trigger_queue_size: int,
                                    shutdown_event: threading.Event,
                                    errors: list[BaseException]) -> None:
        while action_queue.qsize() > trigger_queue_size:
            self._raise_consumer_error(errors)
            if shutdown_event.wait(self.queue_poll_interval_s):
                self._raise_consumer_error(errors)
                raise RuntimeError('Overlap action consumer stopped.')

    def _run_episode(self, default_instruction: str):
        """Run one task using a continuous queue and overlapping inference."""
        overwatch = initialize_overwatch(__name__)
        instructions = self._get_user_task_instruction(default_instruction)
        if not instructions:
            return

        action_queue = ActionQueue(rtc_enabled=True)
        producer_done = threading.Event()
        shutdown_event = threading.Event()
        consumer_errors: list[BaseException] = []
        consumer_thread = None

        try:
            first_actions = self._predict_processed_chunk(instructions[0])
            initial_plan = (
                first_actions if len(instructions) > 1 else
                first_actions[:self.execute_horizon])
            action_queue.merge(initial_plan, initial_plan, real_delay=0)
            self._prev_ctx = self._action_ctx

            consumer_thread = threading.Thread(
                target=self._consume_actions,
                args=(action_queue, producer_done, shutdown_event,
                      consumer_errors),
                daemon=True,
                name='Tron2-overlap-action-consumer',
            )
            consumer_thread.start()

            trigger_queue_size = self.action_chunk - self.execute_horizon
            overwatch.info(
                '[Overlap] Started: chunk=%d, execution_horizon=%d, '
                'trigger_queue=%d, blend=%.2f->%.2f',
                self.action_chunk,
                self.execute_horizon,
                trigger_queue_size,
                self.blend_start_weight,
                self.blend_end_weight,
            )

            for chunk_index, instruction in enumerate(instructions[1:], 2):
                self._wait_for_inference_trigger(
                    action_queue,
                    trigger_queue_size,
                    shutdown_event,
                    consumer_errors,
                )

                # Acquire the fresh observation first, then atomically snapshot
                # the executable plan at the corresponding action time.
                self._action_ctx = SimpleNamespace(instruction=instruction)
                inputs = self._preprocess(instruction)
                self._raise_consumer_error(consumer_errors)
                action_index_before, old_left_over, queue_size_before = (
                    action_queue.snapshot_left_over())

                inference_start = time.perf_counter()
                raw_action = self._predict_action(inputs)
                new_actions = self._validate_action_chunk(
                    self._postprocess_actions(raw_action),
                    'predicted actions',
                )
                inference_elapsed = time.perf_counter() - inference_start
                self._raise_consumer_error(consumer_errors)

                plan, diagnostics = self._blend_overlapping_plan(
                    old_left_over, new_actions)
                measured_delay = min(
                    math.ceil(inference_elapsed / self.dt),
                    self.action_chunk,
                )
                queue_plan = (
                    plan if chunk_index < len(instructions) else
                    plan[:self.execute_horizon])
                used_delay = action_queue.merge(
                    queue_plan,
                    queue_plan,
                    real_delay=measured_delay,
                    action_index_before_inference=action_index_before,
                )
                self._prev_ctx = self._action_ctx

                overwatch.info(
                    '[Overlap %d/%d] e2e=%.1fms measured=%d used=%d '
                    'queue=%d->%d overlap=%d max_arm_diff=%.4frad',
                    chunk_index,
                    len(instructions),
                    inference_elapsed * 1000.0,
                    measured_delay,
                    used_delay,
                    queue_size_before,
                    action_queue.qsize(),
                    diagnostics['overlap_frames'],
                    diagnostics['max_arm_disagreement_rad'],
                )

            producer_done.set()
            while consumer_thread.is_alive():
                consumer_thread.join(timeout=0.1)
                self._raise_consumer_error(consumer_errors)
            self._raise_consumer_error(consumer_errors)
        finally:
            producer_done.set()
            shutdown_event.set()
            if consumer_thread is not None and consumer_thread.is_alive():
                consumer_thread.join(timeout=2.0)
                if consumer_thread.is_alive():
                    overwatch.error(
                        'Overlap action consumer did not stop within 2 seconds.'
                    )
