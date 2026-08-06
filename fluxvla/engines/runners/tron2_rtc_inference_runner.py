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
"""FluxVLA policy adapter for the public TRON2 RTC deployment loop.

The producer/consumer scheduling below is ported from
``tron2_openpi/examples/tron2/pi_client_rtc.py`` at commit
``fb1ca651bc0de96aef6a4d2d1445e98cb9a84ac5``. The policy request itself is the
only intentional substitution: it calls the local FluxVLA model and keeps the
checkpoint's 32-dimensional normalized actions for RTC conditioning.
"""

from __future__ import annotations

import math
from collections import deque
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from ..utils import initialize_overwatch
from ..utils.root import RUNNERS
from .tron2_inference_runner import Tron2InferenceRunner

_PROCESSED_ARM_INDICES = tuple(range(7)) + tuple(range(8, 15))
_PROCESSED_GRIPPER_INDICES = (7, 15)
_INFERENCE_DELAY_HISTORY_SIZE = 10


class _RTCActionPostProcessor:
    """Optional executable-action smoothing from the public TRON2 RTC client."""

    def __init__(self, config: dict | None = None):
        config = dict(config or {})
        self.enabled = bool(config.get('enabled', False))
        self.boundary_blend_frames = int(
            config.get('boundary_blend_frames', 0))
        self.boundary_blend_curve = str(
            config.get('boundary_blend_curve', 'smoothstep'))
        self.boundary_blend_scope = str(
            config.get('boundary_blend_scope', 'arm'))
        self.ema_alpha = float(config.get('ema_alpha', 1.0))
        self.ema_frames = int(config.get('ema_frames', 0))
        self.ema_scope = str(config.get('ema_scope', 'arm'))

        if self.boundary_blend_frames < 0:
            raise ValueError('rtc_config.action_postprocess.'
                             'boundary_blend_frames must be >= 0.')
        if self.boundary_blend_curve not in {'linear', 'smoothstep'}:
            raise ValueError('rtc_config.action_postprocess.'
                             'boundary_blend_curve must be linear or '
                             'smoothstep.')
        for name, scope in (
            ('boundary_blend_scope', self.boundary_blend_scope),
            ('ema_scope', self.ema_scope),
        ):
            if scope not in {'arm', 'gripper', 'all'}:
                raise ValueError('rtc_config.action_postprocess.' + name +
                                 ' must be arm, gripper, or all.')
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError('rtc_config.action_postprocess.ema_alpha must '
                             'satisfy 0 < alpha <= 1.')
        if self.ema_frames < 0:
            raise ValueError('rtc_config.action_postprocess.ema_frames must '
                             'be >= 0.')

    @property
    def active(self) -> bool:
        return self.enabled and (self.boundary_blend_frames > 0
                                 or self.ema_alpha < 1.0)

    def describe(self) -> str:
        if not self.active:
            return 'off'
        modes = []
        if self.boundary_blend_frames > 0:
            modes.append(
                f'blend={self.boundary_blend_frames}:'
                f'{self.boundary_blend_scope}:{self.boundary_blend_curve}')
        if self.ema_alpha < 1.0:
            modes.append(f'ema={self.ema_alpha:.2f}:{self.ema_scope}:'
                         f'{self.ema_frames or "all"}')
        return '+'.join(modes)

    @staticmethod
    def _scope_indices(scope: str, action_dim: int) -> np.ndarray:
        if scope == 'all':
            return np.arange(action_dim, dtype=np.int64)
        indices = (
            _PROCESSED_ARM_INDICES
            if scope == 'arm' else _PROCESSED_GRIPPER_INDICES)
        return np.asarray([index for index in indices if index < action_dim],
                          dtype=np.int64)

    def apply(self, processed_actions: np.ndarray,
              old_processed_leftover: np.ndarray | None,
              merge_delay: int) -> np.ndarray:
        if not self.active:
            return processed_actions

        actions = np.asarray(processed_actions).copy()
        start = max(0, min(int(merge_delay), len(actions)))
        self._apply_boundary_blend(actions, old_processed_leftover, start)
        self._apply_ema(actions, old_processed_leftover, start)
        return actions

    def _apply_boundary_blend(self, actions: np.ndarray,
                              old_processed_leftover: np.ndarray | None,
                              start: int) -> None:
        if (self.boundary_blend_frames <= 0 or old_processed_leftover is None
                or start >= len(actions)):
            return
        count = min(self.boundary_blend_frames,
                    len(actions) - start, len(old_processed_leftover))
        if count <= 0:
            return
        action_dim = min(actions.shape[1], old_processed_leftover.shape[1])
        dims = self._scope_indices(self.boundary_blend_scope, action_dim)
        for offset in range(count):
            alpha = (offset + 1) / (count + 1)
            if self.boundary_blend_curve == 'smoothstep':
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            actions[start + offset, dims] = (
                (1.0 - alpha) * old_processed_leftover[offset, dims] +
                alpha * actions[start + offset, dims])

    def _apply_ema(self, actions: np.ndarray,
                   old_processed_leftover: np.ndarray | None,
                   start: int) -> None:
        if self.ema_alpha >= 1.0 or start >= len(actions):
            return
        count = len(actions) - start
        if self.ema_frames > 0:
            count = min(count, self.ema_frames)
        if count <= 0:
            return
        action_dim = actions.shape[1]
        if old_processed_leftover is not None and len(
                old_processed_leftover) > 0:
            action_dim = min(action_dim, old_processed_leftover.shape[1])
        dims = self._scope_indices(self.ema_scope, action_dim)
        if old_processed_leftover is not None and len(
                old_processed_leftover) > 0:
            previous = old_processed_leftover[0, dims].astype(
                np.float64, copy=True)
        else:
            previous = actions[start, dims].astype(np.float64, copy=True)
        for offset in range(count):
            current = actions[start + offset, dims].astype(
                np.float64, copy=False)
            filtered = (
                self.ema_alpha * current + (1.0 - self.ema_alpha) * previous)
            actions[start + offset, dims] = filtered
            previous = filtered


@RUNNERS.register_module()
class Tron2RTCInferenceRunner(Tron2InferenceRunner):
    """Run FluxVLA through the upstream TRON2 RTC producer/consumer loop."""

    def __init__(self,
                 rtc_config: dict = None,
                 fixed_prefix_execution=None,
                 fixed_prefix_late_tolerance_steps=None,
                 *args,
                 **kwargs):
        if fixed_prefix_execution is not None:
            raise ValueError('fixed_prefix_execution was removed. TRON2 queue '
                             'RTC replaces actions using the actual consumer '
                             'index; remove this option.')
        if fixed_prefix_late_tolerance_steps is not None:
            raise ValueError(
                'fixed_prefix_late_tolerance_steps was removed. '
                'TRON2 queue RTC holds on underflow and merges by '
                'the actual consumer index.')

        self.rtc_config = dict(rtc_config or {})
        if self.rtc_config.get('enabled') is not True:
            raise ValueError('Tron2RTCInferenceRunner requires '
                             'rtc_config.enabled=True.')
        if self.rtc_config.get('method', 'prefix') != 'prefix':
            raise ValueError('The FluxVLA RTC checkpoint requires '
                             "rtc_config.method='prefix'.")
        if 'delay' not in self.rtc_config:
            raise ValueError('rtc_config.delay is required as the initial '
                             'TRON2 RTC delay in action frames.')
        if 'execution_horizon' not in self.rtc_config:
            raise ValueError('rtc_config.execution_horizon is required for '
                             'TRON2 queue-triggered inference.')

        self.initial_delay = self._non_negative_int('rtc_config.delay',
                                                    self.rtc_config['delay'])
        self.execution_horizon = self._non_negative_int(
            'rtc_config.execution_horizon',
            self.rtc_config['execution_horizon'])
        self.trigger_poll_interval_s = self._positive_float(
            'rtc_config.trigger_poll_interval_s',
            self.rtc_config.get('trigger_poll_interval_s', 0.005))
        self.observation_timeout_budget_s = self._positive_float(
            'rtc_config.observation_timeout_budget_s',
            self.rtc_config.get('observation_timeout_budget_s', 5.0))
        self.recovery_blend_frames = self._non_negative_int(
            'rtc_config.recovery_blend_frames',
            self.rtc_config.get('recovery_blend_frames', 6))
        configured_prefix_len = self.rtc_config.get('prefix_len')
        if configured_prefix_len is not None:
            configured_prefix_len = self._non_negative_int(
                'rtc_config.prefix_len', configured_prefix_len)
        self.fixed_model_prefix = configured_prefix_len
        prefix_action_dim = self.rtc_config.get('prefix_action_dim')
        if (prefix_action_dim is not None
                and (isinstance(prefix_action_dim, bool)
                     or not isinstance(prefix_action_dim, int)
                     or prefix_action_dim <= 0)):
            raise ValueError('rtc_config.prefix_action_dim must be a positive '
                             f'integer or None; got {prefix_action_dim!r}.')
        self.prefix_action_dim = prefix_action_dim
        prefix_head_from_observation = self.rtc_config.get(
            'prefix_head_from_observation', False)
        if not isinstance(prefix_head_from_observation, bool):
            raise ValueError(
                'rtc_config.prefix_head_from_observation must be bool; got '
                f'{prefix_head_from_observation!r}.')
        self.prefix_head_from_observation = prefix_head_from_observation
        self._reported_prefix_head = False
        self._action_postprocessor = _RTCActionPostProcessor(
            self.rtc_config.get('action_postprocess'))
        self._warned_delay_outside_training_range = False

        super().__init__(*args, **kwargs)
        if self._use_remote:
            raise ValueError('Tron2RTCInferenceRunner keeps the FluxVLA model '
                             'and RTC state in one process; remote_inference '
                             'must be None.')
        if self.async_execution:
            raise ValueError('TRON2 queue RTC owns its persistent 30 Hz '
                             'consumer. Set inference.async_execution=False.')
        if not 0 <= self.execution_horizon <= self.action_chunk:
            raise ValueError('rtc_config.execution_horizon must satisfy '
                             f'0 <= s <= H; got s={self.execution_horizon}, '
                             f'H={self.action_chunk}.')
        if self.initial_delay >= self.action_chunk:
            raise ValueError(
                'rtc_config.delay must be smaller than the action horizon; '
                f'got delay={self.initial_delay}, H={self.action_chunk}.')
        if (self.fixed_model_prefix is not None
                and self.fixed_model_prefix >= self.action_chunk):
            raise ValueError('rtc_config.prefix_len must be smaller than the '
                             f'action horizon; got prefix_len='
                             f'{self.fixed_model_prefix}, H='
                             f'{self.action_chunk}.')
        self.trigger_queue_size = self.action_chunk - self.execution_horizon

        model_horizon = getattr(self.vla, 'n_action_steps', None)
        if model_horizon is not None and self.action_chunk != model_horizon:
            raise ValueError('RTC action_chunk must equal the model action '
                             f'horizon; got action_chunk={self.action_chunk}, '
                             f'model_horizon={model_horizon}.')
        model_action_dim = getattr(self.vla, 'max_action_dim', None)
        feedback_width = (None if self.prefix_action_dim is None else
                          self.prefix_action_dim +
                          (2 if self.prefix_head_from_observation else 0))
        if (feedback_width is not None and model_action_dim is not None
                and feedback_width > model_action_dim):
            raise ValueError(
                'RTC prefix feedback must not exceed the model action width; '
                f'got {feedback_width} > '
                f'{model_action_dim}.')
        if self.prefix_head_from_observation and self.prefix_action_dim is None:
            raise ValueError(
                'rtc_config.prefix_head_from_observation=True requires '
                'rtc_config.prefix_action_dim.')
        supported_prefixes = (getattr(self.vla, 'rtc_training_config', {})
                              or {}).get('delay_values')
        if (self.fixed_model_prefix is not None and supported_prefixes
                and self.fixed_model_prefix not in supported_prefixes):
            raise ValueError(
                f'rtc_config.prefix_len={self.fixed_model_prefix} was not '
                f'used by this checkpoint; supported values are '
                f'{supported_prefixes}.')
        if self.action_layout != 'tron2_16':
            raise ValueError('The public TRON2 ActionQueue deployment layout '
                             'is tron2_16; got '
                             f'action_layout={self.action_layout!r}.')

    @staticmethod
    def _non_negative_int(name: str, value) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{name} must be a non-negative integer; got '
                             f'{value!r}.')
        return value

    @staticmethod
    def _positive_float(name: str, value) -> float:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be finite and > 0; got {value!r}.')
        return float(value)

    @staticmethod
    def _rtc_runtime_types():
        from tron2_env.rtc import ActionQueue, LatencyTracker
        return ActionQueue, LatencyTracker

    @staticmethod
    def _p95_int(values) -> int:
        if not values:
            return 0
        ordered = sorted(int(value) for value in values)
        index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        return ordered[index]

    def run_setup(self):
        super().run_setup()
        training_config = getattr(self.vla, 'rtc_training_config', {}) or {}
        supported_prefixes = training_config.get('delay_values')
        initialize_overwatch(__name__).info(
            'TRON2 queue RTC enabled: H=%d, execution_horizon=%d, '
            'trigger_queue_size=%d, initial_delay=%d, '
            'delay_history=%d, model_prefixes=%s, fixed_model_prefix=%s, '
            'prefix_action_dim=%s, measured_head=%s, postprocess=%s.',
            self.action_chunk, self.execution_horizon, self.trigger_queue_size,
            self.initial_delay, _INFERENCE_DELAY_HISTORY_SIZE,
            supported_prefixes, self.fixed_model_prefix,
            self.prefix_action_dim, self.prefix_head_from_observation,
            self._action_postprocessor.describe())

    def _action_record_session_metadata(self) -> dict:
        metadata = super()._action_record_session_metadata()
        metadata.update(
            rtc_scheduler='tron2_action_queue',
            rtc_execution_horizon=self.execution_horizon,
            rtc_trigger_queue_size=self.trigger_queue_size,
            rtc_initial_delay=self.initial_delay,
            rtc_delay_estimation='measured_ceil_recent_p95',
            rtc_supported_prefixes=(getattr(self.vla, 'rtc_training_config',
                                            {}) or {}).get('delay_values'),
            rtc_fixed_model_prefix=self.fixed_model_prefix,
            rtc_prefix_action_dim=self.prefix_action_dim,
            rtc_prefix_head_from_observation=(
                self.prefix_head_from_observation),
            rtc_action_postprocess=self._action_postprocessor.describe(),
        )
        metadata.pop('fixed_prefix_execution', None)
        metadata.pop('fixed_prefix_late_tolerance_steps', None)
        return metadata

    def _move_to_prepare_pose(self):
        """Run the public Tron2Env bring-up path for the idle ``r`` key."""
        reset_native_env = getattr(self.ros_operator, 'reset_native_env', None)
        if not callable(reset_native_env):
            return super()._move_to_prepare_pose()
        if self.dry_run:
            print('[Tron2RTCInferenceRunner] dry_run=True, skip native reset.')
            return
        reset_native_env()
        self.last_actions = None

    def _predict_action(self, inputs):
        ctx = self._action_ctx
        ctx.inference_start = time.time()
        ctx.inference_start_monotonic = time.monotonic()
        inference_started = time.perf_counter()

        prev_actions = getattr(ctx, 'rtc_prev_actions', None)
        prefix_len = int(getattr(ctx, 'rtc_prefix_len', 0))
        if prev_actions is not None and prefix_len > 0:
            inputs['prev_actions'] = torch.from_numpy(prev_actions[None]).to(
                device=inputs['states'].device, dtype=inputs['states'].dtype)
            inputs['prefix_len'] = prefix_len
            inputs['rtc_config'] = self.rtc_config

        raw_action = self.vla.predict_action(**inputs)
        ctx.inference_elapsed = time.perf_counter() - inference_started
        raw_numpy = raw_action.detach().cpu().numpy()
        if raw_numpy.ndim == 3 and raw_numpy.shape[0] == 1:
            raw_numpy = raw_numpy[0]
        if raw_numpy.ndim != 2:
            raise ValueError('FluxVLA RTC actions must have shape (H, D); got '
                             f'{raw_numpy.shape}.')
        ctx.raw_actions = raw_numpy[:self.action_chunk].copy()
        return raw_action

    def _prepare_rtc_inputs(
            self, previous_leftover: np.ndarray | None,
            inference_delay: int) -> tuple[np.ndarray | None, int]:
        """Quantize upstream delay to a prefix supported by this checkpoint."""
        if previous_leftover is None or len(previous_leftover) == 0:
            return None, 0

        requested_delay = min(
            max(0, int(inference_delay)), self.action_chunk - 1)
        training_config = getattr(self.vla, 'rtc_training_config', {}) or {}
        configured_values = training_config.get('delay_values')
        if self.fixed_model_prefix is not None:
            supported = (
                sorted({
                    int(value)
                    for value in configured_values
                    if 0 <= int(value) < self.action_chunk
                }) if configured_values else None)
            prefix_len = self.fixed_model_prefix
        elif configured_values:
            supported = sorted({
                int(value)
                for value in configured_values
                if 0 <= int(value) < self.action_chunk
            })
            if not supported:
                raise ValueError(
                    'rtc_training_config.delay_values does not contain a '
                    'valid prefix for this action horizon.')
            prefix_len = next(
                (value for value in supported if value >= requested_delay),
                supported[-1])
        else:
            supported = None
            prefix_len = requested_delay

        feedback_actions = np.asarray(previous_leftover)
        if self.prefix_action_dim is not None:
            if feedback_actions.shape[1] < self.prefix_action_dim:
                raise ValueError('RTC previous-action width is smaller than '
                                 'rtc_config.prefix_action_dim: '
                                 f'{feedback_actions.shape[1]} < '
                                 f'{self.prefix_action_dim}.')
            feedback_actions = feedback_actions[:, :self.prefix_action_dim]
        if self.prefix_head_from_observation:
            normalized_head = self._normalize_measured_head_for_rtc()
            head_prefix = np.broadcast_to(
                normalized_head,
                (len(feedback_actions), len(normalized_head)),
            )
            feedback_actions = np.concatenate([feedback_actions, head_prefix],
                                              axis=1)

        padded = np.zeros((self.action_chunk, feedback_actions.shape[1]),
                          dtype=feedback_actions.dtype)
        copy_count = min(len(previous_leftover), self.action_chunk)
        padded[:copy_count] = feedback_actions[:copy_count]

        max_delay = training_config.get('max_delay')
        outside_training_range = (
            (max_delay is not None and requested_delay >= max_delay)
            or (supported is not None and requested_delay > supported[-1]))
        if (outside_training_range
                and not self._warned_delay_outside_training_range):
            initialize_overwatch(__name__).warning(
                'TRON2 RTC requested delay=%d exceeds this checkpoint\'s '
                'supported prefixes %s; clamping model prefix to %d. Queue '
                'replacement still uses the upstream actual consumer index.',
                requested_delay, supported, prefix_len)
            self._warned_delay_outside_training_range = True
        return padded, prefix_len

    def _normalize_measured_head_for_rtc(self) -> np.ndarray:
        """Normalize the locked measured head like training action dims 16:18."""
        head = np.asarray(
            getattr(self, '_latest_head_position', None), dtype=np.float32)
        if head.shape != (2, ) or not np.all(np.isfinite(head)):
            raise ValueError(
                'RTC measured-head prefix requires a finite two-dimensional '
                f'head observation; got shape={head.shape}, values={head}.')

        normalizer = self.denormalize_action
        if getattr(normalizer, 'norm_type', None) != 'min_max':
            raise ValueError(
                'RTC measured-head prefix currently requires min_max action '
                'normalization, matching the Tron2 checkpoint.')
        all_stats = getattr(normalizer, 'norm_stats', None)
        try:
            action_stats = all_stats[self.task_suite_name]['action']
            low = np.asarray(action_stats['min'], dtype=np.float32)[16:18]
            high = np.asarray(action_stats['max'], dtype=np.float32)[16:18]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                'RTC measured-head prefix could not read action dimensions '
                '16:18 from checkpoint normalization statistics.') from exc
        if low.shape != (2, ) or high.shape != (2, ):
            raise ValueError(
                'RTC measured-head action statistics must contain at least '
                '18 dimensions.')

        normalized = np.clip(
            2.0 * (head - low) / (high - low + 1e-6) - 1.0,
            -1.0,
            1.0,
        ).astype(np.float32)
        if not self._reported_prefix_head:
            initialize_overwatch(__name__).info(
                'RTC prefix head matches training action normalization: '
                'measured=%s normalized=%s.', head.tolist(),
                normalized.tolist())
            self._reported_prefix_head = True
        return normalized

    def _preprocess_with_retry(self, instruction: str,
                               stop_requested: threading.Event):
        started = time.monotonic()
        while not stop_requested.is_set():
            try:
                return self._preprocess(instruction)
            except TimeoutError as exc:
                elapsed = time.monotonic() - started
                if elapsed >= self.observation_timeout_budget_s:
                    raise TimeoutError(
                        'TRON2 observation remained unavailable for '
                        f'{elapsed:.1f}s, exceeding the configured '
                        f'{self.observation_timeout_budget_s:.1f}s budget.'
                    ) from exc
                initialize_overwatch(__name__).warning(
                    'Observation timeout %.1f/%.1fs; holding the latest '
                    'ServoJ target while waiting for a fresh observation: %s',
                    elapsed, self.observation_timeout_budget_s, exc)
        return None

    def _validate_action_chunks(self,
                                processed_actions: np.ndarray) -> np.ndarray:
        """Validate one model/postprocess pair before queue replacement.

        PI0.5 deliberately keeps a 32-dimensional normalized model-space
        action in ``ActionQueue.original_queue`` for the next RTC prefix, while
        denormalization removes padding and produces the 16-dimensional TRON2
        command. Their time dimensions must match; their feature dimensions
        normally do not.
        """
        raw_actions = np.asarray(self._action_ctx.raw_actions)
        processed_actions = np.asarray(processed_actions)
        for name, actions in (('raw', raw_actions), ('postprocessed',
                                                     processed_actions)):
            if actions.ndim != 2:
                raise ValueError(
                    f'RTC {name} actions must have shape (H, D); got '
                    f'{actions.shape}.')
            if len(actions) != self.action_chunk:
                raise ValueError(
                    f'RTC {name} chunk is incomplete: {len(actions)} != '
                    f'{self.action_chunk}.')
            if not np.all(np.isfinite(actions)):
                raise ValueError(f'RTC {name} chunk contains non-finite '
                                 'values.')
        return processed_actions

    def _infer_chunk(self, instruction: str, previous_leftover,
                     inference_delay: int, stop_requested: threading.Event):
        inputs = self._preprocess_with_retry(instruction, stop_requested)
        if inputs is None or stop_requested.is_set():
            return None

        previous_actions, prefix_len = self._prepare_rtc_inputs(
            previous_leftover, inference_delay)
        self._action_ctx = SimpleNamespace(
            instruction=instruction,
            rtc_prev_actions=previous_actions,
            rtc_prefix_len=prefix_len,
        )
        with torch.autocast(
                'cuda',
                dtype=self.mixed_precision_dtype,
                enabled=self.enable_mixed_precision):
            raw_action = self._predict_action(inputs)
        if stop_requested.is_set():
            return None
        processed_actions = self._postprocess_actions(raw_action)
        processed_actions = self._validate_action_chunks(processed_actions)
        return self._action_ctx.raw_actions, processed_actions

    def _execute_rtc_waypoint(self, action: np.ndarray, source_index: int,
                              queue_size: int) -> None:
        if self.disable_puppet_arm or self.dry_run:
            return
        parts = self._action_parts(np.asarray(action)[None])
        self._trajectory_submission_index += 1
        metadata = {
            'trajectory_id': self._trajectory_submission_index,
            'task_id': self._active_task_id,
            'instruction': getattr(self._action_ctx, 'instruction', None),
            'rtc_enabled': True,
            'rtc_scheduler': 'tron2_action_queue',
            'rtc_prefix_len': getattr(self._action_ctx, 'rtc_prefix_len',
                                      None),
            'source_action_index': int(source_index),
            'queue_size_after_get': int(queue_size),
        }
        self.ros_operator.execute_waypoint(
            left_arm=parts['left_arm'][0],
            right_arm=parts['right_arm'][0],
            left_gripper=float(parts['left_gripper'][0]),
            right_gripper=float(parts['right_gripper'][0]),
            head=None if parts['head'] is None else parts['head'][0],
            dt=self.dt,
            action_metadata=metadata,
            trajectory_index=source_index,
        )

    def _consume_actions(self, action_queue, shutdown_event: threading.Event,
                         errors: list) -> None:
        """Port of ``tron2_openpi.pi_client_rtc.control_consumer``."""
        overwatch = initialize_overwatch(__name__)
        last_action = None
        step_count = 0
        stall_count = 0
        recovery_blend_total = self.recovery_blend_frames
        recovery_blend_remaining = 0
        recovery_hold_action = None
        try:
            while not shutdown_event.is_set():
                started = time.perf_counter()
                source_index = action_queue.get_action_index()
                action = action_queue.get()
                if action is None:
                    recovery_blend_remaining = 0
                    recovery_hold_action = None
                    stall_count += 1
                    if stall_count % 50 == 1:
                        overwatch.warning(
                            '[CONSUMER] Queue empty (stalled %d times), '
                            'step=%d.', stall_count, step_count)
                else:
                    if stall_count > 0 and last_action is not None:
                        recovery_blend_remaining = recovery_blend_total
                        recovery_hold_action = last_action.copy()
                        if recovery_blend_remaining > 0:
                            overwatch.warning(
                                '[CONSUMER] Recovered after %d empty ticks; '
                                'blending next %d frames.', stall_count,
                                recovery_blend_total)
                    if (recovery_blend_remaining > 0
                            and recovery_hold_action is not None):
                        blend_index = (
                            recovery_blend_total - recovery_blend_remaining +
                            1)
                        alpha = min(1.0,
                                    blend_index / max(1, recovery_blend_total))
                        action = ((1.0 - alpha) * recovery_hold_action +
                                  alpha * action).astype(
                                      action.dtype, copy=False)
                        recovery_blend_remaining -= 1
                        if recovery_blend_remaining == 0:
                            recovery_hold_action = None

                    if last_action is not None:
                        delta = np.abs(
                            action[list(_PROCESSED_ARM_INDICES)] -
                            last_action[list(_PROCESSED_ARM_INDICES)])
                        max_joint = int(np.argmax(delta))
                        if float(delta[max_joint]) >= 0.5:
                            overwatch.warning(
                                '[CONSUMER] Large action jump: joint %d '
                                'diff=%.4f at step %d.', max_joint,
                                float(delta[max_joint]), step_count)
                    self._execute_rtc_waypoint(action, source_index,
                                               action_queue.qsize())
                    last_action = action.copy()
                    step_count += 1
                    stall_count = 0

                elapsed = time.perf_counter() - started
                sleep_time = max(0.0, self.dt - elapsed - 0.001)
                if sleep_time > 0:
                    # Keep the public tron2_openpi consumer pacing exactly:
                    # the policy thread sleeps; MotionController continues its
                    # independent 300 Hz interpolation/publication loop.
                    time.sleep(sleep_time)
        except BaseException as exc:
            errors.append(exc)
            shutdown_event.set()

    @staticmethod
    def _raise_consumer_error(errors: list) -> None:
        if errors:
            raise RuntimeError(
                'TRON2 RTC action consumer failed.') from errors[0]

    def _run_continuous_task(self, instruction: str,
                             stop_requested: threading.Event):
        """Port the upstream RTC producer, substituting local FluxVLA infer."""
        ActionQueue, LatencyTracker = self._rtc_runtime_types()
        action_queue = ActionQueue(rtc_enabled=True)
        latency_stats = LatencyTracker()
        inference_delay_buffer = deque(maxlen=_INFERENCE_DELAY_HISTORY_SIZE)
        consumer_errors = []
        consumer_thread = None
        infer_count = 0
        merge_delay_cap = self.action_chunk - 1
        overwatch = initialize_overwatch(__name__)
        self._prev_ctx = None

        try:
            overwatch.info('[WARMUP] Starting TRON2-compatible RTC warmup.')
            initial = self._infer_chunk(instruction, None, 0, stop_requested)
            if initial is None:
                return
            initial_raw, initial_processed = initial
            dummy_leftover = np.zeros_like(initial_raw)
            rtc_warmup = self._infer_chunk(instruction, dummy_leftover,
                                           self.initial_delay, stop_requested)
            if rtc_warmup is None:
                return
            overwatch.info('[WARMUP] Complete: raw=%s, processed=%s.',
                           initial_raw.shape, initial_processed.shape)
            action_queue.merge(initial_raw, initial_processed, real_delay=0)

            begin_stream = getattr(self.ros_operator, 'begin_waypoint_stream',
                                   None)
            if callable(begin_stream) and not self.dry_run:
                begin_stream()
            consumer_thread = threading.Thread(
                target=self._consume_actions,
                args=(action_queue, stop_requested, consumer_errors),
                daemon=True,
                name='Tron2-RTC-action-consumer',
            )
            consumer_thread.start()
            overwatch.info(
                '[RTC] Started persistent consumer: H=%d, s=%d, trigger=%d, '
                'initial_delay=%d.', self.action_chunk, self.execution_horizon,
                self.trigger_queue_size, self.initial_delay)

            while not stop_requested.is_set():
                self._raise_consumer_error(consumer_errors)
                if action_queue.qsize() > self.trigger_queue_size:
                    stop_requested.wait(
                        min(self.trigger_poll_interval_s, self.dt * 0.25))
                    continue

                inference_delay_p95 = self._p95_int(inference_delay_buffer)
                inference_delay = (
                    min(inference_delay_p95, merge_delay_cap)
                    if inference_delay_buffer else self.initial_delay)

                inputs = self._preprocess_with_retry(instruction,
                                                     stop_requested)
                if inputs is None or stop_requested.is_set():
                    break
                (action_index_before, previous_leftover,
                 queue_size_before) = action_queue.snapshot_left_over()
                previous_actions, prefix_len = self._prepare_rtc_inputs(
                    previous_leftover, inference_delay)
                self._action_ctx = SimpleNamespace(
                    instruction=instruction,
                    rtc_prev_actions=previous_actions,
                    rtc_prefix_len=prefix_len,
                )

                with torch.autocast(
                        'cuda',
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision):
                    raw_action = self._predict_action(inputs)
                self._raise_consumer_error(consumer_errors)
                if stop_requested.is_set():
                    overwatch.info(
                        'Discarding the RTC result because s was pressed '
                        'before queue merge.')
                    break

                processed_actions = self._postprocess_actions(raw_action)
                processed_actions = self._validate_action_chunks(
                    processed_actions)
                new_latency = self._action_ctx.inference_elapsed
                measured_inference_delay = (
                    0 if queue_size_before <= 0 else min(
                        math.ceil(new_latency / self.dt), merge_delay_cap))
                if infer_count > 0:
                    inference_delay_buffer.append(measured_inference_delay)
                latency_stats.add(new_latency)
                postprocess_delay = max(
                    0,
                    action_queue.get_action_index() - action_index_before)
                old_processed_leftover = (
                    action_queue.get_processed_left_over()
                    if self._action_postprocessor.active else None)
                processed_actions = self._action_postprocessor.apply(
                    processed_actions, old_processed_leftover,
                    postprocess_delay)
                used_delay = action_queue.merge(
                    self._action_ctx.raw_actions,
                    processed_actions,
                    real_delay=measured_inference_delay,
                    action_index_before_inference=action_index_before,
                    extra_delay=0,
                )
                if used_delay > inference_delay:
                    overwatch.warning(
                        'RTC delay underflow: d=%d, used_delay=%d. Delay '
                        'follows recent measured p95.', inference_delay,
                        used_delay)
                self._prev_ctx = self._action_ctx
                overwatch.info(
                    '[PRODUCER #%d] e2e=%.1fms s=%d/H=%d d=%d p95=%d '
                    'prefix=%d meas=%d used=%d left=%d queue=%d->%d post=%s.',
                    infer_count, new_latency * 1000.0, self.execution_horizon,
                    self.action_chunk, inference_delay, inference_delay_p95,
                    prefix_len, measured_inference_delay, used_delay,
                    0 if previous_leftover is None else len(previous_leftover),
                    queue_size_before, action_queue.qsize(),
                    self._action_postprocessor.describe())
                infer_count += 1
        finally:
            stop_requested.set()
            if consumer_thread is not None:
                consumer_thread.join(timeout=2.0)
                if consumer_thread.is_alive():
                    consumer_errors.append(
                        RuntimeError('RTC consumer did not stop after 2s.'))
            action_queue.clear()
            overwatch.info('[RTC] Stopped after %d replacement inferences. %s',
                           infer_count, latency_stats.summary())
        self._raise_consumer_error(consumer_errors)
