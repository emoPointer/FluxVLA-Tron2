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
"""TRON2 client for stateless server-side inference-time RTC guidance.

The robot client keeps paired normalized/raw and executable/processed NumPy
action queues. It sends the exact unconsumed raw prefix with the next fresh
observation; all flow-matching guidance and Torch computation run on the GPU
server. The processed queue continues driving the robot while inference is in
flight, then the returned chunk is aligned using the number of frames actually
consumed by the client action clock.
"""

from __future__ import annotations

import io
import math
import threading
import time
from types import SimpleNamespace

import numpy as np
from tron2_env.rtc import ActionQueue

from ..utils import initialize_overwatch
from ..utils.root import RUNNERS
from .tron2_overlap_inference_runner import Tron2OverlapInferenceRunner


@RUNNERS.register_module()
class Tron2RemoteRTCInferenceRunner(Tron2OverlapInferenceRunner):
    """Run FluxVLA guidance RTC on a remote GPU with a model-free client."""

    _RTC_CONFIG_KEYS = frozenset({
        'enabled',
        'method',
        'prefix_len',
        'latency_margin_frames',
        'decay_frames',
        'schedule',
        'max_guidance_weight',
        'use_vjp',
    })

    def __init__(self, rtc_config: dict = None, *args, **kwargs):
        config = dict(rtc_config or {})
        unknown = set(config) - self._RTC_CONFIG_KEYS
        if unknown:
            raise ValueError(f'Unknown remote RTC config fields: '
                             f'{sorted(unknown)}.')
        if config.get('enabled') is not True:
            raise ValueError('Remote RTC requires rtc_config.enabled=True.')
        if config.get('method') != 'guidance':
            raise ValueError('This runner requires inference-time '
                             "rtc_config.method='guidance'.")

        prefix_len = config.get('prefix_len')
        if (prefix_len is not None
                and (isinstance(prefix_len, bool)
                     or not isinstance(prefix_len, int) or prefix_len <= 0)):
            raise ValueError(
                'rtc_config.prefix_len must be None or a positive '
                f'integer; got {prefix_len!r}.')
        latency_margin_frames = config.get('latency_margin_frames', 2)
        if (isinstance(latency_margin_frames, bool)
                or not isinstance(latency_margin_frames, int)
                or latency_margin_frames < 0):
            raise ValueError('rtc_config.latency_margin_frames must be a '
                             'non-negative integer.')
        decay_frames = config.get('decay_frames', 5)
        if (isinstance(decay_frames, bool)
                or not isinstance(decay_frames, int) or decay_frames < 0):
            raise ValueError('rtc_config.decay_frames must be a non-negative '
                             'integer.')
        schedule = config.get('schedule', 'exp')
        if schedule not in {'exp', 'linear', 'ones', 'zeros'}:
            raise ValueError('rtc_config.schedule must be one of '
                             'exp/linear/ones/zeros.')
        max_guidance_weight = config.get('max_guidance_weight', 5.0)
        if (isinstance(max_guidance_weight, bool)
                or not isinstance(max_guidance_weight, (int, float))
                or not np.isfinite(max_guidance_weight)
                or not 0.0 < max_guidance_weight <= 100.0):
            raise ValueError('rtc_config.max_guidance_weight must be finite '
                             'and within (0, 100].')
        use_vjp = config.get('use_vjp', False)
        if not isinstance(use_vjp, bool):
            raise ValueError('rtc_config.use_vjp must be boolean.')

        self.rtc_prefix_len = prefix_len
        self.rtc_latency_margin_frames = latency_margin_frames
        self.rtc_decay_frames = decay_frames
        self.rtc_schedule = schedule
        self.rtc_max_guidance_weight = float(max_guidance_weight)
        self.rtc_use_vjp = use_vjp
        self._last_e2e_latency_s = None

        super().__init__(*args, **kwargs)
        if not self._use_remote:
            raise ValueError('Tron2RemoteRTCInferenceRunner requires '
                             'remote_inference configuration.')
        if self._serializer != 'msgpack':
            raise ValueError('Remote RTC currently requires '
                             "remote_inference.serializer='msgpack'.")

    def run_setup(self):
        super().run_setup()
        capability = getattr(self, 'deployment_metadata', {}).get('remote_rtc')
        if (not isinstance(capability, dict)
                or capability.get('wire_format') != 'msgpack'
                or capability.get('returns_raw_actions') is not True
                or 'guidance' not in capability.get('methods', [])):
            raise RuntimeError(
                'The GPU server does not advertise stateless remote guidance '
                'RTC. Restart it with the synchronized server code.')

    def _decode_action_array(self, payload: bytes, name: str) -> np.ndarray:
        if not isinstance(payload, (bytes, bytearray)):
            raise RuntimeError(f'Remote response is missing {name}.')
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        return self._validate_action_chunk(array, name)

    def _resolve_prefix_len(self, raw_left_over: np.ndarray) -> int:
        if raw_left_over is None or len(raw_left_over) == 0:
            raise RuntimeError('Remote RTC requires an unconsumed raw action '
                               'prefix, but the action queue is empty.')
        if self.rtc_prefix_len is None:
            if self._last_e2e_latency_s is None:
                raise RuntimeError('Remote RTC has no previous latency sample '
                                   'for dynamic prefix sizing.')
            prefix_len = (
                math.ceil(self._last_e2e_latency_s / self.dt) +
                self.rtc_latency_margin_frames)
        else:
            prefix_len = self.rtc_prefix_len
        return max(1, min(prefix_len, len(raw_left_over), self.action_chunk))

    def _build_rtc_payload(self, raw_left_over: np.ndarray,
                           prefix_len: int) -> dict:
        decay_end = min(self.action_chunk, prefix_len + self.rtc_decay_frames)
        return {
            'prev_actions': np.asarray(raw_left_over, dtype=np.float32),
            'prefix_len': int(prefix_len),
            'config': {
                'enabled': True,
                'method': 'guidance',
                'decay_end': int(decay_end),
                'schedule': self.rtc_schedule,
                'max_guidance_weight': self.rtc_max_guidance_weight,
                'use_vjp': self.rtc_use_vjp,
            },
        }

    def _request_action_pair(
            self,
            inputs: dict,
            rtc: dict = None) -> tuple[np.ndarray, np.ndarray, float]:
        started = time.perf_counter()
        response = self._request_remote_action(inputs, rtc=rtc)
        raw_actions = self._decode_action_array(
            response.get('raw_action_data'), 'raw actions')
        processed_actions = self._decode_action_array(
            response.get('action_data'), 'processed actions')
        elapsed = time.perf_counter() - started
        if raw_actions.shape != processed_actions.shape:
            raise RuntimeError(
                'Remote raw/processed action shapes must match; '
                f'got {raw_actions.shape} and '
                f'{processed_actions.shape}.')
        return raw_actions, processed_actions, elapsed

    def _run_episode(self, default_instruction: str):
        """Run one episode with server-side guidance and paired action queues."""
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
            self._action_ctx = SimpleNamespace(instruction=instructions[0])
            first_inputs = self._preprocess(instructions[0])
            first_raw, first_processed, first_elapsed = (
                self._request_action_pair(first_inputs))
            self._last_e2e_latency_s = first_elapsed
            initial_length = (
                self.action_chunk
                if len(instructions) > 1 else self.execute_horizon)
            action_queue.merge(
                first_raw[:initial_length],
                first_processed[:initial_length],
                real_delay=0,
            )
            self._prev_ctx = self._action_ctx

            consumer_thread = threading.Thread(
                target=self._consume_actions,
                args=(action_queue, producer_done, shutdown_event,
                      consumer_errors),
                daemon=True,
                name='Tron2-remote-RTC-action-consumer',
            )
            consumer_thread.start()

            trigger_queue_size = self.action_chunk - self.execute_horizon
            overwatch.info(
                '[RemoteRTC] Started: chunk=%d execution_horizon=%d '
                'reserve=%d first_e2e=%.1fms method=guidance',
                self.action_chunk,
                self.execute_horizon,
                trigger_queue_size,
                first_elapsed * 1000.0,
            )

            for chunk_index, instruction in enumerate(instructions[1:], 2):
                self._wait_for_inference_trigger(
                    action_queue,
                    trigger_queue_size,
                    shutdown_event,
                    consumer_errors,
                )

                self._action_ctx = SimpleNamespace(instruction=instruction)
                inputs = self._preprocess(instruction)
                self._raise_consumer_error(consumer_errors)
                action_index_before, raw_left_over, queue_size_before = (
                    action_queue.snapshot_left_over())
                prefix_len = self._resolve_prefix_len(raw_left_over)
                rtc_payload = self._build_rtc_payload(raw_left_over,
                                                      prefix_len)

                new_raw, new_processed, inference_elapsed = (
                    self._request_action_pair(inputs, rtc=rtc_payload))
                self._raise_consumer_error(consumer_errors)

                measured_delay = min(
                    math.ceil(inference_elapsed / self.dt),
                    self.action_chunk,
                )
                final_chunk = chunk_index == len(instructions)
                plan_length = (
                    self.execute_horizon if final_chunk else self.action_chunk)
                used_delay = action_queue.merge(
                    new_raw[:plan_length],
                    new_processed[:plan_length],
                    real_delay=measured_delay,
                    action_index_before_inference=action_index_before,
                )
                self._last_e2e_latency_s = inference_elapsed
                self._prev_ctx = self._action_ctx

                if used_delay > prefix_len:
                    overwatch.warning(
                        '[RemoteRTC %d/%d] actual delay %d exceeded guided '
                        'prefix %d; increase latency margin or reserve.',
                        chunk_index,
                        len(instructions),
                        used_delay,
                        prefix_len,
                    )
                overwatch.info(
                    '[RemoteRTC %d/%d] e2e=%.1fms measured=%d used=%d '
                    'prefix=%d raw_left=%d queue=%d->%d',
                    chunk_index,
                    len(instructions),
                    inference_elapsed * 1000.0,
                    measured_delay,
                    used_delay,
                    prefix_len,
                    len(raw_left_over),
                    queue_size_before,
                    action_queue.qsize(),
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
                        'Remote RTC action consumer did not stop within '
                        '2 seconds.')
