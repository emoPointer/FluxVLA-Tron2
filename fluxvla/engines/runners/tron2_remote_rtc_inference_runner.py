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
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from types import SimpleNamespace

import numpy as np
from tron2_env.rtc import ActionQueue

from ..utils import initialize_overwatch
from ..utils.root import RUNNERS
from .tron2_overlap_inference_runner import Tron2OverlapInferenceRunner


class _TerminalKeyReader:
    """Read single client-side keys while preserving terminal state."""

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
        tty.setcbreak(self._fd)
        self._reader_thread = threading.Thread(
            target=self._read_keys,
            daemon=True,
            name='Tron2-client-key-reader',
        )
        self._reader_thread.start()
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
        self._chunk_accept_lock = threading.Lock()

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
            return 0
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

    @staticmethod
    def _resolve_hold_delay(measured_delay: int, observed_delay: int,
                            queue_empty: bool) -> int:
        """Account for wall-clock frames spent holding an exhausted queue."""
        if not queue_empty:
            return 0
        return max(0, int(measured_delay) - int(observed_delay))

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

    def _wait_for_idle_command(self, key_reader):
        """Select a checkpoint task, start it, or request the prepare pose."""
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
                    'Unknown idle key %r. Use task ID + Enter, '
                    'b, or r.', key)

    def _monitor_active_keys(self, key_reader, stop_requested: threading.Event,
                             monitor_done: threading.Event,
                             monitor_errors: list[BaseException]):
        """Accept only ``s`` while inference or queue draining is active."""
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
                                       'the accepted action queue to finish.')
                    else:
                        overwatch.info(
                            'Stop requested: no further inference request or '
                            'action chunk will be accepted; draining the '
                            'current action queue.')
                elif command == 'r':
                    overwatch.warning(
                        'r is ignored while running or draining. Press s, '
                        'wait for the client to report idle, then press r.')
                elif command == 'b':
                    overwatch.info('Inference is already running. Press s to '
                                   'stop after the accepted queue finishes.')
                elif key not in {'\r', '\n', ' ', '\t'}:
                    overwatch.warning(
                        'Key %r is ignored while inference is '
                        'running; only s is active.', key)
        except BaseException as exc:
            monitor_errors.append(exc)
            with self._chunk_accept_lock:
                stop_requested.set()

    def _wait_for_trigger_or_stop(self, action_queue: ActionQueue,
                                  trigger_queue_size: int,
                                  shutdown_event: threading.Event,
                                  consumer_errors: list[BaseException],
                                  stop_requested: threading.Event) -> bool:
        while action_queue.qsize() > trigger_queue_size:
            self._raise_consumer_error(consumer_errors)
            if stop_requested.wait(self.queue_poll_interval_s):
                return False
            if shutdown_event.is_set():
                self._raise_consumer_error(consumer_errors)
                raise RuntimeError('Remote RTC action consumer stopped.')
        return not stop_requested.is_set()

    def _run_continuous_episode(self, instruction: str,
                                stop_requested: threading.Event):
        """Infer continuously until stopped, then drain the accepted queue."""
        overwatch = initialize_overwatch(__name__)
        if stop_requested.is_set():
            return

        action_queue = ActionQueue(rtc_enabled=True)
        producer_done = threading.Event()
        shutdown_event = threading.Event()
        consumer_errors: list[BaseException] = []
        consumer_thread = None

        try:
            self._action_ctx = SimpleNamespace(instruction=instruction)
            first_inputs = self._preprocess(instruction)
            if stop_requested.is_set():
                return
            first_raw, first_processed, first_elapsed = (
                self._request_action_pair(first_inputs))
            with self._chunk_accept_lock:
                if stop_requested.is_set():
                    overwatch.info(
                        'Discarding the first inference result because s was '
                        'pressed before it was accepted.')
                    return
                self._last_e2e_latency_s = first_elapsed
                action_queue.merge(
                    first_raw[:self.action_chunk],
                    first_processed[:self.action_chunk],
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

            chunk_index = 1
            while not stop_requested.is_set():
                if not self._wait_for_trigger_or_stop(
                        action_queue,
                        trigger_queue_size,
                        shutdown_event,
                        consumer_errors,
                        stop_requested,
                ):
                    break

                self._action_ctx = SimpleNamespace(instruction=instruction)
                inputs = self._preprocess(instruction)
                self._raise_consumer_error(consumer_errors)
                if stop_requested.is_set():
                    break
                action_index_before, raw_left_over, queue_size_before = (
                    action_queue.snapshot_left_over())
                prefix_len = self._resolve_prefix_len(raw_left_over)
                rtc_payload = (None
                               if prefix_len == 0 else self._build_rtc_payload(
                                   raw_left_over, prefix_len))
                if rtc_payload is None:
                    overwatch.warning(
                        '[RemoteRTC %d] no raw prefix remains; requesting '
                        'an unguided recovery chunk while holding the last '
                        'accepted target.', chunk_index + 1)

                new_raw, new_processed, inference_elapsed = (
                    self._request_action_pair(inputs, rtc=rtc_payload))
                self._raise_consumer_error(consumer_errors)
                with self._chunk_accept_lock:
                    discard_result = stop_requested.is_set()
                    if not discard_result:
                        measured_delay = min(
                            math.ceil(inference_elapsed / self.dt),
                            self.action_chunk,
                        )
                        observed_delay = max(
                            0,
                            action_queue.get_action_index() -
                            action_index_before,
                        )
                        hold_delay = self._resolve_hold_delay(
                            measured_delay,
                            observed_delay,
                            action_queue.qsize() == 0,
                        )
                        used_delay = action_queue.merge(
                            new_raw[:self.action_chunk],
                            new_processed[:self.action_chunk],
                            real_delay=measured_delay,
                            action_index_before_inference=action_index_before,
                            extra_delay=hold_delay,
                        )
                        self._last_e2e_latency_s = inference_elapsed
                        self._prev_ctx = self._action_ctx
                if discard_result:
                    overwatch.info(
                        'Discarding RemoteRTC chunk %d because s was pressed '
                        'while inference was in flight.', chunk_index + 1)
                    break

                if prefix_len > 0 and used_delay > prefix_len:
                    overwatch.warning(
                        '[RemoteRTC %d] actual delay %d exceeded guided '
                        'prefix %d; increase latency margin or reserve.',
                        chunk_index + 1,
                        used_delay,
                        prefix_len,
                    )
                overwatch.info(
                    '[RemoteRTC %d] e2e=%.1fms measured=%d used=%d '
                    'held=%d prefix=%d raw_left=%d queue=%d->%d',
                    chunk_index + 1,
                    inference_elapsed * 1000.0,
                    measured_delay,
                    used_delay,
                    hold_delay,
                    prefix_len,
                    0 if raw_left_over is None else len(raw_left_over),
                    queue_size_before,
                    action_queue.qsize(),
                )
                chunk_index += 1

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

    def _run_selected_task(self, key_reader, task_id: str):
        """Run one selected task while a client-side key monitor owns stdin."""
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
                self._run_continuous_episode(instruction, stop_requested)
        finally:
            monitor_done.set()
            monitor_thread.join(timeout=1.0)
            if monitor_thread.is_alive():
                raise RuntimeError('TRON2 active key monitor did not stop.')
        if monitor_errors:
            raise RuntimeError('TRON2 active key monitor failed.') from \
                monitor_errors[0]
        overwatch.info(
            'Task %s stopped. The accepted action queue is empty; '
            'select a task ID before pressing b again.', task_id)

    def run(self, initial_instruction: str = ''):
        """Run the robot-client-only keyboard state machine until Ctrl+C."""
        del initial_instruction
        overwatch = initialize_overwatch(__name__)
        overwatch.info('Starting TRON2 client keyboard control. All b/s/r '
                       'handling runs on this robot computer.')

        with _TerminalKeyReader() as key_reader:
            while True:
                command, task_id = self._wait_for_idle_command(key_reader)
                if command == 'prepare':
                    try:
                        self._move_to_prepare_pose()
                        overwatch.info('Prepare-pose sequence completed. '
                                       'Select a task ID, then press b.')
                    except Exception as exc:
                        overwatch.error(
                            'Prepare-pose command failed; client '
                            'remains idle: %s', exc)
                    continue
                self._run_selected_task(key_reader, task_id)
