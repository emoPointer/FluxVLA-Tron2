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

import json
import os
import pty
import runpy
import subprocess
import sys
import termios
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mmengine import Config

from fluxvla.engines.runners.serving.serve import load_deployment_metadata
from fluxvla.engines.runners.tron2_inference_runner import (
    Tron2InferenceRunner, _ActionJSONLRecorder, _TerminalKeyReader)
from fluxvla.engines.runners.tron2_rtc_inference_runner import (
    Tron2RTCInferenceRunner, _RTCActionPostProcessor)

ROOT = Path(__file__).resolve().parents[1]


def test_tron2_lora_config_deployment_defaults():
    cfg_path = (
        ROOT / 'configs' / 'pi05' / 'pi05_paligemma_tron2_lora_finetune.py')
    cfg = runpy.run_path(cfg_path)
    inference = cfg['inference']
    operator = inference['operator']

    assert cfg['model']['use_lora'] is True
    assert cfg['model']['ori_action_dim'] == 16
    assert inference['action_layout'] == 'tron2_16'
    # This private deployment configuration intentionally enables real actions.
    assert inference['dry_run'] is False
    assert inference['enable_head_control'] is False
    assert inference['publish_rate'] == 30
    assert inference['remote_inference']['server_host'] == '127.0.0.1'
    assert inference['remote_inference']['server_port'] == 5555
    assert inference['type'] == 'Tron2InferenceRunner'
    assert inference['action_chunk'] == 32
    assert inference['action_record_dir'] == 'work_dirs/action_records'
    assert 'rtc_config' not in inference
    assert 'execute_horizon' not in inference
    assert operator['type'] == 'Tron2EnvOperator'
    assert operator['bridge_host'] == 'wss://10.192.1.4'
    assert operator['bridge_ws_path'] == '/bridge/ws'
    assert operator['bridge_image_topics'] == {
        'camera_left': '/camera/left/color/image_resized/compressed',
        'camera_right': '/camera/right/color/image_resized/compressed',
        'camera_top': '/camera/top/color/image_raw/compressed',
    }
    assert operator['bridge_joint_topics'] == {
        'joint_states': '/joint_states',
        'gripper': '/gripper_state',
    }
    assert operator['bridge_verify_tls'] is False
    assert operator['ws_accid'] is None
    assert operator['movej_duration'] == 2.0
    assert operator['servoj_publish_rate'] == 300.0
    assert operator['recovery_blend_frames'] == 6
    assert operator['chunk_boundary_blend_enabled'] is False
    assert operator['chunk_boundary_blend_frames'] == 6
    assert operator['chunk_boundary_blend_scope'] == 'arm'
    assert operator['max_servoj_step_rad'] == 0.5
    assert operator['max_state_source_mismatch_rad'] is None
    assert operator['lock_head'] is True
    assert operator['max_head_hold_error_rad'] == 0.05


def test_tron2_rtc_local_inference_config():
    cfg = Config.fromfile(ROOT / 'configs' / 'pi05' /
                          'pi05_paligemma_tron2_lora_rtc_local_inference.py')
    inference = cfg.inference

    assert inference.type == 'Tron2RTCInferenceRunner'
    assert inference.remote_inference is None
    assert inference.action_chunk == 50
    assert inference.async_execution is False
    assert inference.execute_horizon is None
    assert inference.mixed_precision_dtype == 'bf16'
    assert inference.enable_mixed_precision is True
    assert inference.include_head_in_state is True
    assert inference.rtc_config == {
        'enabled': True,
        'method': 'prefix',
        'delay': 6,
        'execution_horizon': 10,
        'trigger_poll_interval_s': 0.005,
        'observation_timeout_budget_s': 5.0,
        'recovery_blend_frames': 6,
        'prefix_len': 19,
        'prefix_action_dim': 16,
        'prefix_head_from_observation': True,
        'action_postprocess': {
            'enabled': False,
            'boundary_blend_frames': 0,
            'boundary_blend_curve': 'smoothstep',
            'boundary_blend_scope': 'arm',
            'ema_alpha': 1.0,
            'ema_frames': 0,
            'ema_scope': 'arm',
        },
    }
    assert inference.publish_rate == 30
    assert inference.dry_run is False
    assert inference.action_record_dir == 'work_dirs/action_records'
    assert inference.operator.type == 'Tron2NativeEnvOperator'
    assert inference.operator.bridge_state_source == 'legacy'
    assert inference.operator.state_dim == 18
    assert inference.operator.fps == 30.0
    assert inference.operator.publish_rate == 300.0
    assert inference.operator.init_head is None
    assert inference.operator.reset_gripper_open_wait_s == 0.5
    expected_training_rtc = {
        'enabled': True,
        'max_delay': 20,
        'distribution': 'uniform',
        'delay_values': [0, 5, 10, 19],
        'temperature': 1.0,
    }
    assert cfg.model.rtc_training_config == expected_training_rtc
    assert cfg.inference_model.rtc_training_config == expected_training_rtc
    assert inference.task_descriptions == {
        '1':
        'Put the flowers in the vase',
        '2':
        'Put both dolls into the pink basket',
        '3':
        'Put both dolls into the gray basket',
        '4':
        'Put both pens into the pink basket',
        '5':
        'Put both pens into the gray basket',
        '6': ('Put a doll into the gray basket, and put the other doll into '
              'the pink basket.'),
        '7': ('Put a pen into the gray basket, and put the other  pen into '
              'the pink basket.'),
        '8': ('Put both dolls into the pink basket, and put both pens into '
              'the gray basket.'),
        '9': ('Put both dolls into the gray basket, and put both pens into '
              'the pink basket.'),
        '10':
        'Put all the objects into the pink basket.',
        '11':
        'Put all the objects into the gray basket.',
        '12':
        'fold clothes',
    }
    assert issubclass(Tron2RTCInferenceRunner, Tron2InferenceRunner)


def test_tron2_rtc_dynamic_delay_conditions_model_inputs():
    runner = Tron2RTCInferenceRunner.__new__(Tron2RTCInferenceRunner)
    runner.action_chunk = 50
    runner.vla = SimpleNamespace(rtc_training_config={
        'max_delay': 20,
        'delay_values': [0, 5, 10, 19],
    })
    runner._warned_delay_outside_training_range = False
    runner.fixed_model_prefix = None
    runner.prefix_action_dim = 16
    runner.prefix_head_from_observation = False
    previous = np.arange(40 * 32, dtype=np.float32).reshape(40, 32)

    padded, prefix_len = runner._prepare_rtc_inputs(previous, 6)

    assert prefix_len == 10
    assert padded.shape == (50, 16)
    np.testing.assert_array_equal(padded[:40], previous[:, :16])
    np.testing.assert_array_equal(padded[40:], np.zeros((10, 16)))


@pytest.mark.parametrize(('delay', 'expected_prefix'), [
    (0, 0),
    (1, 5),
    (5, 5),
    (6, 10),
    (9, 10),
    (10, 10),
    (19, 19),
    (20, 19),
])
def test_tron2_rtc_quantizes_delay_to_checkpoint_prefixes(
        delay, expected_prefix):
    runner = Tron2RTCInferenceRunner.__new__(Tron2RTCInferenceRunner)
    runner.action_chunk = 50
    runner.vla = SimpleNamespace(rtc_training_config={
        'max_delay': 20,
        'delay_values': [0, 5, 10, 19],
    })
    runner._warned_delay_outside_training_range = False
    runner.fixed_model_prefix = None
    runner.prefix_action_dim = 16
    runner.prefix_head_from_observation = False

    _, prefix_len = runner._prepare_rtc_inputs(
        np.zeros((40, 32), dtype=np.float32), delay)

    assert prefix_len == expected_prefix


def test_tron2_rtc_fixed_prefix_uses_measured_normalized_head():
    runner = Tron2RTCInferenceRunner.__new__(Tron2RTCInferenceRunner)
    runner.action_chunk = 50
    runner.vla = SimpleNamespace(rtc_training_config={
        'max_delay': 20,
        'delay_values': [0, 5, 10, 19],
    })
    runner._warned_delay_outside_training_range = False
    runner.fixed_model_prefix = 19
    runner.prefix_action_dim = 16
    runner.prefix_head_from_observation = True
    runner._latest_head_position = np.array([0.5, 1.5], dtype=np.float32)
    runner._reported_prefix_head = False
    runner.task_suite_name = 'private'
    minimum = np.zeros(18, dtype=np.float32)
    maximum = np.full(18, 2.0, dtype=np.float32)
    runner.denormalize_action = SimpleNamespace(
        norm_type='min_max',
        norm_stats={
            'private': {
                'action': {
                    'min': minimum.tolist(),
                    'max': maximum.tolist(),
                }
            }
        },
    )
    previous = np.arange(40 * 32, dtype=np.float32).reshape(40, 32)

    padded, prefix_len = runner._prepare_rtc_inputs(previous, 6)

    assert prefix_len == 19
    assert padded.shape == (50, 18)
    np.testing.assert_array_equal(padded[:40, :16], previous[:, :16])
    expected_head = np.tile([-0.5, 0.5], (40, 1))
    np.testing.assert_allclose(padded[:40, 16:18], expected_head, atol=1e-6)
    np.testing.assert_array_equal(padded[40:], np.zeros((10, 18)))


def test_tron2_rtc_runtime_uses_upstream_action_queue_and_latency_tracker():
    from tron2_env.rtc import ActionQueue, LatencyTracker

    assert Tron2RTCInferenceRunner._rtc_runtime_types() == (ActionQueue,
                                                            LatencyTracker)


def test_tron2_rtc_recent_delay_p95_matches_public_client():
    assert Tron2RTCInferenceRunner._p95_int([]) == 0
    assert Tron2RTCInferenceRunner._p95_int([2]) == 2
    assert Tron2RTCInferenceRunner._p95_int([1, 2, 3, 4, 5, 6, 7, 8, 9,
                                             10]) == 10


def test_tron2_rtc_accepts_padded_raw_and_tron2_command_dimensions():
    runner = Tron2RTCInferenceRunner.__new__(Tron2RTCInferenceRunner)
    runner.action_chunk = 50
    runner._action_ctx = SimpleNamespace(
        raw_actions=np.zeros((50, 32), dtype=np.float32))
    processed = np.zeros((50, 16), dtype=np.float32)

    assert runner._validate_action_chunks(processed) is processed


def test_tron2_rtc_action_queue_merge_uses_actual_consumed_index():
    from tron2_env.rtc import ActionQueue

    action_queue = ActionQueue(rtc_enabled=True)
    old = np.arange(50, dtype=np.float32)[:, None]
    action_queue.merge(old, old, real_delay=0)
    action_index_before, _, _ = action_queue.snapshot_left_over()
    for _ in range(7):
        action_queue.get()

    new = np.arange(1000, 1050, dtype=np.float32)[:, None]
    used_delay = action_queue.merge(
        new,
        new,
        real_delay=0,
        action_index_before_inference=action_index_before,
    )

    assert used_delay == 7
    np.testing.assert_array_equal(action_queue.get(), [1007.0])


def test_tron2_rtc_consumer_uses_upstream_shutdown_event():
    from tron2_env.rtc import ActionQueue

    runner = Tron2RTCInferenceRunner.__new__(Tron2RTCInferenceRunner)
    runner.dt = 0.001
    runner.recovery_blend_frames = 6
    issued = []
    shutdown_event = threading.Event()

    def execute(action, source_index, queue_size):
        issued.append((action.copy(), source_index, queue_size))
        if len(issued) == 3:
            shutdown_event.set()

    runner._execute_rtc_waypoint = execute
    action_queue = ActionQueue(rtc_enabled=True)
    actions = np.arange(3 * 16, dtype=np.float32).reshape(3, 16)
    action_queue.merge(actions, actions, real_delay=0)
    errors = []

    runner._consume_actions(action_queue, shutdown_event, errors)

    assert not errors
    assert [item[1] for item in issued] == [0, 1, 2]
    assert [item[2] for item in issued] == [2, 1, 0]
    np.testing.assert_array_equal(
        np.stack([item[0] for item in issued]), actions)


def test_tron2_rtc_optional_smoothstep_matches_public_client_boundary():
    processor = _RTCActionPostProcessor({
        'enabled': True,
        'boundary_blend_frames': 2,
        'boundary_blend_curve': 'smoothstep',
        'boundary_blend_scope': 'arm',
        'ema_alpha': 1.0,
    })
    old = np.zeros((2, 16), dtype=np.float64)
    new = np.ones((4, 16), dtype=np.float64)

    processed = processor.apply(new, old, merge_delay=1)

    first_alpha = (1.0 / 3.0)**2 * (3.0 - 2.0 / 3.0)
    second_alpha = (2.0 / 3.0)**2 * (3.0 - 4.0 / 3.0)
    arm_indices = list(range(7)) + list(range(8, 15))
    np.testing.assert_allclose(processed[1, arm_indices], first_alpha)
    np.testing.assert_allclose(processed[2, arm_indices], second_alpha)
    np.testing.assert_allclose(processed[:, [7, 15]], 1.0)


def test_all_tron2_configs_use_tron2_env_operator():
    config_paths = [
        ROOT / 'configs' / 'pi05' / 'pi05_paligemma_tron2_full_finetune.py',
        ROOT / 'configs' / 'pi05' / 'pi05_paligemma_tron2_lora_finetune.py',
        ROOT / 'configs' / 'gr00t' /
        'gr00t_eagle_3b_tron2_3cam_full_finetune.py',
    ]

    for config_path in config_paths:
        operator = runpy.run_path(config_path)['inference']['operator']
        assert operator['type'] == 'Tron2EnvOperator'
        assert operator['bridge_host'] == 'wss://10.192.1.4'
        assert 'img_top_topic' not in operator
        assert 'joint_state_topic' not in operator
        assert operator['servoj_publish_rate'] == 300.0
        assert operator['recovery_blend_frames'] == 6
        assert operator.get('chunk_boundary_blend_enabled', False) is False
        expected_step_limit = (0.5 if config_path.name
                               == 'pi05_paligemma_tron2_lora_finetune.py' else
                               0.2)
        assert operator['max_servoj_step_rad'] == expected_step_limit
        assert operator['max_state_source_mismatch_rad'] is None
        assert operator['lock_head'] is True


def test_checkpoint_metadata_overrides_generic_task_map(tmp_path):
    work_dir = tmp_path / 'task_checkpoint'
    checkpoint = work_dir / 'checkpoints' / 'step.safetensors'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (work_dir / 'config.json').write_text(
        json.dumps({
            'inference': {
                'action_layout': 'tron2_16',
                'task_descriptions': {
                    '1': 'task one',
                    '6': 'fold clothes',
                },
            }
        }),
        encoding='utf-8',
    )
    cfg = Config({
        'inference': {
            'action_layout': 'tron2_16',
            'task_descriptions': {
                '1': 'unrelated generic prompt'
            },
        }
    })

    metadata = load_deployment_metadata(cfg, str(checkpoint))

    assert metadata['task_descriptions'] == {
        '1': 'task one',
        '6': 'fold clothes',
    }
    assert metadata['action_layout'] == 'tron2_16'


def test_explicit_deployment_metadata_overrides_stale_training_config(
        tmp_path):
    work_dir = tmp_path / 'mixed_task_checkpoint'
    checkpoint = work_dir / 'checkpoints' / 'step.safetensors'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (work_dir / 'config.json').write_text(
        json.dumps({
            'inference': {
                'action_layout': 'tron2_16',
                'task_descriptions': {
                    '1': 'stale banana example'
                },
            }
        }),
        encoding='utf-8',
    )
    (work_dir / 'deployment_metadata.json').write_text(
        json.dumps({
            'action_layout': 'tron2_16',
            'task_descriptions': {
                '1': 'task one',
                '3': 'task three',
            },
        }),
        encoding='utf-8',
    )
    cfg = Config({
        'inference': {
            'type': 'Tron2InferenceRunner',
            'action_layout': 'tron2_16',
            'task_descriptions': {
                '1': 'generic launch prompt'
            },
        }
    })

    metadata = load_deployment_metadata(cfg, str(checkpoint))

    assert metadata['task_descriptions'] == {
        '1': 'task one',
        '3': 'task three',
    }
    assert metadata['metadata_source'] == 'deployment_metadata.json'


def test_tron2_checkpoint_metadata_never_falls_back_to_robot_config(tmp_path):
    checkpoint = tmp_path / 'missing_config' / 'checkpoints' / 'step.safe'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    cfg = Config({
        'inference': {
            'type': 'Tron2InferenceRunner',
            'action_layout': 'tron2_16',
            'task_descriptions': {
                '1': 'unrelated robot-side prompt'
            },
        }
    })

    with pytest.raises(
            FileNotFoundError, match='checkpoint-local task metadata'):
        load_deployment_metadata(cfg, str(checkpoint))


def test_tron2_runner_rejects_unknown_task_id():
    runner = Tron2InferenceRunner.__new__(Tron2InferenceRunner)
    runner.task_descriptions = {'6': 'fold clothes'}

    assert runner._get_task_description('6') == 'fold clothes'
    with pytest.raises(ValueError, match='Unknown task ID'):
        runner._get_task_description('1')


class _FakeKeyReader:

    def __init__(self, keys):
        self._keys = iter(keys)

    def get_key(self, timeout=None):
        del timeout
        try:
            return next(self._keys)
        except StopIteration:
            time.sleep(0.001)
            return None


def test_tron2_action_jsonl_recorder_writes_complete_session(tmp_path):
    recorder = _ActionJSONLRecorder(tmp_path)
    path = recorder.start({
        'inference_mode': 'rtc',
        'action_layout': 'tron2_16',
    })
    assert recorder.record({
        'record_type': 'action',
        'trajectory_id': 3,
        'action': list(range(16)),
    })

    stopped_path, count, error = recorder.stop()

    assert stopped_path == path
    assert count == 1
    assert error is None
    records = [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
    ]
    assert [record['record_type'] for record in records
            ] == ['session_start', 'action', 'session_end']
    assert records[1]['sequence_index'] == 0
    assert records[1]['action'] == list(range(16))
    assert records[2]['action_count'] == 1


def test_tron2_runner_records_issued_action_in_training_layout(tmp_path):
    runner = Tron2InferenceRunner.__new__(Tron2InferenceRunner)
    runner.action_layout = 'tron2_16'
    runner._action_recorder = _ActionJSONLRecorder(tmp_path)
    path = runner._action_recorder.start({
        'inference_mode': 'non_rtc',
        'action_layout': 'tron2_16',
    })

    runner._record_issued_action(
        waypoint=np.arange(16, dtype=np.float64),
        left_gripper=0.25,
        right_gripper=0.75,
        trajectory_index=4,
        issued_at_unix=10.0,
        issued_at_monotonic=5.0,
        action_metadata={
            'trajectory_id': 2,
            'task_id': '1'
        },
    )
    runner._action_recorder.stop()

    records = [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
    ]
    action_record = records[1]
    assert action_record['task_id'] == '1'
    assert action_record['trajectory_id'] == 2
    assert action_record['trajectory_frame_index'] == 4
    assert action_record['action'] == (
        list(range(7)) + [0.25] + list(range(7, 14)) + [0.75])


def test_tron2_terminal_key_reader_reads_one_key_and_restores_tty():
    master_fd, slave_fd = pty.openpty()
    saved_attributes = termios.tcgetattr(slave_fd)
    try:
        with os.fdopen(os.dup(slave_fd), 'r', encoding='utf-8') as stream:
            with _TerminalKeyReader(stream) as key_reader:
                os.write(master_fd, b's')
                assert key_reader.get_key(timeout=1.0) == 's'
        assert termios.tcgetattr(slave_fd) == saved_attributes
    finally:
        os.close(master_fd)
        os.close(slave_fd)


def test_tron2_remote_client_import_does_not_require_torch():
    env = os.environ.copy()
    env['FLUXVLA_REMOTE_CLIENT_ONLY'] = '1'
    env['PYTHONPATH'] = str(ROOT)
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; '
            'from fluxvla.engines.runners import Tron2InferenceRunner; '
            "assert 'torch' not in sys.modules; "
            "print(Tron2InferenceRunner.__name__)",
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith('Tron2InferenceRunner')


def test_tron2_idle_keyboard_requires_confirmed_task_before_start():
    runner = Tron2InferenceRunner.__new__(Tron2InferenceRunner)
    runner.task_descriptions = {
        '1': 'put flowers in vase',
        '6': 'fold clothes',
    }
    toggles = []
    runner._toggle_action_recording = lambda: toggles.append('toggle')

    command = runner._wait_for_idle_command(
        _FakeKeyReader(['l', 'b', '6', '\n', 'b']))

    assert command == ('start', '6')
    assert toggles == ['toggle']


def test_tron2_idle_keyboard_r_requests_prepare_pose():
    runner = Tron2InferenceRunner.__new__(Tron2InferenceRunner)
    runner.task_descriptions = {'1': 'put flowers in vase'}

    assert runner._wait_for_idle_command(_FakeKeyReader(['r'])) == ('prepare',
                                                                    None)


def test_tron2_active_keyboard_stops_and_ignores_prepare():
    runner = Tron2InferenceRunner.__new__(Tron2InferenceRunner)
    runner._chunk_accept_lock = threading.Lock()
    stop_requested = threading.Event()
    monitor_done = threading.Event()
    monitor_errors = []
    toggles = []
    runner._toggle_action_recording = lambda: toggles.append('toggle')
    monitor_thread = threading.Thread(
        target=runner._monitor_active_keys,
        args=(
            _FakeKeyReader(['r', 'b', 'l', 's']),
            stop_requested,
            monitor_done,
            monitor_errors,
        ),
    )

    monitor_thread.start()
    assert stop_requested.wait(timeout=1.0)
    monitor_done.set()
    monitor_thread.join(timeout=1.0)

    assert not monitor_thread.is_alive()
    assert not monitor_errors
    assert toggles == ['toggle']


def test_tron2_async_task_drains_before_idle_reset_is_enabled():
    runner = Tron2InferenceRunner.__new__(Tron2InferenceRunner)
    runner.async_execution = True
    runner.task_descriptions = {'1': 'put flowers in vase'}
    runner.task_pose_sequences = {}
    events = []

    class FakeOperator:

        def wait_for_trajectory(self):
            events.append('trajectory_finished')

    runner.ros_operator = FakeOperator()

    def run_continuous_task(instruction, stop_requested):
        assert instruction == 'put flowers in vase'
        events.append('task_stopped')
        stop_requested.set()

    runner._run_continuous_task = run_continuous_task
    runner._run_selected_task(_FakeKeyReader([]), '1')

    assert events == ['task_stopped', 'trajectory_finished']


def _make_non_rtc_runner():
    runner = Tron2InferenceRunner.__new__(Tron2InferenceRunner)
    runner._use_remote = True
    runner._chunk_accept_lock = threading.Lock()
    runner._preprocess = lambda instruction: {'instruction': instruction}
    runner._postprocess_actions = lambda raw_action: raw_action
    return runner


def test_tron2_stop_during_inference_discards_unaccepted_chunk():
    runner = _make_non_rtc_runner()
    stop_requested = threading.Event()
    executed = []

    def predict_action(inputs):
        assert inputs['instruction'] == 'fold clothes'
        stop_requested.set()
        return np.zeros((32, 16), dtype=np.float32)

    runner._predict_action = predict_action
    runner._execute_actions = lambda actions, rate: executed.append(actions)

    runner._run_continuous_task('fold clothes', stop_requested)

    assert executed == []


def test_tron2_stop_during_accepted_chunk_allows_chunk_to_finish():
    runner = _make_non_rtc_runner()
    stop_requested = threading.Event()
    actions = np.zeros((32, 16), dtype=np.float32)
    executed = []
    runner._predict_action = lambda inputs: actions

    def execute_actions(accepted_actions, rate):
        assert rate is None
        executed.append(accepted_actions)
        stop_requested.set()

    runner._execute_actions = execute_actions

    runner._run_continuous_task('fold clothes', stop_requested)

    assert len(executed) == 1
    assert executed[0] is actions


def test_tron2_rejected_chunk_holds_and_continues_inference():
    runner = _make_non_rtc_runner()
    stop_requested = threading.Event()
    actions = np.zeros((32, 16), dtype=np.float32)
    attempts = []
    runner._predict_action = lambda inputs: actions

    def execute_actions(accepted_actions, rate):
        attempts.append((accepted_actions, rate))
        if len(attempts) == 1:
            raise ValueError('unsafe waypoint')
        stop_requested.set()

    runner._execute_actions = execute_actions

    runner._run_continuous_task('fold clothes', stop_requested)

    assert len(attempts) == 2


def test_readme_links_are_current():
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')

    assert 'Training and Deploying FluxVLA on a New Tron2' in readme
    assert 'https://github.com/FluxVLA/FluxVLA/blob/main/README.md' in readme
    assert 'Project Scope and Status' in readme
    assert 'Repository Layout' in readme
    assert 'License and Contributions' in readme
    assert 'README_zh-CN.md' in readme

    zh_readme = (ROOT / 'README_zh-CN.md').read_text(encoding='utf-8')
    assert '项目适用范围与状态' in zh_readme
    assert '目录结构' in zh_readme
    assert 'License、贡献和反馈入口' in zh_readme


def test_release_governance_files_exist():
    required = [
        '.env.example',
        'NOTICE',
        'CONTRIBUTING.md',
        'SECURITY.md',
        'CHANGELOG.md',
        '.github/ISSUE_TEMPLATE/bug_report.md',
        '.github/ISSUE_TEMPLATE/feature_request.md',
        '.github/PULL_REQUEST_TEMPLATE.md',
        'docs/release_license_review.md',
        'docs/tron2_training_deployment.md',
        'fluxvla/models/third_party_models/xvla_models/ATTRIBUTION.txt',
    ]

    for rel_path in required:
        assert (ROOT / rel_path).is_file()
