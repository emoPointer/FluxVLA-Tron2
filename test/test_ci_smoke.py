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

import io
import json
import os
import pty
import runpy
import subprocess
import sys
import termios
from pathlib import Path
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest
import torch
from mmengine import Config
from tron2_env.rtc import ActionQueue

from fluxvla.engines.runners.serving.serializers import MsgSerializer
from fluxvla.engines.runners.serving.serializers import encode_predict_request
from fluxvla.engines.runners.serving.serve import load_deployment_metadata
from fluxvla.engines.runners.serving.zmq_server import create_server
from fluxvla.engines.runners.serving.zmq_server import prepare_remote_rtc_inputs
from fluxvla.engines.runners.tron2_inference_runner import (
    Tron2InferenceRunner, )
from fluxvla.engines.runners.tron2_overlap_inference_runner import (
    Tron2OverlapInferenceRunner, )
from fluxvla.engines.runners.tron2_remote_rtc_inference_runner import (
    Tron2RemoteRTCInferenceRunner, _TerminalKeyReader)

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
    assert inference['type'] == 'Tron2RemoteRTCInferenceRunner'
    assert inference['action_chunk'] == 50
    assert inference['execute_horizon'] == 10
    assert inference['hold_warning_interval_s'] == 1.0
    assert inference['rtc_config'] == {
        'enabled': True,
        'method': 'guidance',
        'prefix_len': None,
        'latency_margin_frames': 2,
        'decay_frames': 5,
        'schedule': 'exp',
        'max_guidance_weight': 5.0,
        'use_vjp': False,
    }
    assert inference['remote_inference']['server_host'] == '127.0.0.1'
    assert inference['remote_inference']['server_port'] == 5555
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
    assert operator['max_servoj_step_rad'] == 0.2
    assert operator['max_state_source_mismatch_rad'] is None
    assert operator['lock_head'] is True
    assert operator['max_head_hold_error_rad'] == 0.05


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
        assert operator['max_servoj_step_rad'] == 0.2
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
            'type': 'Tron2OverlapInferenceRunner',
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


def test_overlap_runner_crossfades_arms_and_switches_grippers():
    runner = Tron2OverlapInferenceRunner.__new__(Tron2OverlapInferenceRunner)
    runner.action_layout = 'tron2_16'
    runner.action_chunk = 5
    runner.blend_start_weight = 0.0
    runner.blend_end_weight = 1.0

    old = np.zeros((5, 16), dtype=np.float32)
    new = np.ones((5, 16), dtype=np.float32)
    plan, diagnostics = runner._blend_overlapping_plan(old, new)

    arm_indices = tuple(range(7)) + tuple(range(8, 15))
    expected_arms = np.repeat(np.linspace(0.0, 1.0, 5)[:, None], 14, axis=1)
    np.testing.assert_allclose(plan[:, arm_indices], expected_arms)
    np.testing.assert_array_equal(plan[:, 7], [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(plan[:, 15], [0, 0, 1, 1, 1])
    assert diagnostics == {
        'overlap_frames': 5,
        'max_arm_disagreement_rad': 1.0,
    }


def test_overlap_queue_discards_frames_actually_consumed_during_inference():
    runner = Tron2OverlapInferenceRunner.__new__(Tron2OverlapInferenceRunner)
    runner.action_layout = 'tron2_16'
    runner.action_chunk = 6
    runner.blend_start_weight = 0.0
    runner.blend_end_weight = 1.0

    action_queue = ActionQueue(rtc_enabled=True)
    initial = np.repeat(np.arange(6, dtype=np.float32)[:, None], 16, axis=1)
    action_queue.merge(initial, initial, real_delay=0)
    for _ in range(3):
        action_queue.get()
    index_before, old_left_over, _ = action_queue.snapshot_left_over()

    new = np.repeat(np.arange(100, 106, dtype=np.float32)[:, None], 16, axis=1)
    plan, _ = runner._blend_overlapping_plan(old_left_over, new)
    action_queue.get()
    action_queue.get()
    used_delay = action_queue.merge(
        plan,
        plan,
        real_delay=1,
        action_index_before_inference=index_before,
    )

    assert used_delay == 2
    np.testing.assert_allclose(action_queue.get(), plan[2])


def test_overlap_episode_executes_one_horizon_per_requested_chunk():
    runner = Tron2OverlapInferenceRunner.__new__(Tron2OverlapInferenceRunner)
    runner.action_layout = 'tron2_16'
    runner.action_chunk = 4
    runner.execute_horizon = 2
    runner.blend_start_weight = 0.0
    runner.blend_end_weight = 1.0
    runner.queue_poll_interval_s = 0.0005
    runner.hold_warning_interval_s = 1.0
    runner.dt = 0.005
    runner._prev_ctx = None
    runner._action_ctx = None

    chunks = [
        np.full((4, 16), value, dtype=np.float32) for value in (0.0, 1.0, 2.0)
    ]
    executed = []
    runner._get_user_task_instruction = lambda _: ['task'] * 3
    runner._predict_processed_chunk = lambda _: chunks.pop(0)
    runner._preprocess = lambda _: {}
    runner._predict_action = lambda _: chunks.pop(0)
    runner._postprocess_actions = lambda actions: actions
    runner._execute_waypoint = lambda action: executed.append(action.copy())

    runner._run_episode('unused')

    assert len(executed) == 3 * runner.execute_horizon


def test_overlap_consumer_holds_on_queue_underrun(caplog):
    caplog.set_level(30)
    runner = Tron2OverlapInferenceRunner.__new__(Tron2OverlapInferenceRunner)
    runner.dt = 0.001
    runner.hold_warning_interval_s = 0.005
    errors = []
    shutdown_event = threading.Event()
    consumer = threading.Thread(
        target=runner._consume_actions,
        args=(ActionQueue(rtc_enabled=True), threading.Event(), shutdown_event,
              errors),
    )

    consumer.start()
    time.sleep(0.015)
    assert consumer.is_alive()
    assert not errors
    shutdown_event.set()
    consumer.join(timeout=1.0)

    assert not consumer.is_alive()
    assert not errors
    assert any('[Hold]' in record.message for record in caplog.records)


def test_overlap_consumer_skips_rejected_waypoint_and_continues():
    runner = Tron2OverlapInferenceRunner.__new__(Tron2OverlapInferenceRunner)
    runner.dt = 0.001
    runner.hold_warning_interval_s = 1.0
    attempts = []

    def execute(action):
        attempts.append(action.copy())
        if len(attempts) == 1:
            raise ValueError('test rejection')

    runner._execute_waypoint = execute
    action_queue = ActionQueue(rtc_enabled=True)
    actions = np.zeros((2, 16), dtype=np.float32)
    action_queue.merge(actions, actions, real_delay=0)
    producer_done = threading.Event()
    producer_done.set()
    errors = []

    runner._consume_actions(
        action_queue,
        producer_done,
        threading.Event(),
        errors,
    )

    assert len(attempts) == 2
    assert not errors


def test_remote_rtc_msgpack_request_round_trip():
    prev_actions = np.arange(48, dtype=np.float32).reshape(3, 16)
    request = encode_predict_request(
        {'qpos': np.zeros(16, dtype=np.float32)},
        'private',
        rtc={
            'prev_actions': prev_actions,
            'prefix_len': 2,
            'config': {
                'enabled': True,
                'method': 'guidance',
            },
        },
    )

    parsed = MsgSerializer.from_bytes(request)

    assert parsed['endpoint'] == 'predict_action'
    assert parsed['data']['unnorm_key'] == 'private'
    np.testing.assert_array_equal(parsed['data']['rtc']['prev_actions'],
                                  prev_actions)
    assert parsed['data']['rtc']['prefix_len'] == 2


def test_remote_client_imports_without_torch():
    script = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ModuleNotFoundError('torch is unavailable on the robot client')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from fluxvla.engines.operators import Tron2EnvOperator
from fluxvla.engines.runners import Tron2RemoteRTCInferenceRunner
print('torch-free remote import ok')
"""
    env = dict(os.environ)
    env['FLUXVLA_REMOTE_CLIENT_ONLY'] = '1'

    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'torch-free remote import ok'


def test_remote_rtc_server_inputs_are_validated_and_moved_to_reference():
    reference = torch.zeros((1, 16), dtype=torch.bfloat16)
    payload = {
        'prev_actions': np.ones((8, 16), dtype=np.float32),
        'prefix_len': 5,
        'config': {
            'enabled': True,
            'method': 'guidance',
            'decay_end': 8,
            'schedule': 'exp',
            'max_guidance_weight': 5.0,
            'use_vjp': False,
        },
    }

    inputs = prepare_remote_rtc_inputs(payload, reference, 50)

    assert inputs['prev_actions'].shape == (1, 8, 16)
    assert inputs['prev_actions'].dtype == torch.bfloat16
    assert inputs['prefix_len'] == 5
    assert inputs['rtc_config'] == {
        'method': 'guidance',
        'decay_end': 8,
        'schedule': 'exp',
        'max_guidance_weight': 5.0,
        'use_vjp': False,
    }

    invalid = dict(payload)
    invalid['prev_actions'] = np.full((8, 16), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match='non-finite'):
        prepare_remote_rtc_inputs(invalid, reference, 50)

    with pytest.raises(ValueError, match='action dimension'):
        prepare_remote_rtc_inputs(payload, reference, 50, 14)


def test_remote_rtc_server_returns_raw_and_processed_action_pairs():

    class FakeVLA:
        n_action_steps = 4
        ori_action_dim = 2

        def __init__(self):
            self.inputs = None

        def eval(self):
            return self

        def to(self, _device):
            return self

        def predict_action(self, **inputs):
            self.inputs = inputs
            return torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)

    fake_vla = FakeVLA()

    def dataset(_obs):
        return {'states': torch.zeros((1, 2), dtype=torch.float32)}

    def denormalize(sample):
        return sample['action'][:, :2] + 100.0

    server = create_server(
        fake_vla,
        dataset=dataset,
        denormalize_action=denormalize,
        host='127.0.0.1',
        port=0,
        device='cpu',
    )
    rtc = {
        'prev_actions': np.ones((3, 2), dtype=np.float32),
        'prefix_len': 2,
        'config': {
            'enabled': True,
            'method': 'guidance',
            'decay_end': 3,
            'schedule': 'linear',
            'max_guidance_weight': 4.0,
            'use_vjp': False,
        },
    }

    try:
        response = server._endpoints['predict_action'].handler(
            obs_data=MsgSerializer.to_bytes({}),
            unnorm_key='private',
            rtc=rtc,
        )
        metadata = server._endpoints['get_deployment_metadata'].handler()
    finally:
        server.socket.close(linger=0)
        server.context.term()

    raw = np.load(io.BytesIO(response['raw_action_data']), allow_pickle=False)
    processed = np.load(
        io.BytesIO(response['action_data']), allow_pickle=False)

    assert raw.shape == processed.shape == (1, 4, 2)
    np.testing.assert_array_equal(raw[0], [[0, 1], [4, 5], [8, 9], [12, 13]])
    np.testing.assert_array_equal(processed, raw + 100.0)
    assert fake_vla.inputs['prev_actions'].shape == (1, 3, 2)
    assert fake_vla.inputs['prefix_len'] == 2
    assert fake_vla.inputs['rtc_config'] == {
        'method': 'guidance',
        'decay_end': 3,
        'schedule': 'linear',
        'max_guidance_weight': 4.0,
        'use_vjp': False,
    }
    assert metadata['remote_rtc']['returns_raw_actions'] is True


def test_remote_rtc_runner_builds_dynamic_guidance_payload():
    runner = Tron2RemoteRTCInferenceRunner.__new__(
        Tron2RemoteRTCInferenceRunner)
    runner.action_chunk = 50
    runner.dt = 1.0 / 30.0
    runner.rtc_prefix_len = None
    runner.rtc_latency_margin_frames = 2
    runner.rtc_decay_frames = 5
    runner.rtc_schedule = 'exp'
    runner.rtc_max_guidance_weight = 5.0
    runner.rtc_use_vjp = False
    runner._last_e2e_latency_s = 0.20
    raw_left_over = np.zeros((40, 16), dtype=np.float32)

    prefix_len = runner._resolve_prefix_len(raw_left_over)
    payload = runner._build_rtc_payload(raw_left_over, prefix_len)

    assert prefix_len == 8
    assert payload['prefix_len'] == 8
    assert payload['config']['decay_end'] == 13
    np.testing.assert_array_equal(payload['prev_actions'], raw_left_over)
    assert runner._resolve_hold_delay(12, 9, queue_empty=True) == 3
    assert runner._resolve_hold_delay(12, 9, queue_empty=False) == 0
    assert runner._resolve_prefix_len(np.empty((0, 16))) == 0


class _FakeKeyReader:

    def __init__(self, keys):
        self._keys = list(keys)
        self._lock = threading.Lock()

    def get_key(self, timeout=None):
        del timeout
        with self._lock:
            if self._keys:
                return self._keys.pop(0)
        time.sleep(0.001)
        return None


def test_remote_rtc_terminal_reader_reads_one_key_and_restores_tty():
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


def test_remote_rtc_idle_keys_require_task_confirmation_before_start():
    runner = Tron2RemoteRTCInferenceRunner.__new__(
        Tron2RemoteRTCInferenceRunner)
    runner.task_descriptions = {'2': 'put both dolls in the pink basket'}

    command, task_id = runner._wait_for_idle_command(
        _FakeKeyReader(['b', '2', 'b', '\n', 'b']))

    assert command == 'start'
    assert task_id == '2'


def test_remote_rtc_idle_r_requests_prepare_pose_without_task():
    runner = Tron2RemoteRTCInferenceRunner.__new__(
        Tron2RemoteRTCInferenceRunner)

    assert runner._wait_for_idle_command(_FakeKeyReader(['r'])) == ('prepare',
                                                                    None)


def test_remote_rtc_active_keys_only_allow_stop(caplog):
    caplog.set_level(20)
    runner = Tron2RemoteRTCInferenceRunner.__new__(
        Tron2RemoteRTCInferenceRunner)
    runner._chunk_accept_lock = threading.Lock()
    stop_requested = threading.Event()
    monitor_done = threading.Event()
    monitor_errors = []
    monitor = threading.Thread(
        target=runner._monitor_active_keys,
        args=(_FakeKeyReader(['r', 'b', 's']), stop_requested, monitor_done,
              monitor_errors),
    )

    monitor.start()
    assert stop_requested.wait(timeout=1.0)
    monitor_done.set()
    monitor.join(timeout=1.0)

    assert not monitor.is_alive()
    assert not monitor_errors
    assert any('r is ignored while running' in record.message
               for record in caplog.records)


def test_remote_rtc_episode_stops_new_inference_and_drains_accepted_actions():
    runner = Tron2RemoteRTCInferenceRunner.__new__(
        Tron2RemoteRTCInferenceRunner)
    runner.action_layout = 'tron2_16'
    runner.action_chunk = 4
    runner.execute_horizon = 2
    runner.queue_poll_interval_s = 0.0005
    runner.hold_warning_interval_s = 1.0
    runner.dt = 0.02
    runner.rtc_prefix_len = 1
    runner.rtc_latency_margin_frames = 0
    runner.rtc_decay_frames = 1
    runner.rtc_schedule = 'linear'
    runner.rtc_max_guidance_weight = 3.0
    runner.rtc_use_vjp = False
    runner._last_e2e_latency_s = None
    runner._prev_ctx = None
    runner._action_ctx = None
    runner._chunk_accept_lock = threading.Lock()

    raw_chunks = [
        np.repeat(np.arange(4, dtype=np.float32)[:, None], 16, axis=1),
        np.full((4, 16), 10.0, dtype=np.float32),
        np.full((4, 16), 20.0, dtype=np.float32),
    ]
    processed_chunks = [chunk + 100.0 for chunk in raw_chunks]
    requests = []
    executed = []
    runner._preprocess = lambda _: {'unnorm_key': 'private'}
    stop_requested = threading.Event()

    def request_pair(inputs, rtc=None):
        del inputs
        requests.append(rtc)
        if len(requests) == 3:
            stop_requested.set()
        return raw_chunks.pop(0), processed_chunks.pop(0), 0.001

    runner._request_action_pair = request_pair
    runner._execute_waypoint = lambda action: executed.append(action.copy())

    runner._run_continuous_episode('task', stop_requested)

    assert len(requests) == 3
    assert executed
    assert all(float(action[0]) >= 100.0 for action in executed)
    assert all(float(action[0]) < 120.0 for action in executed)
    assert requests[0] is None
    assert len(requests[1]['prev_actions']) >= runner.execute_horizon
    assert requests[1]['prev_actions'][0, 0] >= 2.0
    assert requests[1]['config']['method'] == 'guidance'
    assert requests[2] is not None


def test_remote_rtc_stop_during_first_inference_sends_no_actions():
    runner = Tron2RemoteRTCInferenceRunner.__new__(
        Tron2RemoteRTCInferenceRunner)
    runner.action_layout = 'tron2_16'
    runner.action_chunk = 4
    runner.execute_horizon = 2
    runner.queue_poll_interval_s = 0.0005
    runner.hold_warning_interval_s = 1.0
    runner.dt = 0.005
    runner._prev_ctx = None
    runner._action_ctx = None
    runner._chunk_accept_lock = threading.Lock()
    stop_requested = threading.Event()
    executed = []
    chunk = np.zeros((4, 16), dtype=np.float32)
    runner._preprocess = lambda _: {}

    def request_pair(inputs, rtc=None):
        del inputs, rtc
        stop_requested.set()
        return chunk, chunk, 0.001

    runner._request_action_pair = request_pair
    runner._execute_waypoint = lambda action: executed.append(action.copy())

    runner._run_continuous_episode('task', stop_requested)

    assert not executed


def test_tron2_runner_requires_explicit_checkpoint_task_id():
    runner = Tron2InferenceRunner.__new__(Tron2InferenceRunner)
    runner.task_descriptions = {
        '1': 'put flowers in vase',
        '6': 'fold clothes',
    }
    runner.task_pose_sequences = {}

    with patch('builtins.input', side_effect=['6', '2']):
        assert runner._get_user_task_instruction('unused') == [
            'fold clothes', 'fold clothes'
        ]
    with patch('builtins.input', side_effect=['', '1']):
        with pytest.raises(ValueError, match='Unknown task ID'):
            runner._get_user_task_instruction('unused')


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
