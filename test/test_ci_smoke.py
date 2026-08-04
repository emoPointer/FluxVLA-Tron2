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

import numpy as np
import pytest
from mmengine import Config

from fluxvla.engines.runners.serving.serve import load_deployment_metadata
from fluxvla.engines.runners.tron2_inference_runner import (
    Tron2InferenceRunner, _TerminalKeyReader)

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

    command = runner._wait_for_idle_command(
        _FakeKeyReader(['b', '6', '\n', 'b']))

    assert command == ('start', '6')


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
    monitor_thread = threading.Thread(
        target=runner._monitor_active_keys,
        args=(
            _FakeKeyReader(['r', 'b', 's']),
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
