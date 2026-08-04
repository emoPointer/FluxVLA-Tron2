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
import runpy
from pathlib import Path
from unittest.mock import patch

import pytest
from mmengine import Config

from fluxvla.engines.runners.serving.serve import load_deployment_metadata
from fluxvla.engines.runners.tron2_inference_runner import (
    Tron2InferenceRunner, )

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
    assert operator['max_state_source_mismatch_rad'] == 0.5
    assert operator['lock_head'] is True
    assert operator['max_head_hold_error_rad'] == 0.05


def test_all_tron2_configs_use_tron2_env_operator():
    config_paths = [
        ROOT / 'configs' / 'pi05' /
        'pi05_paligemma_tron2_full_finetune.py',
        ROOT / 'configs' / 'pi05' /
        'pi05_paligemma_tron2_lora_finetune.py',
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

    with pytest.raises(FileNotFoundError,
                       match='checkpoint-local task metadata'):
        load_deployment_metadata(cfg, str(checkpoint))


def test_tron2_runner_rejects_unknown_task_id():
    runner = Tron2InferenceRunner.__new__(Tron2InferenceRunner)
    runner.task_descriptions = {'6': 'fold clothes'}

    assert runner._get_task_description('6') == 'fold clothes'
    with pytest.raises(ValueError, match='Unknown task ID'):
        runner._get_task_description('1')


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
