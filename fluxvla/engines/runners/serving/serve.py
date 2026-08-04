"""Launch a ZMQ server that serves a VLA model for remote inference.

Usage::

    python -m fluxvla.engines.runners.serving.serve \\
        --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \\
        --ckpt-path /path/to/checkpoint.pt \\
        --host 0.0.0.0 --port 5555
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Serve a VLA model via ZMQ for remote inference')
    parser.add_argument(
        '--config', required=True, help='Path to mmengine config file')
    parser.add_argument(
        '--ckpt-path', required=True, help='Path to model checkpoint')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument(
        '--dataset-key',
        default=None,
        choices=['inference', 'eval'],
        help='Config key to load dataset pipeline from')
    return parser.parse_args()


def load_deployment_metadata(cfg, ckpt_path: str) -> dict:
    """Resolve task/action metadata from the checkpoint's saved config.

    An explicit checkpoint-local ``deployment_metadata.json`` takes precedence
    over the saved training config, whose inference section may still contain
    an unrelated example prompt. The saved config remains a compatibility
    fallback for older work directories.
    """
    runtime_inference = getattr(cfg, 'inference', {})
    is_tron2 = runtime_inference.get('type') == 'Tron2InferenceRunner'
    checkpoint_work_dir = Path(ckpt_path).resolve().parent.parent
    metadata = {
        'task_descriptions': dict(runtime_inference.get(
            'task_descriptions', {})),
        'action_layout': runtime_inference.get('action_layout'),
        'checkpoint_work_dir': checkpoint_work_dir.name,
    }

    deployment_metadata_path = checkpoint_work_dir / 'deployment_metadata.json'
    saved_config_path = checkpoint_work_dir / 'config.json'
    if deployment_metadata_path.is_file():
        with deployment_metadata_path.open('r', encoding='utf-8') as file:
            explicit_metadata = json.load(file)
        explicit_tasks = explicit_metadata.get('task_descriptions')
        explicit_layout = explicit_metadata.get('action_layout')
        if not explicit_tasks or not isinstance(explicit_tasks, dict):
            raise ValueError('Checkpoint deployment_metadata.json must define '
                             'a non-empty task_descriptions mapping: '
                             f'{deployment_metadata_path}.')
        if not explicit_layout:
            raise ValueError('Checkpoint deployment_metadata.json must define '
                             f'action_layout: {deployment_metadata_path}.')
        metadata['task_descriptions'] = {
            str(task_id): str(description)
            for task_id, description in explicit_tasks.items()
        }
        metadata['action_layout'] = explicit_layout
        metadata['metadata_source'] = deployment_metadata_path.name
        return metadata

    if saved_config_path.is_file():
        with saved_config_path.open('r', encoding='utf-8') as file:
            saved_config = json.load(file)
        saved_inference = saved_config.get('inference', {})
        saved_tasks = saved_inference.get('task_descriptions')
        if is_tron2 and not saved_tasks:
            raise ValueError('Tron2 checkpoint config must define a non-empty '
                             'inference.task_descriptions mapping: '
                             f'{saved_config_path}.')
        if saved_tasks:
            metadata['task_descriptions'] = {
                str(task_id): str(description)
                for task_id, description in saved_tasks.items()
            }
        if is_tron2 and not saved_inference.get('action_layout'):
            raise ValueError('Tron2 checkpoint config must define '
                             f'inference.action_layout: {saved_config_path}.')
        if saved_inference.get('action_layout'):
            metadata['action_layout'] = saved_inference['action_layout']
        metadata['metadata_source'] = saved_config_path.name
    else:
        if is_tron2:
            raise FileNotFoundError(
                'Tron2 deployment requires checkpoint-local task metadata: '
                f'{saved_config_path} does not exist.')
        metadata['metadata_source'] = 'launch_config'

    if is_tron2 and not metadata['task_descriptions']:
        raise ValueError('Tron2 checkpoint config must define a non-empty '
                         'inference.task_descriptions mapping: '
                         f'{saved_config_path}.')
    if is_tron2 and not metadata['action_layout']:
        raise ValueError('Tron2 checkpoint config must define '
                         f'inference.action_layout: {saved_config_path}.')
    return metadata


def main():
    args = parse_args()

    from mmengine import Config

    from fluxvla.engines import build_vla_from_cfg

    cfg = Config.fromfile(args.config)
    deployment_metadata = load_deployment_metadata(cfg, args.ckpt_path)
    print('[serve] Deployment metadata: '
          f"work_dir={deployment_metadata['checkpoint_work_dir']} "
          f"action_layout={deployment_metadata['action_layout']} "
          f"task_ids={sorted(deployment_metadata['task_descriptions'])}")

    print('[serve] Building VLA model from config ...')
    if hasattr(cfg, 'inference_model'):
        vla = build_vla_from_cfg(cfg.inference_model)
    else:
        vla = build_vla_from_cfg(cfg.model)

    ckpt_path = args.ckpt_path
    assert Path(ckpt_path).exists(), f'Checkpoint not found: {ckpt_path}'
    print(f'[serve] Loading checkpoint: {ckpt_path}')
    if ckpt_path.endswith('.safetensors'):
        checkpoint = load_file(ckpt_path, device='cpu')
    else:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    vla.load_state_dict(state_dict, strict=True)

    data_stat_path = os.path.join(
        Path(ckpt_path).resolve().parent.parent, 'dataset_statistics.json')
    if os.path.isfile(data_stat_path):
        with open(data_stat_path, 'r') as f:
            vla.norm_stats = json.load(f)
        print(f'[serve] Loaded norm_stats from {data_stat_path}')

    from fluxvla.engines import (build_dataset_from_cfg,
                                 build_transform_from_cfg)

    dataset = None
    denormalize_action = None
    task_suite_name = ''
    dataset_key = args.dataset_key
    if dataset_key is None:
        if hasattr(cfg, 'inference') and 'dataset' in cfg.inference:
            dataset_key = 'inference'
        elif hasattr(cfg, 'eval') and 'dataset' in cfg.eval:
            dataset_key = 'eval'

    if dataset_key:
        dataset_cfg = dict(getattr(cfg, dataset_key).dataset)
        if 'norm_stats' not in dataset_cfg:
            dataset_cfg['norm_stats'] = data_stat_path
        ds_type = dataset_cfg.get('type', '')
        if 'model_path' not in dataset_cfg and 'Libero' not in ds_type:
            dataset_cfg['model_path'] = os.path.dirname(
                os.path.dirname(ckpt_path))
        if 'task_suite_name' not in dataset_cfg and 'Libero' in ds_type:
            eval_cfg = getattr(cfg, dataset_key, None)
            if eval_cfg and hasattr(eval_cfg, 'task_suite_name'):
                dataset_cfg['task_suite_name'] = eval_cfg.task_suite_name
        dataset = build_dataset_from_cfg(dataset_cfg)
        print(f'[serve] Dataset pipeline built from '
              f'cfg.{dataset_key}.dataset')

        eval_cfg = getattr(cfg, dataset_key, None)
        if eval_cfg and hasattr(eval_cfg, 'denormalize_action'):
            denorm_cfg = dict(eval_cfg.denormalize_action)
            denorm_cfg['norm_stats'] = data_stat_path
            denormalize_action = build_transform_from_cfg(denorm_cfg)
            print('[serve] Denormalize action transform built')
        if eval_cfg and hasattr(eval_cfg, 'task_suite_name'):
            task_suite_name = eval_cfg.task_suite_name
    else:
        print('[serve] WARNING: No dataset pipeline found in config.')

    dtype_map = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32
    }

    from .zmq_server import create_server

    server = create_server(
        vla=vla,
        dataset=dataset,
        denormalize_action=denormalize_action,
        task_suite_name=task_suite_name,
        host=args.host,
        port=args.port,
        device=args.device,
        mixed_precision_dtype=dtype_map[args.dtype],
        deployment_metadata=deployment_metadata,
    )
    print(f'[serve] ZMQ server starting on tcp://{args.host}:{args.port}')
    try:
        server.run()
    except KeyboardInterrupt:
        server.close()
        print('[serve] Server stopped.')


if __name__ == '__main__':
    main()
