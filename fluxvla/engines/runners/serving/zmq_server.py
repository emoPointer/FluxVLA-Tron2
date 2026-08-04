"""ZMQ VLA inference server (2-layer architecture).

Layer 1: PolicyServer -- generic ZMQ REP event loop + endpoint routing.
Layer 2: create_server -- factory that wires a VLA model into the server.

Usage::

    python -m fluxvla.engines.runners.serving.serve \\
        --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \\
        --ckpt-path /path/to/checkpoint.pt
"""
from __future__ import annotations
import io
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import zmq

from .serializers import (FORMAT_PROTOBUF, MsgSerializer, ObsSerializer,
                          decode_predict_request, detect_format,
                          encode_predict_response)


def serialize_actions(actions: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    np.save(buf, actions.cpu().numpy(), allow_pickle=False)
    return buf.getvalue()


def prepare_remote_rtc_inputs(rtc: Optional[dict],
                              reference: torch.Tensor,
                              max_action_steps: int,
                              expected_action_dim: int = None) -> dict:
    """Validate a stateless remote-RTC payload and create model inputs."""
    if rtc is None:
        return {}
    if not isinstance(rtc, dict):
        raise ValueError('Remote RTC payload must be a mapping.')
    unknown = set(rtc) - {'prev_actions', 'prefix_len', 'config'}
    if unknown:
        raise ValueError(f'Unknown remote RTC fields: {sorted(unknown)}.')

    prev_actions = np.asarray(rtc.get('prev_actions'))
    if (prev_actions.ndim != 2 or prev_actions.shape[0] == 0
            or prev_actions.shape[1] == 0):
        raise ValueError('Remote RTC prev_actions must have shape [T, D] with '
                         f'non-zero dimensions; got {prev_actions.shape}.')
    if prev_actions.shape[0] > max_action_steps:
        raise ValueError('Remote RTC prev_actions exceeds the model action '
                         f'horizon: {prev_actions.shape[0]} > '
                         f'{max_action_steps}.')
    if (expected_action_dim is not None
            and prev_actions.shape[1] != expected_action_dim):
        raise ValueError('Remote RTC prev_actions action dimension does not '
                         f'match the model: {prev_actions.shape[1]} != '
                         f'{expected_action_dim}.')
    if not np.issubdtype(prev_actions.dtype, np.floating):
        raise ValueError('Remote RTC prev_actions must use a floating dtype; '
                         f'got {prev_actions.dtype}.')
    if not np.all(np.isfinite(prev_actions)):
        raise ValueError('Remote RTC prev_actions contains non-finite values.')

    prefix_len = rtc.get('prefix_len')
    if isinstance(prefix_len, bool) or not isinstance(prefix_len, int):
        raise ValueError('Remote RTC prefix_len must be an integer.')
    if not 0 < prefix_len <= prev_actions.shape[0]:
        raise ValueError('Remote RTC prefix_len must satisfy '
                         f'0 < prefix_len <= {prev_actions.shape[0]}; got '
                         f'{prefix_len}.')

    config = rtc.get('config')
    if not isinstance(config, dict):
        raise ValueError('Remote RTC config must be a mapping.')
    allowed_config = {
        'enabled', 'method', 'decay_end', 'schedule', 'max_guidance_weight',
        'use_vjp'
    }
    unknown_config = set(config) - allowed_config
    if unknown_config:
        raise ValueError('Unknown remote RTC config fields: '
                         f'{sorted(unknown_config)}.')
    if config.get('enabled') is not True:
        raise ValueError('Remote RTC config.enabled must be true.')
    method = config.get('method')
    if method not in {'prefix', 'guidance'}:
        raise ValueError("Remote RTC method must be 'prefix' or 'guidance'; "
                         f'got {method!r}.')

    model_config = {'method': method}
    if method == 'guidance':
        decay_end = config.get('decay_end')
        if (isinstance(decay_end, bool) or not isinstance(decay_end, int)
                or not prefix_len <= decay_end <= max_action_steps):
            raise ValueError('Remote RTC guidance decay_end must satisfy '
                             f'{prefix_len} <= decay_end <= '
                             f'{max_action_steps}; got {decay_end!r}.')
        schedule = config.get('schedule')
        if schedule not in {'exp', 'linear', 'ones', 'zeros'}:
            raise ValueError('Remote RTC guidance schedule must be one of '
                             f"exp/linear/ones/zeros; got {schedule!r}.")
        max_guidance_weight = config.get('max_guidance_weight')
        if (isinstance(max_guidance_weight, bool)
                or not isinstance(max_guidance_weight, (int, float))
                or not np.isfinite(max_guidance_weight)
                or not 0.0 < max_guidance_weight <= 100.0):
            raise ValueError('Remote RTC max_guidance_weight must be finite '
                             f'and within (0, 100]; got '
                             f'{max_guidance_weight!r}.')
        use_vjp = config.get('use_vjp')
        if not isinstance(use_vjp, bool):
            raise ValueError('Remote RTC use_vjp must be boolean.')
        model_config.update({
            'decay_end': decay_end,
            'schedule': schedule,
            'max_guidance_weight': float(max_guidance_weight),
            'use_vjp': use_vjp,
        })

    previous_action_tensor = torch.from_numpy(prev_actions.copy())[None].to(
        device=reference.device, dtype=reference.dtype)
    return {
        'prev_actions': previous_action_tensor,
        'prefix_len': prefix_len,
        'rtc_config': model_config,
    }


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class PolicyServer:
    """Generic ZMQ REP server with named endpoint routing.

    Provides a synchronous request-reply event loop over a ZMQ REP socket.
    Endpoints are registered by name; incoming messages are dispatched to
    the matching handler.  Two wire formats are supported:

    - msgpack (default): ``{"endpoint": "<name>", "data": {...}}``
    - protobuf: first byte ``0x01`` triggers the protobuf predict path.

    Built-in endpoints:

    - ping -- health check, returns ``{"status": "ok"}``.
    - kill -- graceful shutdown.

    Attributes:
        running: Flag controlling the event loop; set to ``False`` to stop.
        context: The underlying ``zmq.Context``.
        socket: The bound ``zmq.REP`` socket.
    """

    def __init__(self, host: str = '*', port: int = 5555):
        """Create and bind the ZMQ REP socket.

        Args:
            host: Bind address (``'*'`` for all interfaces).
            port: TCP port to listen on.
        """
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f'tcp://{host}:{port}')
        self._endpoints: dict[str, EndpointHandler] = {}

        self.register_endpoint('ping', self._handle_ping, requires_input=False)
        self.register_endpoint('kill', self._kill_server, requires_input=False)

    def register_endpoint(self,
                          name: str,
                          handler: Callable,
                          requires_input: bool = True):
        """Register a named endpoint handler.

        Args:
            name: Endpoint name used in the ``"endpoint"`` field of
                incoming msgpack messages.
            handler: Callable invoked when this endpoint is requested.
                If *requires_input* is ``True``, the ``"data"`` dict from
                the request is unpacked as keyword arguments.
            requires_input: Whether the handler expects input data.
                ``False`` for no-arg endpoints like ``ping``.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _handle_ping(self) -> dict:
        return {'status': 'ok', 'message': 'Server is running'}

    def _kill_server(self):
        self.running = False
        return {'status': 'ok', 'message': 'Server shutting down'}

    def run(self):
        """Start the blocking event loop.

        Polls the ZMQ socket every 500 ms. Each incoming message is
        decoded, dispatched to the registered endpoint handler, and the
        result is serialized back. The loop exits when ``self.running``
        becomes ``False`` (via ``kill`` endpoint or ``close()``).
        """
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f'Server is ready and listening on {addr}')
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while self.running:
            try:
                socks = dict(poller.poll(timeout=500))
                if self.socket not in socks:
                    continue

                message = self.socket.recv()

                if detect_format(message) == FORMAT_PROTOBUF:
                    self._handle_protobuf_predict(message)
                    continue

                request = MsgSerializer.from_bytes(message)
                endpoint = request.get('endpoint', 'predict_action')
                if endpoint not in self._endpoints:
                    raise ValueError(f'Unknown endpoint: {endpoint}')

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(**request.get('data', {}))
                    if handler.requires_input else handler.handler())
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as e:
                print(f'Error in server: {e}')
                self.socket.send(MsgSerializer.to_bytes({'error': str(e)}))

        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()
        self.context.term()

    def _handle_protobuf_predict(self, message: bytes):
        """Decode a protobuf ``PredictActionRequest`` and reply.

        Args:
            message: Raw bytes whose first byte is ``FORMAT_PROTOBUF``.
        """
        try:
            _, obs, unnorm_key = decode_predict_request(message)
            handler = self._endpoints.get('predict_action')
            if handler is None:
                resp = encode_predict_response(
                    b'',
                    0.0,
                    FORMAT_PROTOBUF,
                    error='predict_action endpoint not registered')
            else:
                result = handler.handler(
                    obs_data=None,
                    unnorm_key=unnorm_key,
                    _obs_dict=obs,
                    _wire_format=FORMAT_PROTOBUF)
                resp = encode_predict_response(
                    result.get('action_data', b''),
                    result.get('infer_time', 0.0),
                    FORMAT_PROTOBUF,
                    error=result.get('error', ''))
            self.socket.send(resp)
        except Exception as e:
            print(f'Error in protobuf handler: {e}')
            self.socket.send(
                encode_predict_response(
                    b'', 0.0, FORMAT_PROTOBUF, error=str(e)))

    def close(self):
        """Signal the event loop to stop.

        The socket is closed and the ZMQ context terminated when the
        ``run()`` loop finishes its current iteration.
        """
        self.running = False


def create_server(
    vla,
    dataset=None,
    denormalize_action=None,
    task_suite_name: str = '',
    host: str = '*',
    port: int = 5555,
    device: str = 'cuda:0',
    mixed_precision_dtype=torch.bfloat16,
    deployment_metadata: Optional[dict] = None,
) -> PolicyServer:
    """Create a ZMQ server that wraps a VLA model.

    Args:
        vla: VLA model (already loaded with weights).
        dataset: Optional dataset transform pipeline for preprocessing.
        denormalize_action: Optional denormalization transform.
        task_suite_name: Task suite name for denormalization lookup.
        host: Bind address.
        port: Bind port.
        device: CUDA device.
        mixed_precision_dtype: Dtype for autocast.
        deployment_metadata: Checkpoint-specific task/action metadata exposed
            read-only to remote clients.
    """
    torch_device = torch.device(device)
    vla.eval()
    vla.to(torch_device)

    lock = threading.Lock()
    total_requests = 0
    total_infer_time = 0.0
    start_time = time.time()

    def predict_action(obs_data: bytes = None,
                       unnorm_key: str = '',
                       rtc: dict = None,
                       _obs_dict: dict = None,
                       _wire_format: int = 0) -> dict:
        nonlocal total_requests, total_infer_time

        obs = _obs_dict if _obs_dict is not None else \
            ObsSerializer.from_bytes(obs_data)

        if dataset is not None:
            result = dataset(obs)
            batch = result[0] if isinstance(result, tuple) else result
        else:
            batch = obs
        if unnorm_key:
            batch['unnorm_key'] = unnorm_key

        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(
                torch_device.type,
                dtype=mixed_precision_dtype,
                enabled=torch_device.type == 'cuda'):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(torch_device)
            if rtc is not None:
                reference = batch.get('states')
                if not isinstance(reference, torch.Tensor):
                    reference = next((value for value in batch.values()
                                      if isinstance(value, torch.Tensor)),
                                     None)
                if reference is None:
                    raise ValueError('Remote RTC requires at least one tensor '
                                     'model input to determine device/dtype.')
                max_action_steps = int(getattr(vla, 'n_action_steps', 0))
                if max_action_steps <= 0:
                    raise ValueError('Remote RTC requires the model to expose '
                                     'a positive n_action_steps value.')
                expected_action_dim = getattr(vla, 'ori_action_dim', None)
                if expected_action_dim is not None:
                    expected_action_dim = int(expected_action_dim)
                batch.update(
                    prepare_remote_rtc_inputs(rtc, reference, max_action_steps,
                                              expected_action_dim))
            actions = vla.predict_action(**batch)
        infer_time = time.perf_counter() - t0

        raw_action_dim = getattr(vla, 'ori_action_dim', None)
        if raw_action_dim is None:
            raw_action_dim = actions.shape[-1]
        raw_action_dim = int(raw_action_dim)
        if not 0 < raw_action_dim <= actions.shape[-1]:
            raise ValueError('Model ori_action_dim is incompatible with its '
                             f'output: {raw_action_dim} vs '
                             f'{actions.shape[-1]}.')
        raw_action_bytes = serialize_actions(
            actions[..., :raw_action_dim].detach().float())

        if denormalize_action is not None:
            actions_np = actions.cpu().numpy()
            d = denormalize_action(
                dict(action=actions_np[0], task_suite_name=task_suite_name))
            actions = torch.from_numpy(d[None].astype(np.float32))

        action_bytes = serialize_actions(actions)

        with lock:
            total_requests += 1
            total_infer_time += infer_time
            n = total_requests
            should_print = (n % 50 == 0)
            avg = total_infer_time / n if should_print else 0.0

        if should_print:
            print(
                f'[VLAServer] req={n}  '
                f'infer={infer_time*1000:.1f}ms  '
                f'avg_infer={avg*1000:.1f}ms',
                flush=True)

        return {
            'action_data': action_bytes,
            'raw_action_data': raw_action_bytes,
            'infer_time': infer_time,
        }

    def reset() -> dict:
        return {'status': 'ok'}

    def get_status() -> dict:
        with lock:
            n = total_requests
            avg = (total_infer_time / n) if n > 0 else 0.0
        return {
            'status': 'ready',
            'uptime_s': time.time() - start_time,
            'total_requests': n,
            'avg_infer_time': avg,
        }

    def get_deployment_metadata() -> dict:
        metadata = dict(deployment_metadata or {})
        metadata['remote_rtc'] = {
            'wire_format': 'msgpack',
            'methods': ['guidance', 'prefix'],
            'returns_raw_actions': True,
        }
        return metadata

    server = PolicyServer(host=host, port=port)
    server.register_endpoint('predict_action', predict_action)
    server.register_endpoint('reset', reset, requires_input=False)
    server.register_endpoint('get_status', get_status, requires_input=False)
    server.register_endpoint(
        'get_deployment_metadata',
        get_deployment_metadata,
        requires_input=False,
    )
    return server
