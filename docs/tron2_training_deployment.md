# Training and Deploying FluxVLA on a New Tron2

This document describes the recommended workflow for fine-tuning PI0.5 with
custom Tron2 data and deploying it with remote inference.  It is written for
the common setup where a GPU workstation/server runs the VLA model, while the
Tron2 robot computer consumes TRON2 Bridge WebSocket observations and sends
WebSocket control commands to the robot.

The current Tron2 PI0.5 LoRA config is:

```text
configs/pi05/pi05_paligemma_tron2_lora_finetune.py
```

The current deployment path is:

```text
TRON2 Bridge WebSocket -> robot-side FluxVLA client -> SSH tunnel -> GPU server ZMQ
    -> PI0.5 policy inference -> action returned to robot client
    -> Tron2 WebSocket control service
```

## 1. Hardware and Network Assumptions

This guide assumes three logical components:

| Component               | Role                                                     |
| ----------------------- | -------------------------------------------------------- |
| GPU server              | Fine-tunes PI0.5 and serves the model through ZMQ        |
| Power Computing Module  | External Tron2 computer; runs TRON2 Bridge and FluxVLA client |
| Internal robot computer | Internal controller; not exposed directly to users       |

Tron2 has two computers.  The internal robot computer is inside the robot and
is not normally exposed to users.  The user-facing computer is the external
Power Computing Module, which is usually mounted outside the robot and usually
has IP address `10.192.1.4`.

In our current deployment, the GPU server is reachable from the Power Computing
Module only through SSH.  The robot client therefore opens an SSH local port
forward:

```text
Power Computing Module localhost:5555 -> GPU server 127.0.0.1:3333
```

The observation and control WebSockets are separate from remote inference:

```text
wss://10.192.1.4/bridge/ws        # aligned images, joints, and grippers
ws://10.192.1.2:5000              # robot state and MoveJ/ServoJ control
```

The robot control WebSocket is used only when actions are executed:

```text
ws://<TRON2_CONTROLLER_IP>:<TRON2_WS_PORT>
```

In most Tron2 deployments, the WebSocket controller IP is `10.192.1.2` and the
port is `5000`:

```text
TRON2_CONTROLLER_IP = 10.192.1.2
TRON2_WS_PORT       = 5000
```

For a new robot, `robot_ip` usually stays `10.192.1.2`. The public
[`tron2_env`](https://github.com/limxdynamics/tron2_env) transport detects the
controller account ID from server messages, so `ws_accid` must remain `None`.

Important: remote inference requires the Power Computing Module to reach the
GPU server.  To connect the robot to the public network, either configure Wi-Fi
from the `10.192.1.2:8080` web interface, or connect an Ethernet cable to any
ETH port on the Power Computing Module.

## 2. Prepare the GPU Server

Create the full FluxVLA training environment on the GPU server.  Follow the
main installation guide in `README.md`; the environment name used here is
`fluxvla`.

```bash
conda activate fluxvla
cd /path/to/FluxVLA
```

The PI0.5 base checkpoint should be available at:

```text
checkpoints/pi05_base/
```

The config expects at least:

```text
checkpoints/pi05_base/model.safetensors
checkpoints/pi05_base/tokenizer_config.json
checkpoints/pi05_base/tokenizer.model or tokenizer.json
```

Do not point the tokenizer to a training output directory unless that directory
contains a complete Hugging Face tokenizer.  For the Tron2 LoRA config, keep
the tokenizer path as:

```python
model_path='checkpoints/pi05_base'
```

### Optional Environment File

The repository provides `.env.example` as a safe environment-variable template.
It contains placeholders and public defaults only; do not put real tokens,
robot account IDs, private hosts, or credentials into files committed to Git.

FluxVLA does not automatically load `.env`.  If you keep a private local `.env`
file, load it explicitly in the shell before running training or inference:

```bash
set -a
source .env
set +a
```

## 3. Prepare the Dataset

Data can be exported directly from the cloud data platform.  Ask the delivery
engineer for the cloud data platform login URL and account credentials.

Place the LeRobot-style Tron2 dataset under `datasets/`.  The default config
uses:

```text
datasets/lerobot_dataset
```

If your dataset path is different, update:

```python
train_dataloader.dataset.datasets[0].data_root_path
```

in:

```text
configs/pi05/pi05_paligemma_tron2_lora_finetune.py
```

The current config expects these data fields:

| Field                                | Meaning                |
| ------------------------------------ | ---------------------- |
| `observation.state`                  | robot proprioception   |
| `action` / `actions`                 | action sequence        |
| `observation.images.cam_high`        | top camera RGB image   |
| `observation.images.cam_left_wrist`  | left camera RGB image  |
| `observation.images.cam_right_wrist` | right camera RGB image |

The current action layout is 16-dimensional:

```text
left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)
```

The model still pads state/action tensors to 32 dimensions internally for
PI0.5 compatibility, but denormalization and execution use the first 16 action
dimensions.

Before training on a new dataset, check at least:

```bash
ls datasets/lerobot_dataset
```

and verify a few episodes manually.  In particular, confirm:

- the camera names match the config;
- the action dimension is 16;
- the gripper convention matches training and execution;
- the task language is correct;
- failed or abnormal episodes are removed.

## 4. Update the Tron2 Config

Start from:

```text
configs/pi05/pi05_paligemma_tron2_lora_finetune.py
```

For a new Tron2, update these values.

### Dataset path

```python
data_root_path=[
    './datasets/lerobot_dataset',
]
```

### Task descriptions

```python
task_descriptions={
    '1': 'Pick up the banana from the desk and place it on the plate',
}
```

The interactive client asks for a task ID.  If the user enters `1`, this
description is sent to the model.

### TRON2 Bridge WebSocket observations

The FluxVLA client does not subscribe to ROS. It receives synchronized images,
joints, and grippers from the public `tron2_env` Bridge provider:

```python
operator=dict(
    type='Tron2EnvOperator',
    bridge_host='wss://10.192.1.4',
    bridge_ws_path='/bridge/ws',
    bridge_image_topics=dict(
        camera_left='/camera/left/color/image_resized/compressed',
        camera_right='/camera/right/color/image_resized/compressed',
        camera_top='/camera/top/color/image_raw/compressed',
    ),
    bridge_joint_topics=dict(
        joint_states='/joint_states',
        gripper='/gripper_state',
    ),
    bridge_verify_tls=False,
    robot_ip='10.192.1.2',
    ws_port=5000,
    ws_accid=None,
    movej_duration=2.0,
    servoj_publish_rate=300.0,
    recovery_blend_frames=6,
    chunk_boundary_blend_enabled=False,
    chunk_boundary_blend_frames=6,
    chunk_boundary_blend_scope='arm',
    max_servoj_step_rad=0.5,
    max_state_source_mismatch_rad=None,
    lock_head=True,
    max_head_hold_error_rad=0.05,
)
```

This deployment uses a self-signed Bridge certificate, so
`bridge_verify_tls=False` is allowed only on the isolated robot LAN. Update the
explicit mappings if the Bridge publishes different topic paths.

### WebSocket control parameters

The WebSocket controller IP is generally `10.192.1.2`, so it usually does not
need to be changed:

```python
robot_ip='10.192.1.2'
ws_port=5000
ws_accid=None
```

`Tron2EnvOperator` uses the public `tron2_env` runtime, which auto-detects the
account ID. Do not set `ws_accid` to a non-`None` value. Bridge observations
use `wss://10.192.1.4`; prepare poses use MoveJ, while policy actions use the
separate robot WebSocket and a measured-state-seeded 300 Hz ServoJ publisher.
Policy waypoints are fed at `publish_rate=30` Hz. If the previous ServoJ
trajectory drains and the publisher is holding its final target, the next
valid trajectory blends from that held command for
`recovery_blend_frames=6` policy frames (about 0.2 seconds at 30 Hz). Normal
asynchronous replacement of a trajectory that is still active does not
trigger recovery blending. The separate
optional `chunk_boundary_blend_enabled=True` path snapshots the old
trajectory's unissued actions and smoothstep-blends them with the first six
aligned actions of the replacement chunk. It is currently disabled with
`chunk_boundary_blend_enabled=False`; the separate six-frame recovery blend
remains enabled. When boundary blending is enabled, the current `arm` scope
leaves gripper commands unfiltered and does not change the locked head. MoveJ
reset clears the saved state for both blend paths.

## 5. Fine-Tune PI0.5 with LoRA

The current LoRA settings are defined in the model config:

```python
use_lora=True
lora_rank=256
lora_alpha=512
lora_dropout=0.0
modules_to_save=[
    'action_in_proj',
    'action_out_proj',
    'time_mlp_in',
    'time_mlp_out',
]
```

The current training schedule is:

```python
runner.max_steps=30000
train_dataloader.per_device_batch_size=16
```

With 4 GPUs, this gives a global batch size of 64 when gradient accumulation is
1:

```text
16 per GPU x 4 GPUs = 64
```

Example training command on GPUs 0, 1, 2, 3:

```bash
conda activate fluxvla
cd /path/to/FluxVLA

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=fluxvla-tron2

NPROC_PER_NODE=4 bash scripts/train.sh \
  configs/pi05/pi05_paligemma_tron2_lora_finetune.py \
  work_dirs/pi05_paligemma_tron2_lora_finetune \
  --cfg-options \
  train_dataloader.per_device_batch_size=16 \
  runner.max_steps=30000
```

If you want to resume training:

```bash
NPROC_PER_NODE=4 bash scripts/train.sh \
  configs/pi05/pi05_paligemma_tron2_lora_finetune.py \
  work_dirs/pi05_paligemma_tron2_lora_finetune \
  --resume-from /path/to/checkpoint.pt
```

Training outputs are written to:

```text
work_dirs/pi05_paligemma_tron2_lora_finetune/
```

The server should normally use the `.safetensors` checkpoint for inference if
available, for example:

```text
work_dirs/pi05_paligemma_tron2_lora_finetune/checkpoints/step-XXXXX.safetensors
```

The training script also saves dataset statistics under the work directory.
Remote inference uses these statistics for action denormalization.

## 6. Start the Remote Inference Server

Run this on the GPU server.

If the robot connects through SSH tunnel only, bind the ZMQ server to
`127.0.0.1`:

```bash
conda activate fluxvla
cd /path/to/FluxVLA

python -m fluxvla.engines.runners.serving.serve \
  --config configs/pi05/pi05_paligemma_tron2_lora_finetune.py \
  --ckpt-path work_dirs/pi05_paligemma_tron2_lora_finetune/checkpoints/step-XXXXX.safetensors \
  --host 127.0.0.1 \
  --port 3333 \
  --device cuda:0 \
  --dtype bf16
```

If the robot can reach the server directly on the LAN, bind to `0.0.0.0` or
the server LAN IP and update the client config accordingly.

The server loads:

1. the PI0.5 model;
2. the LoRA checkpoint;
3. the inference dataset preprocessing pipeline;
4. the denormalization transform;
5. `dataset_statistics.json` from the training work directory.

The server first looks for `deployment_metadata.json` in the selected
checkpoint work directory. If absent, it loads `inference.task_descriptions`
and `action_layout` from the saved `config.json`. Use the explicit sidecar when
the training config contains a stale example inference prompt. The robot client
lists the resolved task IDs during startup and rejects unknown IDs. When
switching checkpoints or metadata, restart the server; no task-specific
robot-side code change is required.

## 7. Prepare the Power Computing Module

The Power Computing Module does not need to install the full CUDA training
stack.  It should run the lightweight remote client path.

Install Miniconda if needed:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Create the client environment:

```bash
conda create -n fluxvla python=3.10 -y
conda activate fluxvla
```

Install the Python packages needed by the client:

```bash
pip install mmengine pyzmq msgpack numpy safetensors websocket-client
pip install "tron2-env[bridge] @ git+https://github.com/limxdynamics/tron2_env.git@5b7b145229416f3731f61657e6fa71c89c37bc9d"
```

The FluxVLA client does not require a ROS environment. The Bridge service may
use ROS internally, outside the client process.

Do not require `pip install -e .` on the Power Computing Module.  The remote
client script sets:

```bash
FLUXVLA_REMOTE_CLIENT_ONLY=1
PYTHONPATH="$(pwd):${PYTHONPATH}"
```

This avoids importing the full model/CUDA stack on the robot.

## 8. Verify Bridge and Robot-Control WebSockets

Check both endpoints:

```bash
python -c "import socket; socket.create_connection(('10.192.1.4', 443), timeout=3); print('bridge tcp ok')"
python -c "import socket; socket.create_connection(('10.192.1.2', 5000), timeout=3); print('control tcp ok')"
```

If the control port is unstable, fix the Tron2 control service before running
`dry_run=False`. Dry run still requires the Bridge observation WebSocket, but
does not require the robot-control WebSocket.

## 9. Run Remote Inference in Dry Run Mode

The checked-in private deployment config currently enables real actions:

```python
dry_run=False
```

For dry-run validation, explicitly override it with
`inference.dry_run=True`. Dry run performs the full perception and remote
inference flow but does not execute actions:

```text
Bridge WebSocket observations -> SSH tunnel -> GPU inference -> action returned -> print
```

Run this on the Power Computing Module:

```bash
cd ~/FluxVLA
conda activate fluxvla

bash scripts/remote_inference_client.sh \
  configs/pi05/pi05_paligemma_tron2_lora_finetune.py \
  --ssh-host USER@SERVER_PUBLIC_IP \
  --ssh-port 22 \
  --local-port 5555 \
  --remote-port 3333 \
  --cfg-options inference.dry_run=True
```

If the server is reachable only through port 22, keep this SSH tunnel mode.
If SSH key login is configured, the command will not ask for a password.

During dry run, the client first prints the task IDs advertised by the active
checkpoint. Type one of those IDs, press Enter to confirm it, and press `b` to
start. Press `s` to stop generating and accepting further chunks:

```text
Task ID: 6
Press b to start, or type another task ID and press Enter.
```

There is no repeat-count prompt. The keyboard state machine runs on the Power
Computing Module; the GPU server only handles prediction requests. In dry-run
mode, pressing `r` while idle prints and skips the prepare-pose command.

Expected dry-run output includes a printed action:

```text
[Tron2InferenceRunner] dry_run=True, skip execution.
```

## 10. Execute on the Robot

Only run real execution after:

- dry run succeeds;
- Bridge observations are stable;
- WebSocket handshake succeeds;
- a physical emergency stop is available.

Start the standard non-RTC client:

```bash
bash scripts/remote_inference_client.sh \
  configs/pi05/pi05_paligemma_tron2_lora_finetune.py \
  --ssh-host USER@SERVER_PUBLIC_IP \
  --ssh-port 22 \
  --local-port 5555 \
  --remote-port 3333 \
  --cfg-options inference.dry_run=False
```

The current config uses:

```python
action_chunk=32
```

The client executes one complete 32-step chunk synchronously, then captures a
new observation and requests the next chunk. This path has no overlap queue,
guidance prefix, or inference-time RTC. Pressing `s` discards a prediction that
has not yet been accepted; an already accepted chunk is allowed to finish, and
no subsequent prediction is sent to the controller. A new run requires a new
task-ID selection followed by `b`.

### Single-process prefix RTC

Train-time RTC and inference-time RTC are two stages of one deployment method.
The training config teaches the model to continue after a known clean action
prefix; the runtime runner supplies the unconsumed previous chunk as that
prefix. An RTC-trained checkpoint can run without runtime conditioning, but it
then behaves as an ordinary chunk policy and does not use the trained RTC path.

To keep the model and RTC state in one process, run the following on the GPU
computer. Do not start the ZMQ server, SSH tunnel, or remote client:

```bash
python scripts/inference.py \
  --config configs/pi05/pi05_paligemma_tron2_lora_rtc_local_inference.py \
  --ckpt-path /path/to/merged-rtc-checkpoint.safetensors
```

This config uses `Tron2RTCInferenceRunner`, `method='prefix'`, and a 50-frame
chunk matching the model horizon. `delay=6` seeds the first request. Subsequent
dynamic delays follow the public client's measured latency converted with
`ceil(latency / policy_period)` and the recent ten-sample P95. The selected
fold-clothes checkpoint supports prefixes `{0, 5, 10, 19}`; the FluxVLA
adapter rounds each dynamic delay upward to the first supported value and
clamps values above 19 with a warning.

Execution follows the public TRON2 ActionQueue client. A persistent 30 Hz
consumer starts the next inference when the queue reaches
`H - execution_horizon` actions. The consumer continues issuing the old queue
during inference. When the new chunk returns, the runner atomically replaces
the queue with `new_actions[actual_consumed:]`, where `actual_consumed` is the
consumer-index difference observed while inference was in flight. The 30 Hz
consumer and the 300 Hz `MotionController` are not restarted at chunk
boundaries. The config constructs the pinned public `Tron2Env` directly, so
Bridge images, robot-WebSocket state, `step()` action extraction, gripper
clipping, current-head passthrough, interpolation, and ServoJ publication are
owned by upstream code. If the queue drains, the latest ServoJ target is held;
recovery uses the public client's six-frame blend.

The runner uses direct Bridge/control WebSocket connections. The
`b`/`s`/`r`/`l` keyboard state machine runs in this GPU-computer terminal.
After `s`, producer and consumer stop with the public-client semantics and the
300 Hz controller holds its latest target. Idle-only `r` disconnects that
controller before MoveJ, so the two command modes cannot race.

The checkpoint training range is `[0, 20)`. The runner logs once if the recent
P95 reaches 20 frames or more; queue replacement still follows the actual
consumer index.

### Issued-action recording

Both the RTC and non-RTC runners accept `l` while idle or running. The first
press starts a JSONL session and the next press flushes and closes it. The
default destination is
`work_dirs/action_records/tron2_actions_<rtc|non_rtc>_<timestamp>_<pid>.jsonl`.
In remote non-RTC mode this path is on the robot-side client; in single-process
RTC mode it is on the GPU computer.

Each action row contains issue timestamps, task ID, instruction, trajectory
and frame indices, RTC state, the effective prefix length, and the complete
action vector. The callback is placed after the validated policy waypoint is
accepted by the ServoJ controller, so preempted old-chunk tails are excluded.
These are 30 Hz policy-rate waypoints after configured blending, not the 300 Hz
interpolated publisher samples. A background writer keeps filesystem I/O out
of the ServoJ feeder. Dry run produces no action rows because it sends no robot
commands. RTC rows identify the ActionQueue scheduler, fixed model prefix,
source action index, and queue size after each issued action.

## 11. Move to the Prepare Pose

While the client is idle, press `r` to run the same configured prepare-pose
sequence previously exposed as task ID `0`:

```text
[TRON2 client idle] Type task ID and press Enter. b=start, r=prepare pose, l=toggle action recording.
```

`r` is ignored while a task is running; press `s`, wait until the client
reports idle, and only then press `r`. In dry-run mode, prepare-pose execution
is skipped. Native reset opens both grippers through the still-connected old
environment and waits 0.5 seconds before disconnecting ServoJ and starting
MoveJ; failure to open the grippers blocks the reset motion. In real execution mode,
each prepare pose uses MoveJ without a head target. Before the first policy
action, the complete chunk is checked against both Bridge and control feedback;
only then may a fresh `tron2_env` MotionController stream ServoJ at 300 Hz.
Policy head trajectories are rejected and the measured head position is held.
If prepare is requested again, the ServoJ publisher is disconnected before any
new MoveJ command is sent.

## 12. Important Safety Notes

`Ctrl+C` runs client cleanup and disconnects the
ServoJ publisher and control WebSocket. It is still not a robot emergency stop:
it does not disable torque or guarantee that a command already accepted by the
Tron2 controller is canceled.

For first deployment on a new robot:

- keep one operator near the physical emergency stop;
- verify the selected checkpoint task ID before pressing `b`;
- keep object placement conservative;
- verify gripper open/close convention before full-speed runs;
- the current PI0.5 LoRA deployment uses `max_servoj_step_rad=0.5`; this only
  relaxes the adjacent-target delta guard and does not disable finite-value,
  joint-limit, head-lock, or ServoJ interpolation checks.

## 13. Troubleshooting

### `ModuleNotFoundError: No module named 'fluxvla'`

Run the client through:

```bash
bash scripts/remote_inference_client.sh ...
```

The script sets `PYTHONPATH` automatically.  Avoid installing the full editable
package on the Power Computing Module unless the machine has a compatible
compiler/CUDA environment.

### `ModuleNotFoundError: No module named 'websockets'`

Install the official Bridge extra in the lightweight client environment:

```bash
pip install "tron2-env[bridge] @ git+https://github.com/limxdynamics/tron2_env.git@5b7b145229416f3731f61657e6fa71c89c37bc9d"
```

### `TRON2 Bridge did not provide one complete ... observation`

Check the Bridge endpoint, configured five topic paths, camera services, and
Bridge logs. The client does not reuse stale or partial observations.

### `Cannot reach VLA server at tcp://127.0.0.1:5555`

The SSH tunnel or GPU server is not running.  Check:

```bash
ssh -p 22 -L 5555:127.0.0.1:3333 USER@SERVER_PUBLIC_IP -N
```

and confirm the server is listening on port 3333.

### `Connection refused` for `ws://10.192.1.2:5000`

This is the Tron2 control WebSocket, not the FluxVLA remote inference server.
Check that the Tron2 control service is running and that the IP/port are
correct:

```bash
python -c "import socket; socket.create_connection(('10.192.1.2', 5000), timeout=3); print('tcp port ok')"
```

If the port is unstable, fix or restart the Tron2 control service before
running with `inference.dry_run=False`.

If the WebSocket cannot connect stably, check whether the real robot is in
high-level dev mode.

### WebSocket connects but commands do not work

Check:

- the controller emits an account ID that `tron2_env` can auto-detect;
- no other official control UI is holding an exclusive connection;
- the robot is in the correct external-control/API-control mode;
- the command path matches the robot controller's expected WebSocket API.

## 14. Files Changed for Tron2 Deployment

The current Tron2 deployment branch changes these main files:

```text
configs/pi05/pi05_paligemma_tron2_lora_finetune.py
fluxvla/engines/runners/tron2_inference_runner.py
fluxvla/engines/operators/tron2_env_operator.py
fluxvla/engines/operators/tron2_operator.py
fluxvla/engines/runners/base_inference_runner.py
fluxvla/transforms/normalize.py
scripts/remote_inference_client.sh
```

The most important runtime switches are:

| Option                        | Meaning                                  |
| ----------------------------- | ---------------------------------------- |
| `inference.dry_run=True`      | full inference flow, no robot action     |
| `inference.dry_run=False`     | execute returned actions on the robot    |
| `inference.action_record_dir` | JSONL directory toggled by the `l` key   |
| `inference.include_head_in_state` | keep the training-time 18-D measured proprio while commands remain 16-D |
| `inference.rtc_config.delay` | initial RTC delay; subsequent model prefixes use measured P95 |
| `inference.rtc_config.execution_horizon` | fixed `s`; starts inference when queue size reaches `H-s` |
| `inference.rtc_config.recovery_blend_frames` | ActionQueue underflow-recovery blend frames |
| `inference.rtc_config.prefix_len` | fixed trained model prefix; ActionQueue still crops by actual consumption |
| `inference.rtc_config.prefix_action_dim` | feed back the 16 supervised action dimensions before appending measured head conditioning |
| `inference.rtc_config.prefix_head_from_observation` | normalize the measured locked head with checkpoint action statistics for dimensions 16--17 |
| `inference.rtc_config.action_postprocess.enabled` | optional post-merge boundary/EMA smoothing (disabled by default) |
| `inference.operator.bridge_host` | TRON2 Bridge WebSocket origin        |
| `inference.operator.ws_port`  | Tron2 WebSocket controller port          |
| `inference.operator.servoj_publish_rate` | ServoJ background rate (300 Hz) |
| `inference.operator.recovery_blend_frames` | ServoJ stream-recovery blend frames (default 6; 0 disables) |
| `inference.operator.chunk_boundary_blend_enabled` | active-chunk smoothstep boundary blending (currently disabled) |
| `inference.operator.chunk_boundary_blend_frames` | active chunk boundary blend frames (currently 6) |
| `inference.operator.chunk_boundary_blend_scope` | blend scope: `arm`, `gripper`, or `all` |
| `inference.operator.max_servoj_step_rad` | Per-waypoint delta guard (rad) |
| `inference.operator.max_state_source_mismatch_rad=None` | disable duplicate Bridge/control mismatch blocking |
| `inference.operator.lock_head` | reject policy head targets and hold measured head |
| `inference.operator.max_head_hold_error_rad` | block ServoJ if the measured head drifts (rad) |
