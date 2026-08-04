# Training and Deploying FluxVLA on a New Tron2

[English](README.md) | [简体中文](README_zh-CN.md)

Original upstream FluxVLA README:
<https://github.com/FluxVLA/FluxVLA/blob/main/README.md>

This project is developed based on the upstream
[FluxVLA](https://github.com/FluxVLA/FluxVLA) project. We thank them for their
excellent work.

This document describes the recommended workflow for fine-tuning PI0.5 with
custom Tron2 data and deploying it with remote inference.  It is written for
the common setup where a GPU workstation/server runs the VLA model, while the
Tron2 robot computer consumes TRON2 Bridge WebSocket observations and sends
WebSocket control commands to the robot.

The current deployment path is:

```text
TRON2 Bridge WebSocket -> robot-side FluxVLA client -> SSH tunnel -> GPU server ZMQ
    -> PI0.5 policy inference -> action returned to robot client
    -> Tron2 WebSocket control service
```

## Project Scope and Status

### Intended Users

This repository is intended for:

- course and lab learning around VLA training and robot deployment workflows;
- research reproduction and adaptation of FluxVLA / PI0.5 on Tron2;
- tool integration for dataset conversion, fine-tuning, remote inference, and
  robot-side execution clients;
- experimental validation with dry-run inference and controlled real-robot
  tests.

### Current Status

This is an experimental research release for Tron2-oriented FluxVLA training
and deployment.  It is not a production autonomy stack, not a certified robot
safety system, and not a stable SDK/API guarantee.  Real-robot execution must
be validated by the user in a controlled environment with physical safety
measures.

### Core Features

- Tron2 PI0.5 LoRA fine-tuning configuration.
- LeRobot-style dataset layout and field expectations for Tron2 data.
- Synchronized three-camera, joint, and gripper observations through the
  TRON2 Bridge WebSocket on the Power Computing Module.
- ZMQ-based remote inference with SSH tunnel support.
- Dry-run mode that completes observation collection, image transfer, server
  inference, and action return without publishing robot actions.
- 16-dimensional Tron2 action layout:
  `left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)`.
- Open-source release metadata, CI smoke tests, issue templates, and pull
  request checklist.

### Not Included or Not Supported

- PI0.5 base checkpoints must be obtained separately by the user.
- Robot network access, TRON2 Bridge, and WebSocket control services must be
  configured on the user's own Tron2 environment.
- General task planning, motion planning, collision avoidance, certified safety
  control, and unattended production operation are out of scope.
- Different robot firmware, Bridge topic layouts, controller APIs, or camera
  setups may require local configuration or code changes.

### Repository Layout

| Path                             | Purpose                                                 |
| -------------------------------- | ------------------------------------------------------- |
| `configs/pi05/`                  | PI0.5 training, inference, and Tron2 LoRA configs       |
| `fluxvla/`                       | Core Python package, models, runners, transforms, ops   |
| `scripts/`                       | Training, inference, evaluation, and remote client CLIs |
| `docs/`                          | Extended deployment, remote inference, and review docs  |
| `tools/`                         | Dataset conversion and utility scripts                  |
| `test/`                          | Unit tests and lightweight CI smoke checks              |
| `.github/`                       | CI workflows and Issue / PR templates                   |
| `datasets/`                      | Local datasets; ignored by Git except `.gitkeep`        |
| `checkpoints/`                   | Local model weights; ignored by Git except `.gitkeep`   |
| `work_dirs/`                     | Local training outputs; ignored by Git                  |
| `.env.example`                   | Public environment-variable template without secrets    |
| `NOTICE`                         | Third-party code, model, and data-source notices        |
| `docs/release_license_review.md` | Third-party dependency license review                   |

### License and Contributions

Code in this repository is distributed under the Apache License 2.0 unless
otherwise noted.  See `LICENSE`, `NOTICE`, and
`docs/release_license_review.md` for license and third-party attribution
details.

Contribution guidelines are in `CONTRIBUTING.md`.  Use GitHub Issues for bug
reports and feature requests, and use the pull request template for code
changes.  Security-sensitive reports should follow `SECURITY.md` instead of
being posted publicly.

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

For a new robot, `robot_ip` usually stays `10.192.1.2`.  The public
[`tron2_env`](https://github.com/limxdynamics/tron2_env) transport detects the
controller account ID from server messages, so `ws_accid` must remain `None`.

Important: remote inference requires the Power Computing Module to reach the
GPU server.  To connect the robot to the public network, either configure Wi-Fi
from the `10.192.1.2:8080` web interface, or connect an Ethernet cable to any
ETH port on the Power Computing Module.

## 2. Prepare the GPU Server

Create the full FluxVLA training environment on the GPU server.  Follow the
installation guide from the upstream
[FluxVLA](https://github.com/FluxVLA/FluxVLA) project; the environment name used
here is `fluxvla`.

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

The FluxVLA client does not subscribe to ROS. It uses the public
`tron2_env.BridgeObservationProvider` to receive aligned images, joints, and
grippers from the Bridge:

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
    max_servoj_step_rad=0.2,
    max_state_source_mismatch_rad=None,
    lock_head=True,
    max_head_hold_error_rad=0.05,
)
```

`bridge_verify_tls=False` is required by this deployment's self-signed Bridge
certificate and is acceptable only on the isolated, trusted robot LAN. If the
Bridge publishes different topic paths, update the explicit mappings before
inference.

### WebSocket control parameters

The WebSocket controller IP is generally `10.192.1.2`, so it usually does not
need to be changed:

```python
robot_ip='10.192.1.2'
ws_port=5000
ws_accid=None
```

`Tron2EnvOperator` uses the public `tron2_env` runtime, which auto-detects the
account ID. Do not set `ws_accid` to a non-`None` value. Bridge WebSocket
observations use `wss://10.192.1.4`; prepare poses use MoveJ and policy actions
use the separate robot WebSocket with a 300 Hz ServoJ publisher seeded from
measured joint state. The current PI0.5 LoRA deployment feeds policy waypoints
at `publish_rate=30` Hz.

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

The server reads task metadata from `deployment_metadata.json` beside the
selected checkpoint when that file exists; otherwise it uses
`inference.task_descriptions` and `action_layout` from the work directory's
saved `config.json`. The explicit sidecar is required when a saved training
config contains a stale example inference prompt. The robot client lists the
resolved task IDs at startup and rejects unknown IDs. After changing a
checkpoint or its metadata, restart the server; robot-side code does not need
task-specific edits.

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

Do not require `pip install -e .` on the Power Computing Module.  The remote
client script sets:

```bash
FLUXVLA_REMOTE_CLIENT_ONLY=1
PYTHONPATH="$(pwd):${PYTHONPATH}"
```

This avoids importing the full model/CUDA stack on the robot.

No ROS environment is required by the FluxVLA client. The TRON2 Bridge service
may use ROS internally, but that is outside the client process.

## 8. Verify Bridge and Robot-Control WebSockets

Check that the Bridge TLS endpoint and robot-control TCP port are reachable:

```bash
python -c "import socket; socket.create_connection(('10.192.1.4', 443), timeout=3); print('bridge tcp ok')"
python -c "import socket; socket.create_connection(('10.192.1.2', 5000), timeout=3); print('control tcp ok')"
```

For a read-only end-to-end observation check, explicitly disable the control
transport:

```bash
FLUXVLA_REMOTE_CLIENT_ONLY=1 python -c "from fluxvla.engines.operators import Tron2EnvOperator; o=Tron2EnvOperator(connect_websocket=False); x=o.get_observation(); print(x['state'].shape, {k:v.shape for k,v in x['images'].items()}); o.close()"
```

This must report an 18-dimensional state and all three images. If the control
port is unstable, fix it before running `dry_run=False`.

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

## 11. Move to the Prepare Pose

While the client is idle, press `r` to run the same configured prepare-pose
sequence previously exposed as task ID `0`:

```text
[TRON2 client idle] Type task ID and press Enter. b=start, r=prepare pose.
```

`r` is ignored while a task is running; press `s`, wait until the client
reports idle, and only then press `r`. In dry-run mode, prepare-pose execution
is skipped. In real execution mode,
each prepare pose uses MoveJ and does not send a head target. Before the first
policy action, the full chunk is checked against both Bridge and control
feedback before a fresh `tron2_env` MotionController may stream ServoJ at
300 Hz. Policy head trajectories are rejected and the measured head position
is held. If prepare is requested again, the ServoJ publisher is disconnected
before any new MoveJ command is sent.

## 12. Important Safety Notes

`Ctrl+C` runs client cleanup and disconnects the
ServoJ publisher and control WebSocket.  It is still not a robot emergency
stop: it does not disable torque or guarantee that a command already accepted
by the Tron2 controller is canceled.

For first deployment on a new robot:

- keep one operator near the physical emergency stop;
- verify the selected checkpoint task ID before pressing `b`;
- keep object placement conservative;
- verify gripper open/close convention before full-speed runs;
- keep `max_servoj_step_rad=0.2` until recorded trajectories justify a tighter
  deployment-specific value.

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

Install the official Bridge extra in the client environment:

```bash
pip install "tron2-env[bridge] @ git+https://github.com/limxdynamics/tron2_env.git@5b7b145229416f3731f61657e6fa71c89c37bc9d"
```

### `TRON2 Bridge did not provide one complete ... observation`

Check `wss://10.192.1.4/bridge/ws`, the five configured topic paths, camera
services, and Bridge logs. The client deliberately fails instead of reusing a
stale or partial observation.

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
| `inference.operator.ws_port`  | Tron2 WebSocket controller port          |
| `inference.operator.servoj_publish_rate` | ServoJ background rate (300 Hz) |
| `inference.operator.max_servoj_step_rad` | Per-waypoint delta guard (rad) |
| `inference.operator.max_state_source_mismatch_rad=None` | disable duplicate Bridge/control mismatch blocking |
| `inference.operator.lock_head` | reject policy head targets and hold measured head |
| `inference.operator.max_head_hold_error_rad` | block ServoJ if the measured head drifts (rad) |
