# Training and Deploying FluxVLA on a New Tron2

This document describes the recommended workflow for fine-tuning PI0.5 with
custom Tron2 data and deploying it with remote inference.  It is written for
the common setup where a GPU workstation/server runs the VLA model, while the
Tron2 robot computer only collects ROS observations and sends WebSocket control
commands to the robot.

The current Tron2 PI0.5 LoRA config is:

```text
configs/pi05/pi05_paligemma_tron2_lora_finetune.py
```

The current deployment path is:

```text
Tron2 ROS topics -> robot-side FluxVLA client -> SSH tunnel -> GPU server ZMQ
    -> PI0.5 policy inference -> action returned to robot client
    -> Tron2 WebSocket control service
```

## 1. Hardware and Network Assumptions

This guide assumes three logical components:

| Component               | Role                                                     |
| ----------------------- | -------------------------------------------------------- |
| GPU server              | Fine-tunes PI0.5 and serves the model through ZMQ        |
| Power Computing Module  | External Tron2 computer; runs ROS and the FluxVLA client |
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

The Tron2 control WebSocket is separate from remote inference.  It is used only
when actions are executed:

```text
ws://<TRON2_CONTROLLER_IP>:<TRON2_WS_PORT>
```

In most Tron2 deployments, the WebSocket controller IP is `10.192.1.2` and the
port is `5000`:

```text
TRON2_CONTROLLER_IP = 10.192.1.2
TRON2_WS_PORT       = 5000
TRON2_WS_ACCID      = <YOUR_TRON2_ACCID>
```

For a new robot, `robot_ip` usually stays `10.192.1.2`; replace `ws_accid` with
the account ID of the target robot.

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

### ROS topics

The current config uses:

```python
operator=dict(
    type='Tron2Operator',
    img_left_topic='/camera/left/color/image_raw',
    img_right_topic='/camera/right/color/image_raw',
    img_top_topic='/camera/top/color/image_raw',
    joint_state_topic='/joint_states',
    gripper_state_topic='/gripper_state',
    ee_pose_left_topic='/left_arm/ee_pose',
    ee_pose_right_topic='/right_arm/ee_pose',
    ws_accid=None,
)
```

On a new robot, first list the available topics:

```bash
rostopic list
```

Then check the frequency of the required topics:

```bash
rostopic hz /camera/left/color/image_raw
rostopic hz /camera/right/color/image_raw
rostopic hz /camera/top/color/image_raw
rostopic hz /joint_states
rostopic hz /gripper_state
```

If the camera topics are different, update the config before running
inference.

### WebSocket control parameters

The WebSocket controller IP is generally `10.192.1.2`, so it usually does not
need to be changed.  The robot-specific value is normally `ws_accid`:

```python
robot_ip='10.192.1.2'
ws_port=5000
ws_accid=None
```

For a new robot, replace `ws_accid` with the target robot's account ID if the
controller does not auto-detect it.

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
pip install rospkg catkin_pkg empy defusedxml netifaces
```

Source ROS before running the client:

```bash
source /opt/ros/noetic/setup.bash
```

Do not require `pip install -e .` on the Power Computing Module.  The remote
client script sets:

```bash
FLUXVLA_REMOTE_CLIENT_ONLY=1
PYTHONPATH="$(pwd):${PYTHONPATH}"
```

This avoids importing the full model/CUDA stack on the robot.

## 8. Verify Robot-Side ROS and WebSocket

Check ROS topics:

```bash
rostopic hz /camera/left/color/image_raw
rostopic hz /camera/right/color/image_raw
rostopic hz /camera/top/color/image_raw
rostopic hz /joint_states
rostopic hz /gripper_state
```

Check the Tron2 WebSocket control service:

```bash
ping -c 3 10.192.1.2
```

```bash
python -c "import socket; socket.create_connection(('10.192.1.2', 5000), timeout=3); print('tcp port ok')"
```

If the TCP port is unstable, fix the Tron2 control service before running
`dry_run=False`.  Remote inference can work without WebSocket in dry-run mode,
but real action execution cannot.

## 9. Run Remote Inference in Dry Run Mode

Dry run is the default in the Tron2 LoRA config:

```python
dry_run=True
```

Dry run performs the full perception and remote inference flow but does not
execute actions:

```text
ROS observations -> SSH tunnel -> GPU inference -> action returned -> print
```

Run this on the Power Computing Module:

```bash
cd ~/FluxVLA
conda activate fluxvla
source /opt/ros/noetic/setup.bash

bash scripts/remote_inference_client.sh \
  configs/pi05/pi05_paligemma_tron2_lora_finetune.py \
  --ssh-host USER@SERVER_PUBLIC_IP \
  --ssh-port 22 \
  --local-port 5555 \
  --remote-port 3333
```

If the server is reachable only through port 22, keep this SSH tunnel mode.
If SSH key login is configured, the command will not ask for a password.

During dry run, first enter task ID `0` to put the robot into the reset /
prepare-pose flow, then enter the actual task ID:

```text
Enter task ID (or press Enter for default): 0
Enter task ID after reset: 1
Number of times to repeat the task: 1
```

In dry-run mode, the prepare-pose command is not executed; this only verifies
the interaction flow before real execution.

Expected dry-run output includes a printed action:

```text
[Tron2InferenceRunner] dry_run=True, skip execution.
```

## 10. Execute on the Robot

Only run real execution after:

- dry run succeeds;
- ROS topics are stable;
- WebSocket handshake succeeds;
- a physical emergency stop is available.

Start with a short execution horizon:

```bash
bash scripts/remote_inference_client.sh \
  configs/pi05/pi05_paligemma_tron2_lora_finetune.py \
  --ssh-host USER@SERVER_PUBLIC_IP \
  --ssh-port 22 \
  --local-port 5555 \
  --remote-port 3333 \
  --cfg-options inference.dry_run=False inference.execute_horizon=4
```

`inference.execute_horizon=4` means the robot executes only the first 4 action
steps from each action chunk, then observes and queries the remote server again.
This is safer for initial deployment.

After the behavior is verified, remove `execute_horizon=4` to execute the full
chunk:

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

so the default execution horizon is the full 32-step chunk.

## 11. Move to the Prepare Pose

During runtime interaction, enter task ID `0` to move the robot to the
configured prepare pose:

```text
Enter task ID (or press Enter for default): 0
Enter task ID after reset: 1
Number of times to repeat the task: 1
```

In dry-run mode, prepare-pose execution is skipped.  In real execution mode,
prepare-pose commands are sent through the Tron2 WebSocket.

## 12. Important Safety Notes

`Ctrl+C` stops the local FluxVLA client and closes the SSH tunnel, but it is not
a robot emergency stop.  It does not automatically disable torque or guarantee
that a command already sent to the Tron2 controller is canceled.

For first deployment on a new robot:

- keep one operator near the physical emergency stop;
- use `inference.execute_horizon=4`;
- keep object placement conservative;
- verify gripper open/close convention before full-speed runs;
- verify `ws_accid` belongs to the current robot.

## 13. Troubleshooting

### `ModuleNotFoundError: No module named 'fluxvla'`

Run the client through:

```bash
bash scripts/remote_inference_client.sh ...
```

The script sets `PYTHONPATH` automatically.  Avoid installing the full editable
package on the Power Computing Module unless the machine has a compatible
compiler/CUDA environment.

### `ModuleNotFoundError: No module named 'rospy'`

Source ROS and install Python ROS dependencies inside the conda environment:

```bash
source /opt/ros/noetic/setup.bash
pip install rospkg catkin_pkg empy defusedxml netifaces
```

### Required ROS topics are missing

If `rostopic list` on the Power Computing Module does not show the required
camera, joint, or gripper topics, start the required robot-side services with
the `install.sh` script inside the Power Computing Module's `limx-agent`
directory:

```bash
cd /path/to/limx-agent
bash install.sh
```

After the services start, run `rostopic list` again and verify the required
topics before launching FluxVLA.

### `ImportError: libp11-kit.so.0: undefined symbol: ffi_type_pointer`

This is a Conda/ROS dynamic-library conflict triggered by `cv_bridge`.
The Tron2 operator now avoids `cv_bridge` for `sensor_msgs/Image` conversion.
Make sure the Power Computing Module has the latest FluxVLA branch with this
change.

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

- `ws_accid` is correct for the robot;
- no other official control UI is holding an exclusive connection;
- the robot is in the correct external-control/API-control mode;
- the command path matches the robot controller's expected WebSocket API.

## 14. Files Changed for Tron2 Deployment

The current Tron2 deployment branch changes these main files:

```text
configs/pi05/pi05_paligemma_tron2_lora_finetune.py
fluxvla/engines/runners/tron2_inference_runner.py
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
| `inference.execute_horizon=4` | execute only the first 4 steps per chunk |
| `inference.operator.ws_port`  | Tron2 WebSocket controller port          |
| `inference.operator.ws_accid` | Tron2 WebSocket account/robot identifier |
