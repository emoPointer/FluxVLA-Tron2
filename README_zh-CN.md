# 在新 Tron2 上训练与部署 FluxVLA

[English](README.md) | 简体中文

原始上游 FluxVLA README：
<https://github.com/FluxVLA/FluxVLA/blob/main/README.md>

本项目基于上游 [FluxVLA](https://github.com/FluxVLA/FluxVLA) 项目开发。
感谢他们的杰出工作。

本文档介绍如何使用自定义 Tron2 数据对 PI0.5 进行 LoRA 微调，并通过
remote inference 在真实 Tron2 上部署策略。常见部署形态是：GPU
工作站/服务器负责模型训练和推理，Tron2 外挂算力模块负责采集 ROS
观测、通过 SSH 隧道请求远程推理，并通过 Tron2 WebSocket 控制服务执行动作。

当前部署链路是：

```text
Tron2 ROS topics -> robot-side FluxVLA client -> SSH tunnel -> GPU server ZMQ
    -> PI0.5 policy inference -> action returned to robot client
    -> Tron2 WebSocket control service
```

## 项目适用范围与状态

### 适用对象

本仓库适用于：

- 课程学习和实验教学中的 VLA 训练、部署流程演示；
- 基于 Tron2 的 FluxVLA / PI0.5 科研复现和二次开发；
- 数据转换、LoRA 微调、remote inference、机器人侧客户端等工具调用集成；
- dry-run 推理链路验证，以及受控条件下的真实机器人实验验证。

### 当前状态

本仓库是面向 Tron2 的 FluxVLA 训练与部署试验性研究版本。它不是生产级自主
机器人系统，不是经过认证的机器人安全系统，也不承诺稳定 SDK/API。真实机器人
执行前，使用者必须在受控环境中完成安全验证，并准备物理急停等安全措施。

### 核心功能

- Tron2 PI0.5 LoRA 微调配置。
- 面向 Tron2 数据的 LeRobot 数据结构和字段说明。
- 在 Tron2 Power Computing Module 上采集三路 ROS 相机观测。
- 基于 ZMQ 的 remote inference，并支持 SSH tunnel。
- dry-run 模式：完整执行观测采集、图像传输、服务器推理、动作返回，但不发布
  机器人动作。
- 16 维 Tron2 动作布局：
  `left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)`。
- 开源发布元数据、CI smoke test、Issue 模板和 PR checklist。

### 不包含或不支持的内容

- PI0.5 base 权重需要用户自行获取。
- 机器人网络、机器人账号、ROS 服务和 WebSocket 控制服务需要用户在自己的
  Tron2 环境中配置。
- 通用任务规划、运动规划、碰撞规避、认证级安全控制和无人值守生产运行不在本仓库
  范围内。
- 不同机器人固件、ROS 话题布局、控制器 API 或相机配置可能需要用户自行调整配置
  或代码。

### 目录结构

| 路径                             | 说明                                          |
| -------------------------------- | --------------------------------------------- |
| `configs/pi05/`                  | PI0.5 训练、推理和 Tron2 LoRA 配置            |
| `fluxvla/`                       | 核心 Python 包、模型、runner、transform、算子 |
| `scripts/`                       | 训练、推理、评估和 remote client 命令入口     |
| `docs/`                          | 部署、remote inference、开源审查等扩展文档    |
| `tools/`                         | 数据转换和工具脚本                            |
| `test/`                          | 单元测试和轻量 CI smoke test                  |
| `.github/`                       | CI workflow、Issue 模板和 PR 模板             |
| `datasets/`                      | 本地数据集目录；除 `.gitkeep` 外被 Git 忽略   |
| `checkpoints/`                   | 本地模型权重目录；除 `.gitkeep` 外被 Git 忽略 |
| `work_dirs/`                     | 本地训练输出目录；被 Git 忽略                 |
| `.env.example`                   | 不含密钥的公开环境变量模板                    |
| `NOTICE`                         | 第三方代码、模型和数据来源说明                |
| `docs/release_license_review.md` | 第三方依赖 License 审查表                     |

### License、贡献和反馈入口

除非另有说明，本仓库代码使用 Apache License 2.0。License 和第三方归属信息见
`LICENSE`、`NOTICE` 和 `docs/release_license_review.md`。

贡献方式见 `CONTRIBUTING.md`。Bug report 和 feature request 请使用 GitHub
Issue 模板，代码改动请使用 PR 模板。安全问题请按 `SECURITY.md` 处理，不要在公开
Issue 中披露漏洞细节。

## 1. 硬件与网络假设

本文档假设存在三个逻辑组件：

| 组件                    | 作用                                           |
| ----------------------- | ---------------------------------------------- |
| GPU server              | 微调 PI0.5，并通过 ZMQ 提供远程推理服务        |
| Power Computing Module  | Tron2 外挂算力模块，运行 ROS 和 FluxVLA 客户端 |
| Internal robot computer | 机器人内部控制电脑，通常不直接开放给用户       |

Tron2 通常有两台电脑。内部机器人电脑在机器人内部，一般不直接开放给用户。
用户侧主要使用外挂的 Power Computing Module，通常挂在机器人外部，IP
一般是 `10.192.1.4`。

在当前部署方式中，Power Computing Module 通过 SSH 访问 GPU server。
机器人侧客户端会建立本地端口转发：

```text
Power Computing Module localhost:5555 -> GPU server 127.0.0.1:3333
```

Tron2 控制 WebSocket 与 remote inference 是两条不同链路。WebSocket
只在真正执行动作时使用：

```text
ws://<TRON2_CONTROLLER_IP>:<TRON2_WS_PORT>
```

多数 Tron2 部署中，控制器 IP 是 `10.192.1.2`，端口是 `5000`：

```text
TRON2_CONTROLLER_IP = 10.192.1.2
TRON2_WS_PORT       = 5000
TRON2_WS_ACCID      = <YOUR_TRON2_ACCID>
```

新机器人一般不需要修改 `robot_ip`，保持 `10.192.1.2` 即可；如控制器
无法自动识别，则需要设置当前机器人的 `ws_accid`。

重要提醒：remote inference 要求 Power Computing Module 能访问 GPU
server。机器人连接公网可以在 `10.192.1.2:8080` Web 界面配置 Wi-Fi，
也可以把有线网络接入 Power Computing Module 的任意 ETH 口。

## 2. 准备 GPU Server

在 GPU server 上安装完整 FluxVLA 训练环境。安装步骤参考上游
[FluxVLA](https://github.com/FluxVLA/FluxVLA) 项目。本文档默认 conda
环境名为 `fluxvla`。

```bash
conda activate fluxvla
cd /path/to/FluxVLA
```

PI0.5 base 权重应放在：

```text
checkpoints/pi05_base/
```

配置至少需要：

```text
checkpoints/pi05_base/model.safetensors
checkpoints/pi05_base/tokenizer_config.json
checkpoints/pi05_base/tokenizer.model or tokenizer.json
```

不要把 tokenizer 指向训练输出目录，除非该目录包含完整 Hugging Face
tokenizer。Tron2 LoRA 配置里应保持：

```python
model_path='checkpoints/pi05_base'
```

### 可选环境变量文件

仓库提供 `.env.example` 作为环境变量模板。该文件只包含占位符和公开默认值；
不要把真实 token、机器人账号 ID、私有主机地址或凭据提交到 Git。

FluxVLA 不会自动加载 `.env`。如果你在本地维护私有 `.env`，运行训练或推理前
需要在 shell 中显式加载：

```bash
set -a
source .env
set +a
```

## 3. 准备数据集

数据可以直接从云端数据平台导出。云端数据平台登录地址和账号请向交付人员获取。

将 LeRobot 格式的 Tron2 数据放到 `datasets/` 下。默认配置使用：

```text
datasets/lerobot_dataset
```

如果数据路径不同，修改：

```python
train_dataloader.dataset.datasets[0].data_root_path
```

当前配置期望的数据字段是：

| 字段                                 | 含义           |
| ------------------------------------ | -------------- |
| `observation.state`                  | 机器人本体状态 |
| `action` / `actions`                 | 动作序列       |
| `observation.images.cam_high`        | 顶部 RGB 相机  |
| `observation.images.cam_left_wrist`  | 左侧 RGB 相机  |
| `observation.images.cam_right_wrist` | 右侧 RGB 相机  |

当前动作维度是 16 维：

```text
left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)
```

PI0.5 内部仍会把 state/action padding 到 32 维以保持兼容，但动作反归一化
和执行只使用前 16 维。

训练前至少检查：

```bash
ls datasets/lerobot_dataset
```

并人工确认几个 episode：

- 相机名称与配置一致；
- action 维度是 16；
- gripper 开合约定与训练和执行一致；
- task language 正确；
- 失败或异常 episode 已移除。

## 4. 修改 Tron2 配置

从下面这个配置开始：

```text
configs/pi05/pi05_paligemma_tron2_lora_finetune.py
```

### 数据路径

```python
data_root_path=[
    './datasets/lerobot_dataset',
]
```

### 任务描述

```python
task_descriptions={
    '1': 'Pick up the banana from the desk and place it on the plate',
}
```

客户端运行时会要求输入 task ID。输入 `1` 时，模型会收到这里对应的任务描述。

### ROS 话题

当前配置使用：

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

在新机器人上，先查看话题：

```bash
rostopic list
```

再检查频率：

```bash
rostopic hz /camera/left/color/image_raw
rostopic hz /camera/right/color/image_raw
rostopic hz /camera/top/color/image_raw
rostopic hz /joint_states
rostopic hz /gripper_state
```

如果话题不同，需要先更新配置。

### WebSocket 控制参数

WebSocket 控制器 IP 一般是 `10.192.1.2`，通常不用改。机器人相关的值主要是
`ws_accid`：

```python
robot_ip='10.192.1.2'
ws_port=5000
ws_accid=None
```

如果控制器无法自动识别 `accid`，请设置当前机器人的账号 ID。

## 5. 使用 LoRA 微调 PI0.5

当前 LoRA 设置在 model 配置中：

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

当前训练计划：

```python
runner.max_steps=30000
train_dataloader.per_device_batch_size=16
```

4 张 GPU 时，全局 batch size 为 64：

```text
16 per GPU x 4 GPUs = 64
```

示例训练命令：

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

恢复训练：

```bash
NPROC_PER_NODE=4 bash scripts/train.sh \
  configs/pi05/pi05_paligemma_tron2_lora_finetune.py \
  work_dirs/pi05_paligemma_tron2_lora_finetune \
  --resume-from /path/to/checkpoint.pt
```

训练输出位于：

```text
work_dirs/pi05_paligemma_tron2_lora_finetune/
```

推理服务通常使用 `.safetensors` checkpoint：

```text
work_dirs/pi05_paligemma_tron2_lora_finetune/checkpoints/step-XXXXX.safetensors
```

训练脚本也会在 work directory 中保存 dataset statistics，remote inference
会使用这些统计信息做动作反归一化。

## 6. 启动 Remote Inference Server

在 GPU server 上运行。

如果机器人只能通过 SSH tunnel 访问服务器，ZMQ server 绑定到 `127.0.0.1`：

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

如果机器人能在局域网直接访问服务器，可以绑定到 `0.0.0.0` 或服务器局域网 IP。

服务端会加载：

1. PI0.5 模型；
2. LoRA checkpoint；
3. inference dataset preprocessing pipeline；
4. 动作反归一化 transform；
5. 训练 work directory 中的 `dataset_statistics.json`。

## 7. 准备 Power Computing Module

Power Computing Module 不需要完整 CUDA 训练栈，只需要运行轻量 remote client。

安装 Miniconda：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

创建环境：

```bash
conda create -n fluxvla python=3.10 -y
conda activate fluxvla
```

安装客户端依赖：

```bash
pip install mmengine pyzmq msgpack numpy safetensors websocket-client
pip install rospkg catkin_pkg empy defusedxml netifaces
```

运行客户端前 source ROS：

```bash
source /opt/ros/noetic/setup.bash
```

Power Computing Module 上不要求执行 `pip install -e .`。客户端脚本会设置：

```bash
FLUXVLA_REMOTE_CLIENT_ONLY=1
PYTHONPATH="$(pwd):${PYTHONPATH}"
```

这可以避免在机器人侧导入完整模型/CUDA 栈。

## 8. 检查 ROS 与 WebSocket

检查 ROS 话题频率：

```bash
rostopic hz /camera/left/color/image_raw
rostopic hz /camera/right/color/image_raw
rostopic hz /camera/top/color/image_raw
rostopic hz /joint_states
rostopic hz /gripper_state
```

检查 Tron2 WebSocket 控制服务：

```bash
ping -c 3 10.192.1.2
```

```bash
python -c "import socket; socket.create_connection(('10.192.1.2', 5000), timeout=3); print('tcp port ok')"
```

如果 TCP 端口不稳定，先修复 Tron2 控制服务。dry run 可以不依赖
WebSocket，但真实动作执行必须依赖它。

## 9. Dry Run

Tron2 LoRA 配置默认：

```python
dry_run=True
```

dry run 会完成完整感知和远程推理链路，但不会执行机器人动作：

```text
ROS observations -> SSH tunnel -> GPU inference -> action returned -> print
```

在 Power Computing Module 上运行：

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

如果服务器只能通过 22 端口访问，保留 SSH tunnel 模式。若已配置 SSH key，
运行时不会再要求输入密码。

dry run 时先输入 task ID `0`，验证 reset / prepare-pose 交互流程，再输入真实
任务 ID：

```text
Enter task ID (or press Enter for default): 0
Enter task ID after reset: 1
Number of times to repeat the task: 1
```

dry run 不会真正执行 prepare pose，只验证真实执行前的交互流程。

期望输出包含：

```text
[Tron2InferenceRunner] dry_run=True, skip execution.
```

## 10. 真实执行

只有在下面条件都满足后，才运行真实执行：

- dry run 成功；
- ROS 话题稳定；
- WebSocket 连接稳定；
- 现场有物理急停。

首次部署建议使用较短的执行 horizon：

```bash
bash scripts/remote_inference_client.sh \
  configs/pi05/pi05_paligemma_tron2_lora_finetune.py \
  --ssh-host USER@SERVER_PUBLIC_IP \
  --ssh-port 22 \
  --local-port 5555 \
  --remote-port 3333 \
  --cfg-options inference.dry_run=False inference.execute_horizon=4
```

`inference.execute_horizon=4` 表示每个 action chunk 只执行前 4 步，然后重新
观测并请求远程推理。初次部署更安全。

确认行为稳定后，可以去掉 `execute_horizon=4`：

```bash
bash scripts/remote_inference_client.sh \
  configs/pi05/pi05_paligemma_tron2_lora_finetune.py \
  --ssh-host USER@SERVER_PUBLIC_IP \
  --ssh-port 22 \
  --local-port 5555 \
  --remote-port 3333 \
  --cfg-options inference.dry_run=False
```

当前配置：

```python
action_chunk=32
```

默认执行完整 32 步 action chunk。

## 11. 进入初始位姿

运行交互中输入 task ID `0`，机器人会进入配置好的 prepare pose：

```text
Enter task ID (or press Enter for default): 0
Enter task ID after reset: 1
Number of times to repeat the task: 1
```

dry run 下不会执行 prepare pose。真实执行模式下，prepare-pose 指令通过
Tron2 WebSocket 发送。

## 12. 安全提醒

`Ctrl+C` 只会停止本地 FluxVLA client 并关闭 SSH tunnel，不是机器人急停。
它不会自动卸力，也不能保证已经发送给 Tron2 控制器的指令被取消。

新机器人首次部署时：

- 操作员应在物理急停旁；
- 使用 `inference.execute_horizon=4`；
- 物体摆放保守；
- 完整运行前先确认夹爪开合约定；
- 确认 `ws_accid` 属于当前机器人。

## 13. 常见问题

### `ModuleNotFoundError: No module named 'fluxvla'`

请通过客户端脚本运行：

```bash
bash scripts/remote_inference_client.sh ...
```

脚本会自动设置 `PYTHONPATH`。除非机器具备兼容的编译器/CUDA 环境，否则
Power Computing Module 上不建议安装完整 editable package。

### `ModuleNotFoundError: No module named 'rospy'`

在 conda 环境中 source ROS 并安装 ROS Python 依赖：

```bash
source /opt/ros/noetic/setup.bash
pip install rospkg catkin_pkg empy defusedxml netifaces
```

### 缺少必要 ROS 话题

如果 Power Computing Module 上执行 `rostopic list` 后缺少必要相机、关节或
夹爪话题，请进入算力模块内部的 `limx-agent` 目录并执行：

```bash
cd /path/to/limx-agent
bash install.sh
```

服务启动后重新执行 `rostopic list` 并检查话题。

### `ImportError: libp11-kit.so.0: undefined symbol: ffi_type_pointer`

这是 Conda/ROS 动态库冲突，通常由 `cv_bridge` 触发。当前 Tron2 operator
已经避免使用 `cv_bridge` 转换 `sensor_msgs/Image`。请确认 Power Computing
Module 使用了包含该改动的 FluxVLA 分支。

### `Cannot reach VLA server at tcp://127.0.0.1:5555`

SSH tunnel 或 GPU server 没有运行。检查：

```bash
ssh -p 22 -L 5555:127.0.0.1:3333 USER@SERVER_PUBLIC_IP -N
```

并确认服务端监听在 3333 端口。

### `Connection refused` for `ws://10.192.1.2:5000`

这是 Tron2 控制 WebSocket，不是 FluxVLA remote inference server。检查
Tron2 控制服务是否运行，以及 IP/端口是否正确：

```bash
python -c "import socket; socket.create_connection(('10.192.1.2', 5000), timeout=3); print('tcp port ok')"
```

如果端口不稳定，先修复或重启 Tron2 控制服务，再使用
`inference.dry_run=False`。

如果 WebSocket 无法稳定连接，请检查真实机器人是否处于 high-level dev mode。

### WebSocket 能连接但命令不生效

检查：

- `ws_accid` 是否属于当前机器人；
- 是否有官方控制 UI 占用了独占连接；
- 机器人是否处于正确的外部/API 控制模式；
- WebSocket 指令路径是否匹配控制器 API。

## 14. 主要文件

当前 Tron2 部署相关主要文件：

```text
configs/pi05/pi05_paligemma_tron2_lora_finetune.py
fluxvla/engines/runners/tron2_inference_runner.py
fluxvla/engines/operators/tron2_operator.py
fluxvla/engines/runners/base_inference_runner.py
fluxvla/transforms/normalize.py
scripts/remote_inference_client.sh
```

常用运行参数：

| 参数                          | 含义                             |
| ----------------------------- | -------------------------------- |
| `inference.dry_run=True`      | 完成推理链路，但不执行机器人动作 |
| `inference.dry_run=False`     | 在机器人上执行返回动作           |
| `inference.execute_horizon=4` | 每个 chunk 只执行前 4 步         |
| `inference.operator.ws_port`  | Tron2 WebSocket 控制端口         |
| `inference.operator.ws_accid` | Tron2 WebSocket 账号/机器人标识  |
