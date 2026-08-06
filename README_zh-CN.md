# 在新 Tron2 上训练与部署 FluxVLA

[English](README.md) | 简体中文

原始上游 FluxVLA README：
<https://github.com/FluxVLA/FluxVLA/blob/main/README.md>

本项目基于上游 [FluxVLA](https://github.com/FluxVLA/FluxVLA) 项目开发。
感谢他们的杰出工作。

本文档介绍如何使用自定义 Tron2 数据对 PI0.5 进行 LoRA 微调，并通过
remote inference 在真实 Tron2 上部署策略。常见部署形态是：GPU
工作站/服务器负责模型训练和推理，Tron2 外挂算力模块通过 TRON2 Bridge
WebSocket 获取观测、通过 SSH 隧道请求远程推理，并通过机器人控制 WebSocket
执行动作。

当前部署链路是：

```text
TRON2 Bridge WebSocket -> robot-side FluxVLA client -> SSH tunnel -> GPU server ZMQ
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
- 在 Tron2 Power Computing Module 上通过 TRON2 Bridge WebSocket 获取对齐后的
  三路图像、关节和夹爪观测。
- 基于 ZMQ 的 remote inference，并支持 SSH tunnel。
- dry-run 模式：完整执行观测采集、图像传输、服务器推理、动作返回，但不发布
  机器人动作。
- 16 维 Tron2 动作布局：
  `left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)`。
- 开源发布元数据、CI smoke test、Issue 模板和 PR checklist。

### 不包含或不支持的内容

- PI0.5 base 权重需要用户自行获取。
- 机器人网络、TRON2 Bridge 和 WebSocket 控制服务需要用户在自己的 Tron2
  环境中配置。
- 通用任务规划、运动规划、碰撞规避、认证级安全控制和无人值守生产运行不在本仓库
  范围内。
- 不同机器人固件、Bridge 话题布局、控制器 API 或相机配置可能需要用户自行调整配置
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
| Power Computing Module  | Tron2 外挂算力模块，运行 TRON2 Bridge 和 FluxVLA 客户端 |
| Internal robot computer | 机器人内部控制电脑，通常不直接开放给用户       |

Tron2 通常有两台电脑。内部机器人电脑在机器人内部，一般不直接开放给用户。
用户侧主要使用外挂的 Power Computing Module，通常挂在机器人外部，IP
一般是 `10.192.1.4`。

在当前部署方式中，Power Computing Module 通过 SSH 访问 GPU server。
机器人侧客户端会建立本地端口转发：

```text
Power Computing Module localhost:5555 -> GPU server 127.0.0.1:3333
```

观测 WebSocket、控制 WebSocket 与 remote inference 是三条不同链路：

```text
wss://10.192.1.4/bridge/ws        # 对齐后的图像、关节和夹爪
ws://10.192.1.2:5000              # 机器人状态及 MoveJ/ServoJ 控制
```

机器人控制 WebSocket 只在真正执行动作时使用：

```text
ws://<TRON2_CONTROLLER_IP>:<TRON2_WS_PORT>
```

多数 Tron2 部署中，控制器 IP 是 `10.192.1.2`，端口是 `5000`：

```text
TRON2_CONTROLLER_IP = 10.192.1.2
TRON2_WS_PORT       = 5000
```

新机器人一般不需要修改 `robot_ip`，保持 `10.192.1.2` 即可。公开的
[`tron2_env`](https://github.com/limxdynamics/tron2_env) transport 会从服务端
消息自动识别账号 ID，因此 `ws_accid` 必须保持为 `None`。

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

### TRON2 Bridge WebSocket 观测

FluxVLA 客户端不再订阅 ROS，而是使用公开的
`tron2_env.BridgeObservationProvider` 从 Bridge 获取对齐后的图像、关节和夹爪：

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

本部署的 Bridge 使用自签名证书，因此设置了 `bridge_verify_tls=False`；只能在隔离、
可信的机器人局域网内这样使用。如果 Bridge 的话题路径不同，必须先修改显式映射。

### WebSocket 控制参数

WebSocket 控制器 IP 一般是 `10.192.1.2`，通常不用改：

```python
robot_ip='10.192.1.2'
ws_port=5000
ws_accid=None
```

`Tron2EnvOperator` 使用公开的 `tron2_env` runtime 自动识别账号 ID，不要把
`ws_accid` 设置为非 `None`。Bridge 观测走 `wss://10.192.1.4`；prepare pose
使用 MoveJ，策略动作通过独立的机器人 WebSocket 和从实测状态初始化的 300 Hz
ServoJ 发布器执行。PI0.5 LoRA 以 `publish_rate=30` Hz 喂入策略路点。

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

如果所选 checkpoint 的 work directory 中存在 `deployment_metadata.json`，服务端
优先从这个文件读取任务元数据；否则读取已保存 `config.json` 中的
`inference.task_descriptions` 和 `action_layout`。当训练配置遗留了示例 inference
prompt 时，必须用这个显式 sidecar 记录真实训练任务。机器人客户端启动时会列出
解析后的任务 ID，并拒绝未登记的 ID。切换 checkpoint 或修改其元数据后必须重启
服务端；机器人侧代码不需要针对不同任务修改。

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
pip install "tron2-env[bridge] @ git+https://github.com/limxdynamics/tron2_env.git@5b7b145229416f3731f61657e6fa71c89c37bc9d"
```

Power Computing Module 上不要求执行 `pip install -e .`。客户端脚本会设置：

```bash
FLUXVLA_REMOTE_CLIENT_ONLY=1
PYTHONPATH="$(pwd):${PYTHONPATH}"
```

这可以避免在机器人侧导入完整模型/CUDA 栈。

FluxVLA 客户端不需要加载 ROS 环境。TRON2 Bridge 服务内部可以使用 ROS，但该
实现细节不进入 FluxVLA 客户端进程。

## 8. 检查 Bridge 与机器人控制 WebSocket

先检查 Bridge TLS 端口和机器人控制端口：

```bash
python -c "import socket; socket.create_connection(('10.192.1.4', 443), timeout=3); print('bridge tcp ok')"
python -c "import socket; socket.create_connection(('10.192.1.2', 5000), timeout=3); print('control tcp ok')"
```

只读端到端观测检查必须显式关闭控制 transport：

```bash
FLUXVLA_REMOTE_CLIENT_ONLY=1 python -c "from fluxvla.engines.operators import Tron2EnvOperator; o=Tron2EnvOperator(connect_websocket=False); x=o.get_observation(); print(x['state'].shape, {k:v.shape for k,v in x['images'].items()}); o.close()"
```

输出必须包含 18 维状态和三路图像。如果控制端口不稳定，必须先修复再运行
`dry_run=False`。

## 9. Dry Run

当前提交的私有部署配置会启用真实动作：

```python
dry_run=False
```

如需进行 dry-run 验证，必须显式覆盖
`inference.dry_run=True`。dry run 会完成完整感知和远程推理链路，但不会执行
机器人动作：

```text
Bridge WebSocket observations -> SSH tunnel -> GPU inference -> action returned -> print
```

在 Power Computing Module 上运行：

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

如果服务器只能通过 22 端口访问，保留 SSH tunnel 模式。若已配置 SSH key，
运行时不会再要求输入密码。

dry run 启动时会先显示当前权重登记的任务 ID。输入其中一个任务 ID 并按
Enter 确认，再按 `b` 开始；按 `s` 停止继续生成和接收新 chunk：

```text
Task ID: 6
Press b to start, or type another task ID and press Enter.
```

现在不再输入 repeat time。按键状态机在机器人 Power Computing Module 上
运行，GPU 服务器只处理推理请求。dry run 空闲时按 `r` 会打印并跳过
prepare pose。

期望输出包含：

```text
[Tron2InferenceRunner] dry_run=True, skip execution.
```

## 10. 真实执行

只有在下面条件都满足后，才运行真实执行：

- dry run 成功；
- Bridge 观测稳定；
- WebSocket 连接稳定；
- 现场有物理急停。

启动标准的无 RTC 客户端：

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

客户端同步执行完整 32 步 action chunk，执行完才重新取观测并请求下一个
chunk。该路径没有 overlap 队列、guidance prefix 或 inference-time RTC。
如果在推理结果尚未被接收时按 `s`，结果会被丢弃；已经接收的 chunk 会执行完，
之后不再向控制器发送新动作。再次运行必须重新选择任务 ID，再按 `b`。

### 单进程 prefix RTC

train-time RTC 和 inference-time RTC 是同一部署方法的两个阶段，但不是同一段
操作：训练阶段让模型学会在已知干净 action 前缀后续写；推理阶段由 runner 把
上一 chunk 尚未执行完的动作作为该前缀传入模型。RTC 权重不提供前缀也能按普通
chunk 策略运行，但不会使用训练得到的 RTC 条件能力。

如果模型和 RTC 状态放在同一进程，在 GPU 电脑上直接运行下面的命令，不启动
ZMQ server、SSH tunnel 或机器人侧 remote client：

```bash
python scripts/inference.py \
  --config configs/pi05/pi05_paligemma_tron2_lora_rtc_local_inference.py \
  --ckpt-path /path/to/merged-rtc-checkpoint.safetensors \
  --cfg-options inference.dry_run=True
```

该配置使用 `Tron2RTCInferenceRunner`、`method='prefix'` 和与模型 horizon
一致的 50 步 chunk，并从 GPU 电脑直接连接观测/控制 WebSocket。`b/s/r` 按键
状态机也运行在这个终端。按 `s` 后，runner 会等待已经接收的异步轨迹执行完才
报告 idle，因此只能在 idle 使用的 `r` 不会让 ServoJ 和 prepare-pose MoveJ
发生竞争。

当前训练配置采样的 prefix 范围是 `[0, 10)`；30 Hz 下覆盖时间不足
0.333 秒。真实执行前必须先用新 RTC 权重测量端到端推理耗时；如果动态需要的
prefix 超出训练范围，runner 会打印一次警告。

## 11. 进入初始位姿

客户端空闲时按 `r`，会执行原 task ID `0` 使用的同一套 prepare pose：

```text
[TRON2 client idle] Type task ID and press Enter. b=start, r=prepare pose.
```

任务运行中按 `r` 无效；必须先按 `s`，等客户端回到 idle 后才能按 `r`。
dry run 下不会执行 prepare pose。真实执行模式下，每个 prepare pose 使用 MoveJ，
且不发送头部目标。第一次策略动作前，会先用 Bridge 与控制 WebSocket 的反馈校验
整个动作段，校验通过后才允许新建 `tron2_env` MotionController 并以 300 Hz
发布 ServoJ。策略头部轨迹会被拒绝，控制器只保持启动时的实测头部位置；再次请求
prepare 时，会先断开 ServoJ 发布器，之后才发送新的 MoveJ。

## 12. 安全提醒

`Ctrl+C` 会执行客户端清理，并断开 ServoJ 发布器和控制
WebSocket，但它仍然不是机器人急停：不会自动卸力，也不能保证控制器已经接受
的指令被取消。

新机器人首次部署时：

- 操作员应在物理急停旁；
- 按 `b` 前再次确认当前权重和所选任务 ID；
- 物体摆放保守；
- 完整运行前先确认夹爪开合约定；
- 在有实测轨迹支持更严格阈值前，保持 `max_servoj_step_rad=0.2`。

## 13. 常见问题

### `ModuleNotFoundError: No module named 'fluxvla'`

请通过客户端脚本运行：

```bash
bash scripts/remote_inference_client.sh ...
```

脚本会自动设置 `PYTHONPATH`。除非机器具备兼容的编译器/CUDA 环境，否则
Power Computing Module 上不建议安装完整 editable package。

### `ModuleNotFoundError: No module named 'websockets'`

在客户端轻量环境中安装官方 Bridge extra：

```bash
pip install "tron2-env[bridge] @ git+https://github.com/limxdynamics/tron2_env.git@5b7b145229416f3731f61657e6fa71c89c37bc9d"
```

### `TRON2 Bridge did not provide one complete ... observation`

检查 `wss://10.192.1.4/bridge/ws`、配置中的五条话题路径、相机服务和 Bridge
日志。客户端会直接失败，不会复用过期或不完整的观测。

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

- 控制器是否发送可被 `tron2_env` 自动识别的账号 ID；
- 是否有官方控制 UI 占用了独占连接；
- 机器人是否处于正确的外部/API 控制模式；
- WebSocket 指令路径是否匹配控制器 API。

## 14. 主要文件

当前 Tron2 部署相关主要文件：

```text
configs/pi05/pi05_paligemma_tron2_lora_finetune.py
fluxvla/engines/runners/tron2_inference_runner.py
fluxvla/engines/operators/tron2_env_operator.py
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
| `inference.operator.ws_port`  | Tron2 WebSocket 控制端口         |
| `inference.operator.servoj_publish_rate` | ServoJ 后台发布频率（300 Hz） |
| `inference.operator.max_servoj_step_rad` | 相邻路点变化上限（rad） |
| `inference.operator.max_state_source_mismatch_rad=None` | 关闭重复的 Bridge/控制反馈差异阻断 |
| `inference.operator.lock_head` | 拒绝策略头部目标并保持实测头部位置 |
| `inference.operator.max_head_hold_error_rad` | 实测头部漂移时阻止 ServoJ（rad） |
