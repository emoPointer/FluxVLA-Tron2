# Project Work Log

This is the project's development and environment-maintenance log. Record only
commands and results that were actually executed.

## 2026-08-03 19:04 CST — Initial environment setup and installability repair

- Purpose: inspect the Tron2 adaptation and its FluxVLA upstream, then prepare a
  reproducible GPU development environment on the current RTX 3090 host.
- Baselines inspected:
  - local commit `d835df5` (`Initial FluxVLA Tron2 release`);
  - upstream `FluxVLA/FluxVLA` commit
    `7ee6b9db8e1b9e1b91999068e6767daf96fcf5e7`;
  - repository README files, deployment/serving documentation, configuration,
    setup script, CI smoke tests, dependency list, and Git status.
- Environment created:
  - installed Miniconda `py310_25.7.0-2` under `$HOME/miniconda3` after
    validating installer SHA-256
    `647b55a8da07136fa2543fbf6b9719d3a4c2369dec5dd31d2c4bda2b51107717`;
  - created conda environment `fluxvla` with Python 3.10.20;
  - installed CUDA nvcc 12.4.131, CUDA runtime development headers, cuBLAS
    development files, CMake 3.31.8, and Ninja;
  - installed PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, torchaudio
    2.6.0+cu124, FlashAttention 2.5.5, the repository requirements, remote
    inference dependencies (`pyzmq`, `msgpack`), and `websocket-client`;
  - persisted GCC/G++, CUDA home, compute capability 8.6, and CUDA include/link
    search paths as conda environment variables;
  - installed FluxVLA in editable mode and generated the default LIBERO config
    at `$HOME/.libero/config.yaml`.
- Repository repairs:
  - added six CUDA/C++ implementation files that `setup.py` referenced but the
    release omitted. Their content was copied from the matching upstream paths
    at the recorded upstream commit and verified byte-for-byte with `cmp`;
  - changed the root experiment ignore rule from `src/` to `/src/` so nested
    package-owned CUDA source directories remain trackable;
  - added PyAV 14.2.0 and compatible TensorFlow Datasets/Metadata pins to
    `requirements.txt`.
- Important failures and root causes:
  - `egl_probe` initially failed because the environment's CMake directory was
    absent from `PATH`; retrying with the activated environment fixed it;
  - the first editable build failed because all six declared CUDA sources were
    absent from the repository; restoring the upstream sources fixed it;
  - unconstrained TFDS resolved to 4.9.10 and TensorFlow Metadata 1.21.0, whose
    generated protobuf code was incompatible with TensorFlow 2.15's protobuf
    4.25.9 constraint. TFDS 4.9.3 plus Metadata 1.17.3 fixed the runtime import;
  - upstream's PyAV 14.4 conda fallback pulled NumPy 2.2.6, conflicting with
    TensorFlow 2.15. That conda revision was rolled back, NumPy 1.26.4/Pillow
    were restored, and the closest available binary wheel, PyAV 14.2.0, was
    installed and tested.
- Validation executed:
  - `pip check`: no broken requirements;
  - `pytest -q test/test_ci_smoke.py test/test_ops/test_flashattn.py`: 8 passed;
  - PyTorch detected `NVIDIA GeForce RTX 3090`, capability 8.6, and CUDA 12.4;
  - FlashAttention GPU tests passed;
  - all three custom CUDA operations imported and matched PyTorch reference
    computations on GPU;
  - `import fluxvla`, TensorFlow 2.15 CPU tensor computation, TFDS/dlimp import,
    and remote-serving dependency imports passed;
  - PyAV encoded and decoded a two-frame in-memory MP4 successfully;
  - a new interactive shell activated `fluxvla`, selected the expected Python,
    and exposed nvcc 12.4 without activation errors.
- Remaining limitations and risks:
  - TensorFlow reports no GPU device because its optional CUDA runtime set was
    not installed; FluxVLA training uses PyTorch for GPU compute, while the
    TensorFlow/dlimp data path was validated on CPU;
  - checkpoints, datasets, ROS, a physical Tron2, and robot credentials were not
    available, so full training, checkpoint inference, and hardware deployment
    were not run;
  - this host has one 24 GB RTX 3090, whereas documented example commands use
    multiple GPUs; real training needs a single-process launch and likely a
    reduced per-device batch size.

## 2026-08-03 20:10 CST — Synchronize latest Tron2 LoRA inference checkpoint

- Purpose: retrieve the newest completed checkpoint and its resolved
  configuration from the active training run on the authorized remote host,
  while excluding training-only or redundant artifacts.
- Remote run inspected:
  - work directory
    `work_dirs/pi05_two_tasks_lora_rank256_20260803_170124`;
  - active eight-GPU training remained healthy and continued running;
  - the newest completed checkpoint before and after transfer was
    `step-010000-epoch-005-loss=0.0039`.
- Artifact selection:
  - synchronized the merged model-only `.safetensors` checkpoint and the
    resolved `config.yaml`, tokenizer `config.json`,
    `dataset_statistics.json`, and `llm_backbone_config.json`;
  - synchronized the six small tokenizer support files referenced through
    `checkpoints/pi05_base`, while explicitly excluding that directory's
    14.47 GB base `model.safetensors` because the merged run checkpoint already
    contains the model weights;
  - did not synchronize the paired `.pt` file because it also contains the
    optimizer and scheduler state needed only to resume training;
  - inspected safetensors metadata and inference call paths. The merged model
    contains 812 tensors: language backbone, vision backbone, action expert,
    multimodal projector, and action/time projections;
  - the action expert's 0.98 GiB token embedding is not exercised by the
    current action inference path, but all inference loaders use
    `load_state_dict(..., strict=True)`. It was therefore retained so the
    synchronized checkpoint remains usable without a code or configuration
    change;
  - removed the locally transferred LoRA adapter, adapter metadata, and its
    generated README after validation because the merged checkpoint already
    includes those updates. Remote originals were not modified. Training logs
    and metrics were also excluded.
- Important commands executed: remote `find`, `pgrep`, `sha256sum`, and
  safetensors-header inspection; resumable `rsync --append-verify`; local
  `stat`, `sha256sum`, and safetensors-header inspection.
- Validation:
  - local checkpoint size is exactly `14,466,989,776` bytes;
  - local and remote checkpoint SHA-256 both equal
    `305f6682cf6bd9679bc6fa74dcf8f258172c3c247b7da1424f514e84eafec92b`;
  - the local safetensors header parses successfully and reports 812 tensors
    with the expected eight top-level model components;
  - all four synchronized configuration/statistics files match their remote
    SHA-256 values;
  - all six base tokenizer support files match their remote SHA-256 values;
    `AutoTokenizer.from_pretrained(..., local_files_only=True)` loaded them as
    `GemmaTokenizerFast` and successfully tokenized a smoke-test prompt;
  - the resolved Python configuration parsed successfully with mmengine and
    selected `PI05FlowMatching` as the inference model;
  - the tracked source configuration
    `configs/pi05/pi05_paligemma_tron2_lora_finetune.py` already matched the
    remote copy (`342fd41f6e6e24d5ad1074acbd34434b1c981a7006207d748cbcdfa9eb8ecdd1`),
    so it was not overwritten.
- Remaining limitations and risks:
  - no GPU model forward pass was run; loading the full 14.47 GB checkpoint and
    executing robot inference was outside this synchronization task;
  - the remote training run is still active, so a newer checkpoint may be
    produced later.

## 2026-08-03 20:24 CST — Launch and validate Tron2 remote inference service

- Purpose: use this RTX 3090 host as the FluxVLA compute server and expose the
  documented ZeroMQ inference endpoint for a remote Tron2 client.
- Service configuration:
  - launched the documented `fluxvla.engines.runners.serving.serve` entry point
    with the Tron2 LoRA configuration and synchronized step-010000 merged
    checkpoint;
  - selected `cuda:0` and `bf16`, and bound only to `127.0.0.1:3333` so the
    robot-side client can access it through the documented SSH tunnel;
  - launched it as the user-level transient systemd unit
    `fluxvla-tron2-step010000.service`, with the repository as its working
    directory and the `fluxvla` conda environment's Python executable;
  - added the runtime symlink
    `work_dirs/pi05_two_tasks_lora_rank256_20260803_170124/tokenizer` pointing
    to `checkpoints/pi05_base`, because the inference dataset resolves the
    tokenizer relative to the checkpoint work directory. The target contains
    the synchronized and checksum-verified tokenizer support files.
- Root-cause repair:
  - the first realistic request exposed a `(16,)` versus `(18,)` broadcasting
    failure: the `tron2_16` server layout supplies 16 active state/action
    dimensions, while the training statistics retain two additional head-joint
    dimensions;
  - updated `fluxvla/transforms/normalize.py` so normalization and
    denormalization validate statistic vectors and align them to the configured
    active state/action dimension. Exact-dimension configurations retain their
    previous behavior, and undersized statistics now raise a contextual error;
  - added focused regression coverage in
    `test/test_transforms/test_normalize.py` for 16-dimensional state
    normalization, 16-dimensional action denormalization, and invalid
    undersized statistics.
- Important commands executed:
  - `systemd-run --user --unit=fluxvla-tron2-step010000 ... python -m
    fluxvla.engines.runners.serving.serve ... --host 127.0.0.1 --port 3333
    --device cuda:0 --dtype bf16`;
  - `systemctl --user restart fluxvla-tron2-step010000.service` after the
    normalization repair;
  - targeted pytest, Python bytecode compilation, ZeroMQ `ping`, `get_status`,
    and compressed MsgPack `predict_action` protocol checks;
  - `systemctl`, `journalctl`, `ss`, and `nvidia-smi` runtime inspection.
- Validation:
  - `pytest -q test/test_transforms/test_normalize.py test/test_ci_smoke.py`:
    6 passed; only existing third-party deprecation/runtime warnings remained;
  - Python bytecode compilation and `git diff --check` passed;
  - the service loaded the checkpoint, tokenizer, and normalization statistics,
    then reported ready on `tcp://127.0.0.1:3333`;
  - ZeroMQ `ping` returned `status=ok`;
  - a synthetic request containing three 224x224 RGB images and 16 joint values
    completed a full GPU forward pass and returned finite `float32` actions of
    shape `(1, 50, 16)`; measured server inference time was 0.738 seconds and
    round-trip time was 0.758 seconds. This was a protocol/compute smoke test,
    not a model-quality or robot-safety evaluation;
  - after the request, the service remained ready with one successful request;
    its process used approximately 20.6 GiB of the 24 GiB RTX 3090 memory.
- Remaining limitations and risks:
  - no physical robot command was issued; the robot client still needs the SSH
    tunnel and its own dry-run/safety validation before motion is enabled;
  - only approximately 3.4 GiB of GPU memory remains while this model is
    resident, so concurrent GPU workloads may cause out-of-memory failures;
  - the transient user service is running now but is not configured as a
    persistent boot-time service and has no automatic restart policy;
  - TensorFlow, robosuite, and OpenGL emitted non-blocking environment warnings
    during initialization; they did not prevent serving or the full request.

## 2026-08-03 20:49 CST — Enable Tron2 real-action mode

- Purpose: explicitly disable the robot client's dry-run guard at the user's
  request after network, ROS topic, controller-port, and inference-service
  checks were completed.
- Change:
  - set `inference.dry_run=False` in
    `configs/pi05/pi05_paligemma_tron2_lora_finetune.py`;
  - retained `action_layout='tron2_16'` and
    `enable_head_control=False` without changing any other runtime setting;
  - updated the deployment smoke-test expectation and the English/Chinese
    deployment documentation to state that this private config enables real
    actions; documented `inference.dry_run=True` as the required explicit
    override for later dry-run validation;
  - did not start the Power Computing Module client and did not send any robot
    command as part of this configuration change.
- Validation:
  - parsed the configuration with mmengine and verified that the resolved
    inference setting is `dry_run=False`, while `action_layout='tron2_16'`,
    `enable_head_control=False`, and `action_chunk=32` remained unchanged;
  - Python source compilation and `git diff --check` passed;
  - `pytest -q test/test_transforms/test_normalize.py test/test_ci_smoke.py`:
    6 passed with only existing third-party warnings;
  - the previous transient inference process was found to have received an
    external `SIGKILL` at 20:36 CST. No kernel OOM, CUDA Xid, or low-memory
    condition was recorded; it was restarted solely to finish validation;
  - after restart, the checkpoint and preprocessing pipeline loaded and ZMQ
    `get_status` returned `ready` on `127.0.0.1:3333`;
  - at the user's request, stopped the inference service after validation and
    verified that its unit was `inactive/dead`, port 3333 was released, and no
    model process remained on the GPU.
- Remaining risk: the current config has no `execute_horizon`, so a robot-side
  launch will execute the configured full 32-step action chunk. The physical
  emergency stop and operator supervision remain required, and a separate copy
  of this configuration on the Power Computing Module must be synchronized or
  overridden there before the change affects its client. The GPU inference
  service is intentionally stopped and must be started manually before use.
