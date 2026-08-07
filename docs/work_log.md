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

## 2026-08-04 01:49 CST — Synchronize step-030000 checkpoint

- Purpose: copy the latest requested training checkpoint from the training
  server into the matching local work directory without modifying or stopping
  the remote training job.
- Operation:
  - confirmed the remote checkpoint was a regular file and recorded its size,
    modification time, and SHA-256 digest before transfer;
  - confirmed the local target did not already exist and that approximately
    789 GiB of disk space was available;
  - transferred the file over SSH with resumable rsync using partial-file and
    append-verification support. One transient SSH disconnect occurred at about
    79%; rsync retained the partial file and resumed successfully;
  - copied only
    `work_dirs/pi05_two_tasks_lora_rank256_20260803_170124/checkpoints/step-030000-epoch-016-loss=0.0020.safetensors`;
    no remote file or training process was modified.
- Validation:
  - rsync completed with exit code 0 after the resumed transfer;
  - remote and local sizes are both 14,466,989,776 bytes;
  - remote and local modification times both resolve to
    `2026-08-04 01:09:49.082775268 +0800`;
  - the independently computed remote and local SHA-256 digests both equal
    `2a363378d51b7969b8165576be5273bcda26f92c5fcab19e648698d1b50f588a`;
  - `git check-ignore` confirmed the checkpoint remains excluded through the
    existing `work_dirs` ignore rule and was not staged or committed.

## 2026-08-04 01:55 CST — Publish repository to a private personal GitHub repo

- Purpose: configure GitHub CLI for the workstation and publish the current
  source repository to the user's personal account with private visibility.
- Configuration and scope:
  - installed official GitHub CLI 2.97.0 under `$HOME/.local/bin` after
    verifying the release archive against the official checksum file;
  - authenticated the personal GitHub account `emoPointer`, configured the
    repository-local commit identity with the GitHub noreply address, and added
    the workstation's existing ED25519 public key to that account. The private
    key was not read or uploaded;
  - created `emoPointer/FluxVLA-Tron2` and verified its visibility was
    `PRIVATE`;
  - kept the original `clearlab-sustech/FluxVLA-Tron2` remote as `upstream`
    and configured the personal private repository as `origin`;
  - inspected staged and untracked files, searched for common credential
    patterns, and confirmed that checkpoints, work directories, datasets,
    environment files, and the nested `src/libero` checkout remained ignored.
- Publication:
  - committed the reviewed deployment, dependency, CUDA-source, normalization,
    test, and documentation changes as local commit `02c9ff5`;
  - normal HTTPS Git push stalled in the network's chunked receive-pack path;
    a fixed-length retry also stalled, and GitHub SSH on ports 22 and 443 was
    blocked after connection establishment;
  - initialized the empty private repository with a temporary bootstrap commit,
    then used the GitHub Git Database API to upload 415 unique blobs, validating
    every returned Git object SHA;
  - recreated the initial and deployment trees. Their remote SHA values exactly
    matched the local trees (`cc3a1e1` and `889f511`), then moved `main` to the
    recreated deployment commit, leaving the bootstrap file outside the final
    branch tree;
  - the API-generated commit SHA differs from the local SHA because GitHub
    serialized the commit metadata differently, but the full file-tree SHA is
    identical. The published deployment tree contains 418 files.
- Validation:
  - `pytest -q test/test_transforms/test_normalize.py test/test_ci_smoke.py`:
    6 passed with five existing third-party warnings;
  - Python compilation and `git diff --check` passed;
  - repository visibility, default branch, remote commit, remote tree, and file
    count were queried again through the authenticated GitHub API;
  - no checkpoint, dataset, secret, or local environment file was included in
    the GitHub tree.

## 2026-08-04 10:30 CST — Synchronize fold-clothes step-030000 checkpoint

- Purpose: copy the requested fold-clothes checkpoint from the authorized
  training server into its distinct matching local work directory without
  modifying the source file or any remote process.
- Operation:
  - verified that the remote source was a regular file and not a symbolic
    link, then recorded its size, modification time, and SHA-256 digest before
    transfer;
  - confirmed that the local target did not exist and that more than 829 GB of
    disk space was available, then created only the required checkpoint parent
    directory;
  - copied only
    `work_dirs/pi05_paligemma_tron2_lora_finetune_fold_clothes_20260803/checkpoints/step-030000-epoch-001-loss=0.0122.safetensors`
    with rsync over SSH using partial-file and append-verification support;
  - rsync completed without a disconnect or retry and did not modify or stop
    any remote training process.
- Validation:
  - rsync completed with exit code 0;
  - remote and local sizes are both 14,466,989,776 bytes;
  - remote and local modification times are both
    `2026-08-04 05:20:52.464946634 +0800`;
  - SHA-256 was recomputed independently on both sides after transfer and both
    digests equal
    `986778ba22e10b6bc18d63d5805f0846c851d21425dca8740f3bcd7ff4f9a8bf`;
  - `git check-ignore -v` confirmed that the checkpoint is excluded by the
    existing `work_dirs` rule, and `git ls-files` confirmed it is not tracked.

## 2026-08-04 13:47 CST — Integrate tron2_env MoveJ/ServoJ operator

- Purpose: replace the local policy-action ServoJ implementation with the
  public `tron2_env` runtime while preserving MoveJ for task-ID-0 prepare-pose
  initialization and the existing ROS observation path.
- Upstream baseline:
  - inspected `limxdynamics/tron2_env` at commit
    `5b7b145229416f3731f61657e6fa71c89c37bc9d`;
  - pinned that exact commit in `requirements.txt` and installed
    `tron2-env==0.1.0` into the existing `fluxvla` conda environment without
    changing global Python packages.
- Implementation:
  - added the registered `Tron2EnvOperator`, which reuses `Tron2Operator` only
    for ROS image/state collection and delegates robot control to
    `tron2_env.WebsocketTransport` and `MotionController`;
  - made MoveJ and ServoJ mutually exclusive: prepare-pose execution stops and
    disconnects any ServoJ publisher, while the next policy chunk starts a
    fresh controller seeded from measured 18-dimensional robot state;
  - the current PI0.5 LoRA policy waypoints are fed at 30 Hz to the upstream
    interpolator, which publishes ServoJ at 300 Hz; one-waypoint chunks are
    supported;
  - added finite/shape/gripper validation, a default 0.2-radian transition
    guard from measured state through every policy waypoint, optional explicit
    16-dimensional joint bounds, connection checks, MoveJ completion timeout,
    and deterministic control cleanup;
  - switched all three Tron2 inference configurations and the runner default
    to `Tron2EnvOperator`, and updated both inference entry points so
    `Ctrl+C` executes cleanup;
  - updated English/Chinese deployment documentation and added hardware-free
    adapter tests.
- Validation actually executed:
  - `python -m compileall -q` passed for the changed Python modules, configs,
    scripts, and tests;
  - `pytest -q test/test_tron2_env_operator.py test/test_ci_smoke.py`: 8 passed
    with five pre-existing third-party warnings;
  - the upstream `tron2_env` test suite at the pinned commit: 26 passed;
  - `pip check`: no broken requirements;
  - installed-package provenance reports version `0.1.0`, the expected GitHub
    URL, and exact commit
    `5b7b145229416f3731f61657e6fa71c89c37bc9d`;
  - `git diff --check` passed.
- Remaining limitations and risks:
  - no ROS master, robot WebSocket, or physical robot was contacted during
    validation, so MoveJ/ServoJ hardware behavior remains untested;
  - the 0.2-radian transition guard is an engineering fail-safe, not a
    certified robot safety limit, and verified deployment-specific absolute
    joint limits have not yet been configured;
  - the public `tron2_env` runtime explicitly does not replace the robot's
    private low-level safety controller or the physical emergency stop.

## 2026-08-04 13:55 CST — Synchronize ServoJ remote client to the Power Computing Module

- Purpose: align the lightweight client at `guest@10.192.1.4` with the new
  `Tron2EnvOperator` path while keeping model inference on the GPU server and
  avoiding a full Conda/PyTorch installation on the robot computer.
- Preserved deployment-specific code:
  - did not overwrite the guest's NumPy-based, torch-free
    `base_inference_runner.py`;
  - did not overwrite its legacy `Tron2Operator`, which remains available for
    the older MoveJ path;
  - retained the guest launcher's safe SSH ControlMaster, occupied-port check,
    and forward-success validation when merging the local changes.
- Synchronized scope: the remote launcher, inference entry point, PI0.5 LoRA
  config, operator registration, `Tron2EnvOperator`, and Tron2 inference runner.
  The launcher now enables `FLUXVLA_REMOTE_CLIENT_ONLY`, selects the guest's
  existing `/home/guest/.venvs/fluxvla-tron2` Python when needed, and performs
  ROS/`tron2_env`/operator import checks before inference starts.
- Environment: built the pinned upstream commit as a pure-Python wheel and
  installed `tron2-env==0.1.0` with `--no-deps` into the existing lightweight
  guest venv. The transferred wheel's SHA-256 was
  `108eac8d1d2a39e580b07eee727965825efeb7a8b2a96ee9595ae606cdab270e`.
- Final control timing: policy waypoints use 30 Hz, the upstream ServoJ
  interpolator publishes at 300 Hz, and initialization MoveJ uses 2.0 seconds.
  The earlier 20 Hz setting was a historical test setting and was superseded
  at the user's request.
- Validation actually executed: guest shell syntax and Python compilation
  passed; the lightweight environment imported ROS, `tron2_env`, and
  `Tron2EnvOperator`; MMEngine resolved the three timing values as
  `30`, `2.0`, and `300.0`; `pip check` reported no broken requirements; and
  synchronized file SHA-256 values were compared with the local sources.
  Local focused tests also passed (8 tests), along with local compilation,
  shell syntax, `pip check`, and `git diff --check`.
- Safety and recovery: no inference client, SSH forwarding tunnel, robot
  WebSocket controller, or robot action was started. Replaced guest files were
  retained under `.sync-backups/20260804_135355`, with the interim 20 Hz config
  additionally retained under `.sync-backups/20260804_135355-policy-30`.

## 2026-08-04 14:15 CST — Move all policy observations to TRON2 Bridge WebSocket

- Purpose: correct the earlier ROS-observation interpretation and implement
  the requested `tron2_env` deployment model in which the FluxVLA client gets
  all policy inputs through WebSocket. The Bridge and robot-control interfaces
  remain separate network connections.
- Upstream and deployed interface verification:
  - rechecked the official `tron2_env` README and implementation at pinned
    commit `5b7b145229416f3731f61657e6fa71c89c37bc9d`;
  - confirmed that `BridgeObservationProvider` produces the OpenPI layout
    `[left7, left_gripper, right7, right_gripper, head2]` and aligned
    `cam_high`, `cam_left_wrist`, and `cam_right_wrist` images;
  - queried the running PCM Bridge at `wss://10.192.1.4/bridge/ws` read-only and
    received all five configured streams: three image topics, 16 joint
    positions, and two gripper positions.
- Implementation:
  - made `Tron2EnvOperator` independent of `Tron2Operator`; it no longer
    initializes ROS or subscribes to ROS topics;
  - added fail-fast Bridge endpoint/topic validation, first-complete-observation
    startup validation, per-observation shape/type/finite checks, and clean
    Bridge thread shutdown;
  - kept MoveJ initialization and 300 Hz ServoJ policy control on the separate
    `ws://10.192.1.2:5000` robot connection, with 30 Hz policy waypoints and the
    existing transition guard;
  - updated `Tron2InferenceRunner` to run without `rospy` and to preserve both
    historical FluxVLA state layouts while consuming the official 18-D Bridge
    state;
  - replaced ROS topic configuration with explicit Bridge host/path/topic
    mappings in all three Tron2 configs, updated the lightweight launcher
    preflight, requirements, tests, and deployment documentation.
- Environment and PCM synchronization:
  - installed the official Bridge optional dependency `websockets==16.1.1` in
    the local `fluxvla` conda environment and the existing PCM lightweight venv;
  - synchronized the launcher, PI0.5 LoRA deployment config, operator, runner,
    and requirements to `/home/guest/FluxVLA-Tron2`;
  - retained replaced PCM files under
    `.sync-backups/20260804_141400-bridge-websocket`.
- Validation actually executed:
  - focused hardware-free tests: 11 passed with five existing third-party
    warnings;
  - local compile, shell syntax, dependency, and diff checks passed;
  - a local read-only real Bridge check, with ROS variables unset and robot
    control explicitly disabled, returned an 18-D state and three
    `480x640x3` RGB images;
  - the same read-only check passed inside the PCM lightweight venv, and
    `rospy` was absent from `sys.modules`;
  - PCM shell syntax, Python compilation, config parsing, and `pip check`
    passed; synchronized source hashes were compared with the local files.
- Safety boundary: no inference episode, SSH inference tunnel, MoveJ, ServoJ,
  gripper command, or robot-control WebSocket was started. The Bridge uses a
  self-signed certificate (`bridge_verify_tls=False`), which is acceptable
  only on the isolated trusted robot LAN and is not suitable for exposure to
  an untrusted network.

## 2026-08-04 14:42 CST — Block rejected ServoJ chunks, lock the head, and bind tasks to checkpoints

- Problem background:
  - a real client run rejected the first policy waypoint because joint 3 was
    `1.378144` rad away from the measured pose, exceeding the configured
    `0.2`-rad guard;
  - the head moved briefly even though the policy action was rejected;
  - one robot client must support several task/checkpoint combinations without
    retaining an unrelated prompt map.
- Root cause and safety fix:
  - the first policy chunk was validated only after the upstream
    `MotionController.start()` had begun its 300 Hz measured-hold ServoJ
    publisher; the rejected chunk therefore did not guarantee zero ServoJ
    frames;
  - the operator now validates the first complete chunk before starting the
    publisher, validates again against the exact post-start feedback, and
    disconnects control on any failure;
  - Bridge and robot-control 18-D state sources must agree within `0.1` rad;
    policy head trajectories are rejected, the controller holds the measured
    control-feedback head position, and ServoJ is blocked if that measured head
    drifts by more than `0.05` rad;
  - delta errors now report both measured and target joint values.
- Multi-task metadata:
  - the GPU service reads `task_descriptions` and `action_layout` from the
    selected checkpoint work directory's `config.json` and exposes them through
    a read-only ZMQ endpoint;
  - the robot client adopts that map during setup, prints every available task,
    rejects empty/unknown IDs and invalid repeat counts, and verifies the action
    layout before inference;
  - Tron2 service startup now fails if checkpoint-local task metadata is absent,
    rather than silently falling back to the robot/launch config.
- Dataset verification on the authorized lim server:
  - `/root/data/lerobot_dataset_task1/meta/tasks.jsonl` contains exactly
    `Put the flowers in the vase` across all 100 episodes;
  - `/root/data/lerobot_dataset_task2/meta/tasks.jsonl` contains exactly the
    four dolls/pens and pink/gray basket combinations across all 200 episodes;
  - the declared task sets exactly matched the task strings used by every
    episode;
  - those five exact strings match task IDs 1-5 in the local fold-clothes
    checkpoint `config.json`; task ID 6 is `fold clothes` and comes from the
    separate fold-clothes training data.
- PCM synchronization and read-only validation:
  - backed up the replaced operator, runner, and PI0.5 LoRA config under
    `/home/guest/FluxVLA-Tron2/.sync-backups/20260804_144200-servoj-head-multitask`;
  - synchronized those three files to `guest@10.192.1.4` and verified matching
    SHA-256 values;
  - PCM compilation and config parsing passed; a real Bridge-only observation
    returned one 18-D state and three `480x640x3` images with
    `connect_websocket=False`, no control transport, and no `rospy` import.
- Local validation actually executed:
  - focused compile and tests passed: 18 tests, with five existing third-party
    warnings;
  - `git diff --check` passed;
  - previous no-control model dry runs at the measured prepare pose produced
    first-step arm deltas of `0.02227` rad for the fold prompt and `0.03961` rad
    for the banana prompt, both below the `0.2`-rad guard.
- Remaining risks and deployment state:
  - no MoveJ, ServoJ, gripper command, inference episode, or control WebSocket
    was started during this correction;
  - the original `1.378144`-rad input pair was not logged by the old code, so
    its upstream model/state cause cannot be reconstructed exactly;
  - no inference service was listening on local ports 3333 or 5555 at the final
    check. The service must be started with the synchronized code so the client
    can retrieve checkpoint metadata before any subsequent dry run or real
    test.

## 2026-08-04 14:50 CST — Correct mixed-task checkpoint deployment metadata

- Correction: the checkpoint
  `pi05_two_tasks_lora_rank256_20260803_170124/step-030000-epoch-016-loss=0.0020`
  is a five-prompt model trained from the combined task1 and task2 datasets. Its
  saved `inference.task_descriptions` contained a stale banana example and must
  not be interpreted as the training task set.
- Evidence:
  - both the local and original lim-server `config.json` list
    `./datasets/lerobot_dataset_task1` and
    `./datasets/lerobot_dataset_task2` under the training dataset;
  - both saved configs independently retain the unrelated banana inference
    example, confirming a metadata/configuration defect rather than a different
    checkpoint;
  - the lim-server task and episode metadata previously verified five exact
    training prompts: flowers, and the four dolls/pens × pink/gray basket
    combinations.
- Implementation:
  - added a checkpoint-local `deployment_metadata.json` sidecar containing the
    five exact prompts as task IDs 1-5 and the `tron2_16` action layout;
  - changed server metadata resolution to prefer this explicit sidecar over the
    saved training config while retaining `config.json` compatibility for older
    correctly described work directories;
  - added a regression test proving an explicit sidecar overrides a stale
    training inference prompt.
- Validation actually executed:
  - focused compilation and tests passed: 19 tests with five existing
    third-party warnings;
  - JSON parsing, `git diff --check`, and checkpoint-sidecar loading passed;
  - resolved metadata reports source `deployment_metadata.json`, layout
    `tron2_16`, and task IDs 1-5 with the expected exact prompts.
- Deployment state: no inference service was listening on port 3333 at the
  final check, and no robot process or action was started. The next server start
  with this checkpoint will load the corrected task table automatically.

## 2026-08-04 14:56 CST — Restrict Bridge/control cross-check to ServoJ startup

- Problem: a real task-3 run completed seven 32-step policy chunks (through
  published step 224) and then stopped because Bridge joint 10 was `-1.1429`
  rad while the 200 Hz control feedback was `-1.3024` rad, exceeding the
  `0.1`-rad cross-source threshold by reporting a `0.1595`-rad mismatch.
- Root cause: the Bridge state is captured before remote model inference,
  whereas the control state is sampled immediately before execution. Applying
  a static cross-source threshold to every chunk compares different timestamps
  while the robot is moving and therefore produces expected false positives.
- Fix:
  - retain the `0.1`-rad Bridge/control layout/source check before the first
    ServoJ publisher starts and repeat it against the exact post-start state;
  - skip only that stale-snapshot comparison for later chunks on an already
    active controller;
  - keep the live-control-to-first-waypoint `0.2`-rad delta guard, all inter-
    waypoint delta checks, joint limits, gripper validation, connection checks,
    and head lock unchanged for every chunk.
- Validation actually executed:
  - added a regression test proving an active ServoJ controller accepts an
    expected stale Bridge snapshot without being disconnected;
  - focused compilation and tests passed: 20 tests with five existing
    third-party warnings; `git diff --check` passed;
  - backed up the PCM operator under
    `.sync-backups/20260804_145500-dynamic-state-crosscheck`, synchronized the
    corrected operator, compiled it in the lightweight venv, and verified the
    local/PCM SHA-256 digest
    `fa7bbff7da25ddedc75d3836cbfe22c143fd24e098e5393795f3ff1f87a2e511`.
- Safety boundary: no client, inference service, Bridge observation, control
  WebSocket, MoveJ, ServoJ, gripper command, or real robot action was started
  during this correction.

## 2026-08-04 15:00 CST — Relax initial Bridge/control mismatch threshold

- User-requested tuning: increased
  `max_state_source_mismatch_rad` from `0.1` to `0.2` rad for the startup-only
  Bridge/control feedback consistency check.
- Scope: updated the `Tron2EnvOperator` default and all three Tron2 inference
  configurations. The live-control-to-first-waypoint and inter-waypoint
  `max_servoj_step_rad=0.2` guard, head lock, joint limits, gripper validation,
  and connection checks remain unchanged.
- Validation actually executed:
  - focused compilation and tests passed: 20 tests with five existing
    third-party warnings; `git diff --check` passed;
  - backed up the PCM files under
    `.sync-backups/20260804_150000-startup-threshold-02`, synchronized the
    operator and PI0.5 LoRA config, compiled both, parsed the effective PCM
    threshold as `0.2`, and matched local/PCM SHA-256 digests.
- Safety boundary: no robot client or action was started. The repeated
  14:51 log predates both the cross-check timing fix and this threshold change.

## 2026-08-04 15:09 CST — Set startup source-mismatch threshold to 0.5 rad

- User-requested configuration change: increased the startup-only
  `max_state_source_mismatch_rad` from `0.2` to `0.5` rad (approximately
  28.65 degrees) in the operator default and all Tron2 inference configs.
- Safety scope preserved: the cross-source comparison remains disabled for
  later moving chunks, while `max_servoj_step_rad` remains `0.2` rad for the
  live control state and every policy transition. Head lock, joint limits,
  gripper validation, and connection checks are unchanged.
- Validation actually executed:
  - focused compilation and tests passed: 20 tests with five existing
    third-party warnings; `git diff --check` passed;
  - backed up the PCM files under
    `.sync-backups/20260804_150800-startup-threshold-05`, synchronized the
    operator and PI0.5 LoRA config, compiled them, and verified effective PCM
    values `startup_mismatch_limit=0.5` and `waypoint_limit=0.2`;
  - local and PCM SHA-256 digests matched for both synchronized files.
- No client, WebSocket control session, or robot action was started.

## 2026-08-04 20:50 CST — Port relaxed ServoJ rejection handling to no-RTC client

- Objective: retain only the requested policy-action guards on the no-RTC
  `Tron2InferenceRunner` baseline: 0.2-rad adjacent ServoJ target delta, head
  lock/drift check, and shape/finite/gripper data-integrity validation.
- Changes:
  - disabled the duplicate Bridge/control feedback mismatch block by setting
    `max_state_source_mismatch_rad=None` in the operator default and all TRON2
    configs;
  - classify an invalid live 18-D state as a recoverable action-validation
    `ValueError`;
  - a rejected trajectory no longer disconnects an already active
    MotionController. No replacement command is issued, so the last accepted
    ServoJ target remains active; connection/protocol/controller faults still
    disconnect and propagate.
- Focused hardware-free tests cover the disabled source comparison, retained
  delta/head/data checks, first-chunk rejection without starting ServoJ, and
  active-controller target preservation. No robot process or action was
  started for this change.

## 2026-08-04 21:15 CST — Restore the non-RTC PCM client with local keyboard control

- Objective: retire the experimental TRON2 overlap/remote-RTC deployment while
  retaining the requested robot-computer keyboard workflow and the relaxed
  ServoJ rejection behavior.
- Implementation:
  - the active PI0.5 deployment again uses the standard
    `Tron2InferenceRunner`, `action_chunk=32`, and the original prediction
    request/response protocol; no overlap queue, guidance prefix, or
    inference-time RTC is active;
  - replaced the repeat-count prompt with a PCM-local TTY state machine: enter
    and confirm a checkpoint task ID, `b` starts continuous sequential chunks,
    `s` stops future inference/chunk acceptance while an already accepted full
    chunk finishes, and idle-only `r` runs the former task-0 MoveJ prepare-pose
    sequence;
  - rejected action-validation chunks log a hold warning and leave the last
    accepted ServoJ target unchanged instead of terminating the client;
  - retained the independent lightweight-client import path so the PCM does
    not need PyTorch. This compatibility was separated from the retired RTC
    transport and runners after a PCM import check exposed the dependency.
- PCM synchronization:
  - confirmed no inference, tunnel, model-serving, or MotionController process
    was running before and after synchronization;
  - backed up all replaced runtime files and both RTC-only runner files under
    `/home/guest/FluxVLA-Tron2/.sync-backups/20260804_210742-no-rtc-keyboard`;
  - synchronized the three TRON2 configs, operator, standard runner/base
    runner, standard serializers/server files, lightweight utility imports,
    and launcher to `guest@10.192.1.4`;
  - removed only `tron2_overlap_inference_runner.py` and
    `tron2_remote_rtc_inference_runner.py` after backup. The upstream
    `tron2_rtc_inference_runner.py` remains present but is neither registered
    by the lightweight path nor selected by the active configuration.
- Validation actually executed:
  - focused local hardware-free tests passed (28 tests with five existing
    third-party warnings), including the dedicated subprocess import
    regression; local bytecode compilation, YAPF, shell syntax, torch-free
    import, and `git diff --check` passed;
  - on the PCM, Python 3.10 bytecode compilation and launcher shell syntax
    passed; the resolved config reported `Tron2InferenceRunner`, chunk size 32,
    no `rtc_config`/`execute_horizon`, a 0.2-rad ServoJ step limit, disabled
    duplicate source mismatch check, and head lock enabled;
  - the PCM imported the operator, standard runner, and terminal key reader
    with no `torch` module loaded; a pseudo-TTY test read a single `s` key and
    verified that terminal attributes were restored;
  - SHA-256 values matched for all 13 synchronized runtime files.
- Safety boundary: no inference client/server, SSH tunnel, Bridge observation,
  robot-control WebSocket, MoveJ, ServoJ, gripper command, or physical action
  was started during this migration. The two removed RTC-only files are
  recoverable from the recorded PCM backup.

## 2026-08-05 11:37 CST — Synchronize the six-task LoRA adapter bundle

- Purpose: copy the requested task1/task2 LoRA adapter and its deployment
  configuration from the authorized training server into the matching local
  `work_dirs/pi05_task1_task2_lora_rank256_4gpu_bs12_20260805_005442`
  directory without transferring optimizer state, logs, W&B data, or the
  14–17 GB merged/training checkpoints.
- Transfer:
  - resumed the interrupted `adapter_model.safetensors` download with one
    serial `rsync --partial --append-verify` process and SSH keepalives;
  - binding SSH to local address `192.168.110.44` timed out in SYN-SENT during
    two attempts, while a read-only port probe showed `wlo1` was reachable, so
    the transfer used `ssh -B wlo1`; zlib level-1 stream compression reduced
    the remaining network transfer without changing the resulting file;
  - synchronized the remote root-level `README.md`, `adapter_config.json`,
    `config.json`, `config.yaml`, `dataset_statistics.json`, and
    `llm_backbone_config.json`. The remote work directory has no tokenizer or
    VLM-backbone configuration directory to copy.
- Deployment metadata:
  - verified the source dataset task metadata directly under
    `lerobot_dataset_task1/meta/tasks.jsonl` and
    `lerobot_dataset_task2/meta/tasks.jsonl`;
  - added checkpoint-local `deployment_metadata.json` with action layout
    `tron2_16` and task IDs 1–6. Task 6 is `Put a doll into the gray basket,
    and put the other doll into the pink basket.`
- Validation actually executed:
  - local and remote byte sizes and SHA-256 digests match for the adapter and
    all six synchronized support files; the adapter is 1,133,695,816 bytes
    with SHA-256
    `d8835b237bff4383296e198ce3669ba2bda09975f24561c3d7aec95176d3cd7f`;
  - JSON and YAML parsing passed; direct safetensors-header validation found
    838 BF16 tensors and an exact final data extent;
  - the adapter contains PEFT `base_model.*`/LoRA keys, so it is not a merged
    model checkpoint that the current serving path can load directly with
    `strict=True`.
- Safety boundary: no training, inference service, PCM client, WebSocket
  connection, or robot-control process was started. The synchronized adapter,
  checkpoint-local configuration, and deployment metadata remain ignored
  under `work_dirs/` and were not added to Git; only this work-log record is
  intended for version control.

## 2026-08-06 11:05 CST — Add single-process train-time/inference-time prefix RTC deployment

- Purpose: deploy a checkpoint trained with clean-prefix RTC conditioning on
  the GPU workstation without splitting RTC state across the robot client and
  a ZeroMQ model server, while preserving the interactive `b`/`s`/idle-only
  `r` safety workflow.
- Read-only network validation:
  - inspected the installed `tron2_env` transport before connecting and
    confirmed that `init_joints=None` and `init_head=None` disable its implicit
    initialization motion;
  - subscribed once to `wss://10.192.1.4/bridge/ws` and received three finite
    `480x640x3` images plus one finite 18-D state;
  - connected to `ws://10.192.1.2:5000` at a reduced 10 Hz polling rate and
    received one finite 18-D state using only
    `request_get_joint_state`/`request_get_limx_2fclaw_state`;
  - disconnected both endpoints without sending MoveJ, ServoJ, MoveH, gripper,
    or emergency-stop requests.
- Implementation:
  - synchronized the training-time RTC config (`max_delay=10`, exponential
    delay sampling) and added a separate single-process deployment config using
    `Tron2RTCInferenceRunner`, `method='prefix'`, no remote-inference client,
    asynchronous execution, and a 50-frame chunk matching the model horizon;
  - retained the inherited real-action and 30 Hz policy settings rather than
    silently changing established deployment conditions;
  - local model mode now adopts checkpoint-local task/action metadata just as
    the remote client does;
  - added an operator wait primitive and kept the active key monitor running
    until the accepted asynchronous trajectory feeder finishes. The runner
    does not report idle or accept reset `r` before that drain completes;
  - added validation that prevents the local RTC runner from being combined
    with the old remote-inference protocol or a truncated action chunk, and a
    one-time warning when the dynamic inference prefix is outside the
    train-time sampled range.
- Validation actually executed:
  - mmengine resolved the deployment to local
    `Tron2RTCInferenceRunner`, `action_chunk=50`, prefix RTC, 30 Hz, and the two
    expected WebSocket endpoints;
  - checkpoint-local metadata resolution returned `tron2_16` and task IDs
    1–6 from `deployment_metadata.json`;
  - YAPF, Python bytecode compilation, and `git diff --check` passed;
  - focused hardware-free tests passed: 34 tests with five existing
    third-party warnings, including RTC config inheritance, keyboard/reset
    drain ordering, and non-canceling async trajectory wait behavior.
- Remaining limitations and risks:
  - an end-to-end local-model dry run was not started because an unrelated user
    serving process (PID 495264) already occupied about 19.8 GiB of the 24 GiB
    RTX 3090; that process was left untouched;
  - the available complete local checkpoint used for metadata validation was
    trained without RTC. The remote RTC run remained active at the final check;
    its newest completed merged checkpoint was step 36000 of 40000. A selected
    RTC checkpoint still needs to be synchronized before model-quality or
    real-robot RTC validation;
  - the prior RTX 3090 inference measurement was about 0.738 seconds (roughly
    22 policy frames at 30 Hz), outside the training configuration's `[0, 10)`
    prefix distribution. The new RTC checkpoint needs a no-control latency
    measurement before real execution.

## 2026-08-06 11:18 CST — Stabilize task IDs for the expanded task1/task2 set

- Purpose: keep one forward-compatible task-ID table while preventing a
  checkpoint from advertising prompts that were absent from its training
  data.
- Source inspection:
  - read the authorized training server's
    `lerobot_dataset_task1/meta/tasks.jsonl` and
    `lerobot_dataset_task2/meta/tasks.jsonl`; task1 contains the flower prompt,
    while task2 contains ten represented prompts in dataset `task_index`
    order;
  - confirmed the separate task3 metadata uses the exact prompt
    `fold clothes` and that the active RTC training command contains only
    task1 and task2.
- Changes:
  - assigned stable global IDs 1–12: flower is ID 1, the ten task2 prompts are
    IDs 2–11, and the future fold-clothes task is ID 12;
  - added the global table to the single-process RTC deployment config;
  - added checkpoint-local metadata that exposes only IDs 1–11 for the active
    RTC task1/task2 work directory and only ID 12 for the existing standalone
    fold-clothes checkpoint;
  - synchronized the RTC metadata to the matching training-server work
    directory. The local and remote files have matching SHA-256
    `df57c2766951c34959be523fa48150db60a6a208b7b4298d7c7c392df1055e4f`.
- Validation actually executed:
  - parsed both local JSON sidecars and the synchronized remote JSON;
  - checkpoint metadata resolution returned IDs 1–11 for the RTC work
    directory and ID 12 for the fold-clothes work directory;
  - YAPF reported no remaining formatting diff, `git diff --check` passed,
    Python bytecode compilation passed, and all 18 focused CI-smoke tests
    passed with five existing third-party warnings.
- Remaining boundary: the global ID 12 is intentionally reserved for a future
  combined checkpoint, but the currently training RTC checkpoint does not
  expose it. A future combined work directory must receive checkpoint-local
  metadata containing IDs 1–12 after fold-clothes data is actually included.

## 2026-08-06 12:10 CST — Synchronize the final 40k task1/task2 RTC checkpoint

- Purpose: place the requested final merged RTC checkpoint and all files
  required by single-process inference in the matching local work-directory
  structure.
- Transfer:
  - used resumable `rsync --partial --append-verify` over the authorized SSH
    connection to copy
    `step-040000-epoch-004-loss=0.0035.safetensors` into the local
    `checkpoints/` directory;
  - synchronized `config.json`, `config.yaml`, `dataset_statistics.json`,
    `deployment_metadata.json`, `llm_backbone_config.json`, README and the
    small adapter manifest files;
  - copied the six tokenizer assets from the training repository's
    `checkpoints/pi05_base` into this work directory's `tokenizer/` directory,
    matching the inference dataset's checkpoint-local tokenizer resolution;
  - intentionally excluded `.pt` checkpoints, older steps, the LoRA adapter
    weights, training logs and metric files because the selected merged
    checkpoint does not use them during inference.
- Validation actually executed:
  - local size is 14,466,989,776 bytes and local/remote SHA-256 both equal
    `a4c8115e33cad1439f49414b906c94bb5ccc58b69f7fa18633e8b8e276af8093`;
  - all synchronized root metadata and tokenizer files match their remote
    SHA-256 values;
  - JSON and YAML parsing passed; safetensors-header validation found 812
    tensors and an exact final data boundary;
  - the checkpoint-local Gemma tokenizer loaded with offline mode enabled, the
    configured inference dataset built successfully, and deployment metadata
    resolved `tron2_16` with task IDs 1–11.
- Safety boundary: no model was loaded onto the GPU, no inference or robot
  client was started, and no observation or control WebSocket was contacted.

## 2026-08-06 13:57 CST — Relax the PI0.5 LoRA ServoJ transition guard

- Purpose: prevent repeated holds observed at the first transition of newly
  predicted chunks while retaining the adjacent-target safety guard.
- Evidence: the running non-RTC deployment rejected joint-13 transition-zero
  deltas between 0.212055 and 0.292077 rad against the previous 0.2-rad
  threshold and held the last accepted ServoJ target.
- Change: raised only the PI0.5 LoRA deployment's
  `max_servoj_step_rad` from 0.2 to 0.5 rad. The RTC deployment inherits this
  operator value. The operator default and other model configurations remain
  unchanged at 0.2 rad; finite-value, joint-limit, gripper, head-lock and
  300-Hz interpolation behavior remain enabled.
- Runtime boundary: the already-running process retains the old value loaded
  at construction time and must be stopped and restarted before 0.5 rad takes
  effect.
- Validation actually executed: both the standard PI0.5 LoRA config and its
  single-process RTC derivative resolved the limit to 0.5 rad; YAPF and
  `git diff --check` passed, and all 18 focused CI-smoke tests passed with five
  existing third-party warnings.

## 2026-08-06 14:45 CST — Fix inference-time RTC prefix to nine steps

- Purpose: use a deterministic prefix length at the upper edge of the
  checkpoint's train-time RTC range for the requested deployment test.
- Change: set the single-process `Tron2RTCInferenceRunner` configuration to
  `enabled=True`, `method='prefix'`, and `prefix_len=9`. The action horizon
  remains 50, so each conditioned prediction locks nine actions from the
  previous chunk and generates the remaining 41.
- Scope: this does not change the trained weights or train-time distribution
  (`max_delay=10`, exponential). It also does not change the separate standard
  non-RTC runner.
- Runtime boundary: no inference or robot-control process was running while
  this configuration was changed; a new RTC process must be started for the
  fixed prefix to take effect.
- Validation actually executed: the resolved deployment reported
  `Tron2RTCInferenceRunner`, local model mode, a 50-step horizon, asynchronous
  execution, `prefix_len=9`, and the 0.5-rad ServoJ transition limit; YAPF and
  `git diff --check` passed, and all 18 focused CI-smoke tests passed with five
  existing third-party warnings.

## 2026-08-06 21:45 CST — Synchronize the 18k task3 fold-clothes RTC checkpoint

- Purpose: place the requested standalone fold-clothes RTC checkpoint in an
  isolated local work directory without mixing it with the existing non-RTC
  fold checkpoint or task1/task2 RTC checkpoint.
- Transfer:
  - after the requested target changed from 16k to 18k, used resumable
    `rsync --partial --append-verify` over the authorized SSH connection to
    copy only `step-018000-epoch-001-loss=0.0137.safetensors`;
  - synchronized `config.json`, `config.yaml`, `dataset_statistics.json`,
    `llm_backbone_config.json`, README and the adapter manifest, and populated
    the checkpoint-local tokenizer from the matching local PI0.5 base assets;
  - intentionally excluded the 16k checkpoint, `.pt` training states, the 18k
    adapter-only weight, logs and metrics because they are not needed for the
    selected merged-checkpoint inference path.
- Deployment metadata:
  - verified the source dataset prompt from
    `lerobot_dataset_task3/meta/tasks.jsonl` is `fold clothes`;
  - added a checkpoint-local sidecar advertising only the previously reserved
    global task ID `12`, so switching to this single-task checkpoint cannot
    reinterpret or advertise the multi-task IDs 1–11;
  - recorded training RTC settings from the saved config: uniform sampling
    over explicit delays `[0, 5, 10, 19]` with `max_delay=20`.
- Validation actually executed:
  - the local file size is 14,466,989,776 bytes and local/remote SHA-256 both
    equal `5966e6f80cc7d30060bb45fb589871c6e41a1d35aaaf1531c5abfd02645cf2e0`;
  - safetensors header validation opened the file successfully and found 812
    tensors;
  - JSON validation and deployment metadata resolution passed with
    `action_layout=tron2_16` and only `task_descriptions={'12': 'fold clothes'}`;
  - confirmed that no 16k checkpoint exists in the new local directory.
- Safety boundary: no model was loaded onto the GPU, no inference process was
  started, and no robot observation or control connection was opened.

## 2026-08-06 22:17 CST — Add six-frame ServoJ stream-recovery blending

- Purpose: avoid an abrupt jump when policy waypoint delivery resumes after a
  trajectory has drained and `MotionController` has been holding its final
  ServoJ target.
- Implementation:
  - added `recovery_blend_frames` to `Tron2EnvOperator`, with an explicit
    deployment value of six frames for all tracked Tron2 configurations;
  - record the last successfully commanded arm/head waypoint and gripper
    values, detect natural trajectory drain, and linearly blend the next six
    30 Hz policy frames from that held command to the recovered trajectory;
  - normal asynchronous preemption does not mark the stream as drained, so
    ordinary RTC chunk replacement is not unnecessarily blended;
  - MoveJ/reset and control-WebSocket restart clear the stored hold state;
    finite-value, gripper-range, head-lock, joint-limit and 0.5-rad PI0.5 LoRA
    waypoint-delta checks remain in force, including a second delta check on
    the blended trajectory.
- Validation actually executed:
  - Python bytecode compilation of the modified operator passed;
  - hardware-free operator and CI-smoke tests passed: 34 tests with five
    existing third-party warnings;
  - new tests verify exact 1/6 through 6/6 arm and gripper blending after a
    drained trajectory, verify that replacement of an active asynchronous
    trajectory does not trigger recovery blending, and verify that MoveJ reset
    clears the old recovery origin.
- Safety/runtime boundary: no inference model was loaded, no Bridge or robot
  WebSocket was contacted, and no robot command was sent. A running deployment
  must be restarted, and any separate PCM checkout must receive the source and
  configuration changes before this behavior can take effect there.

## 2026-08-06 22:31 CST — Add active-chunk smoothstep boundary blending

- Purpose: smooth normal asynchronous RTC chunk replacement separately from
  the previously added queue-drain recovery path.
- Implementation:
  - added `chunk_boundary_blend_enabled`, `chunk_boundary_blend_frames`, and
    `chunk_boundary_blend_scope` to `Tron2EnvOperator`;
  - track the prepared executable trajectory and the next unissued index under
    the control lock; when an active feeder is preempted, atomically copy its
    remaining actions before starting the replacement feeder;
  - smoothstep-blend corresponding old/new actions over at most six frames,
    using `u^2 * (3 - 2u)` weights; a feeder stopped before issuing any action
    is not allowed to contribute an unexecuted plan;
  - enabled the feature for the PI0.5 LoRA configuration inherited by local
    RTC deployment, with six frames and `arm` scope. Gripper commands remain
    unfiltered and the locked head is unchanged;
  - preserve the separate six-frame drained-stream recovery path, MoveJ state
    reset, and post-blend ServoJ waypoint-delta validation.
- Validation actually executed:
  - YAPF and Python bytecode compilation passed;
  - hardware-free operator and CI-smoke tests passed: 35 tests with five
    existing third-party warnings;
  - the new regression test verifies the exact smoothstep weights against the
    old unissued trajectory and confirms that `arm` scope leaves grippers at
    the new chunk values; the existing disabled-switch preemption test remains
    passing.
- Safety/runtime boundary: no model, Bridge connection, control WebSocket, or
  robot command was used. The changed process must be restarted; an external
  PCM checkout must be synchronized separately.
## 2026-08-06 23:08 CST — Disable active-chunk boundary blending

- Set `inference.operator.chunk_boundary_blend_enabled=False` in the PI0.5
  TRON2 LoRA deployment config at operator request.
- Kept the independent `recovery_blend_frames=6` behavior enabled for stream
  recovery; no controller implementation or safety limit was changed.
- Validated the resolved Python config and confirmed boundary blending is
  disabled while recovery blending remains six frames.

## 2026-08-06 23:14 CST — Synchronize SeetaCloud task1/task2 RTC checkpoint

- Used a resumable transfer for
  `step-022000-epoch-002-loss=0.0046.safetensors` from the authorized
  SeetaCloud training host into the matching local work directory.
- The gateway did not execute non-interactive rsync/SFTP requests, so the
  transfer used an HTTP Range endpoint bound only to remote loopback through
  an SSH tunnel; the temporary endpoint and tunnel were stopped afterward.
- Validation actually executed: remote and local sizes both equal
  14,466,989,776 bytes; SHA-256 on both ends equals
  `8f683100bb38a0da8ad144b6b7ef7277835ebcdb21c6b18a56b769d9b630efbc`;
  the safetensors header contains 812 tensors and its maximum data offset
  lands exactly at EOF. The `.part` file was atomically renamed only after
  these checks passed.
- No model, inference, training, Bridge connection, or robot control process
  was started.

## 2026-08-06 23:28 CST — Add keyboard-toggled issued-action recording

- Purpose: collect directly comparable RTC and non-RTC action streams without
  recording unexecuted tails from asynchronously preempted chunks.
- Implementation:
  - added `l` as an idle/running toggle that opens and closes one JSONL
    session under configurable `inference.action_record_dir` (default
    `work_dirs/action_records`);
  - attached the recorder after a validated 30 Hz policy waypoint is accepted
    by the ServoJ controller, recording issue timestamps, task/prompt,
    trajectory and frame indices, RTC state, effective prefix length, and the
    16- or 18-dimensional action in training layout;
  - kept 300 Hz interpolation samples and unissued preempted-chunk tails out of
    the dataset, and moved JSON serialization/filesystem writes to a background
    queue so the ServoJ feeder does not perform disk I/O;
  - RTC and non-RTC runners share the same implementation; dry run records no
    action rows because it sends no robot commands.
- PCM synchronization:
  - atomically synchronized the runner, WebSocket operator, and PI0.5 LoRA
    config to `/home/guest/FluxVLA-Tron2` after matching incoming SHA-256
    values;
  - preserved the replaced PCM files under
    `/home/guest/fluxvla-action-record-backup.Hp73DM`;
  - no inference/client process was running during synchronization.
- Validation actually executed:
  - local Python compilation and `git diff --check` passed;
  - 38 hardware-free runner/operator tests passed with five existing
    third-party warnings;
  - the PCM's selected minimal runtime
    `/home/guest/.venvs/fluxvla-tron2/bin/python` passed a remote-client-only
    config/import and JSONL write/read smoke test. System `python3` lacks
    `mmengine`, while the launch script correctly selects the minimal venv.
- Safety/runtime boundary: no model was loaded, no observation/control
  WebSocket was contacted, and no robot action was sent.

## 2026-08-06 23:59 CST — Align fixed-prefix RTC execution and remain length

- Root cause: the RTC model used a fixed `prefix_len=9`, but the runner sliced
  each returned chunk at the measured, fractional inference latency. The
  resulting replacement index and remain length changed from inference to
  inference and were not aligned with the prefix boundary used to condition
  the model.
- Implementation:
  - added a runner-level asynchronous offset hook and used monotonic timestamps
    for inference/action time alignment;
  - enabled `fixed_prefix_execution=True` for local RTC deployment: an early
    result waits until the fixed prefix boundary and an accepted 50-step chunk
    always starts at index 9, leaving exactly 41 actions;
  - added `fixed_prefix_late_tolerance_steps=1.0`; results that miss the fixed
    boundary by more than one policy step are held while the old trajectory
    continues, then the loop re-observes and re-infers instead of executing a
    stale offset or exiting;
  - action JSONL rows now include the actual execution offset and fixed-switch
    lateness. A fixed zero-step prefix is explicitly warned as unable to cover
    non-zero inference latency.
- Validation actually executed:
  - Python compilation and `git diff --check` passed;
  - 42 hardware-free runner/operator tests passed with five existing
    third-party warnings, including exact 50-to-41 remain length, early-result
    boundary waiting, and late-result hold tests for prefix lengths zero and
    two;
  - atomically synchronized the base/RTC runners and the new local-RTC config
    to `/home/guest/FluxVLA-Tron2`, verified all three SHA-256 values, and
    passed the PCM minimal-runtime remote-client import smoke test;
  - preserved the two replaced PCM runner files under
    `/home/guest/fluxvla-fixed-rtc-backup.MXu7B5` (the local-RTC config did not
    previously exist on the PCM).
- Safety/runtime boundary: no model was loaded, no inference/client process
  was started, no WebSocket was contacted, and no robot command was sent.

## 2026-08-07 00:32 CST — Replace fixed-boundary RTC with TRON2 ActionQueue scheduling

- Purpose: keep FluxVLA's train-time prefix-RTC inference path but remove all
  inference-latency/P95 step estimation and align deployment with the public
  TRON2 producer/consumer queue structure.
- Upstream comparison:
  - rechecked the public `tron2_openpi/examples/tron2/pi_client_rtc.py` and the
    pinned `tron2_env` `ActionQueue` implementation;
  - retained its `H - execution_horizon` producer trigger, persistent
    policy-rate consumer, atomic leftover snapshot and queue replacement by
    the consumer's actual index delta, six-frame underflow recovery, optional
    boundary/EMA postprocessing, and continuous `MotionController` lifetime;
  - intentionally removed the public client's `LatencyTracker`, measured
    delay conversion, rolling P95 buffer, and dynamically supplied model
    delay. The configured `rtc_config.prefix_len` is the only model prefix and
    never controls or estimates the executable queue index.
- Implementation:
  - replaced the prior wait-until-fixed-boundary/resample runner with a local
    FluxVLA producer and one persistent 30 Hz `ActionQueue` consumer;
  - inference continues concurrently with old-queue execution; merge uses the
    actual consumer-index advance observed during inference, with no wall-time
    to action-step conversion;
  - added the `Tron2EnvOperator.execute_waypoint()` path so queue replacement
    does not restart the 300 Hz ServoJ `MotionController`;
  - queue underflow holds the most recent accepted ServoJ target and applies
    the existing six-frame recovery blend; rejected/non-finite/incomplete
    chunks warn and leave the current queue/target unchanged;
  - preserved the `b`/`s`/`r`/`l` state machine, MoveJ reset, head lock,
    finite/range checks, 0.5-rad adjacent ServoJ target guard, and current
    FluxVLA checkpoint/config/prompt loading;
  - local RTC defaults are `H=50`, `execution_horizon=10`, fixed
    `prefix_len=9`, and action boundary/EMA postprocessing disabled.
- Validation actually executed:
  - YAPF, Python bytecode compilation, and `git diff --check` passed;
  - RTC runner/operator and CI-smoke tests passed: 44 tests with five existing
    third-party warnings, including a no-estimator runtime assertion;
  - the complete repository test selection reported 41 passed, 21 skipped and
    two DreamZero KV-cache failures. Those failures reproduce independently in
    unchanged `DreamZeroHead` code/tests (`_create_kv_cache` is absent and
    cache-mode results remain identical) and are unrelated to TRON2 RTC.
- Safety/runtime boundary: no checkpoint was loaded for inference, no Bridge
  or robot-control WebSocket was opened, and no robot command was sent. The
  robot-side checkout was not changed because this RTC mode is the local
  single-process GPU deployment.

## 2026-08-07 00:39 CST — Accept PI0.5 padded raw actions in RTC queue

- Root cause: the new RTC integrity check incorrectly required normalized raw
  and denormalized executable actions to have equal feature dimensions. PI0.5
  intentionally returns 32-dimensional padded raw actions for the next RTC
  prefix, while the deployment denormalizer emits the 16-dimensional TRON2
  command. Every valid initial chunk was therefore held before the consumer
  could start.
- Fix: keep horizon/finite-value validation for both arrays but allow their
  feature dimensions to differ; retain raw 32-D actions in the original queue
  and processed 16-D actions in the executable queue. Invalid-initial-chunk
  retries now also pause for the configured poll interval to avoid log spam.
- Validation actually executed: YAPF, Python compilation and
  `git diff --check` passed; RTC/operator regression tests passed, 45 tests
  with five existing third-party warnings, including a 32-D raw/16-D command
  regression test.
- Hardware boundary: diagnosis used the operator-provided log only. No model
  or robot connection was started by this fix, and the already running process
  must be stopped and restarted to load it.

## 2026-08-07 00:49 CST — Create exact-upstream TRON2 RTC execution branch

- Version-control boundary:
  - saved the complete previous `main` worktree, including untracked files, as
    `stash@{0}` with message `pre-tron2env-exact-rtc-20260807-0045`;
  - created and switched to `tron2env-rtc-execution`;
  - applied the stash without dropping it, so the new branch has the required
    WebSocket, keyboard/reset, checkpoint and recording foundation while the
    original state remains recoverable from the retained stash.
- Upstream baseline:
  - pinned `tron2_env` commit
    `5b7b145229416f3731f61657e6fa71c89c37bc9d` and `tron2_openpi` commit
    `fb1ca651bc0de96aef6a4d2d1445e98cb9a84ac5`;
  - ported the RTC producer's `H-s` trigger, `ceil(latency / policy_period)`
    measured delay, ten-sample recent P95, `LatencyTracker`, atomic
    `ActionQueue` snapshot/merge, optional postprocessor and warmup structure;
  - ported the consumer's 30 Hz loop, large-jump diagnostic, empty-queue hold,
    six-frame recovery blend and immediate shutdown-event semantics.
- Native execution path:
  - replaced the RTC waypoint path with the pinned `Tron2Env.step` action
    extraction, gripper `[0, 100]` clipping, current-head passthrough and
    no-argument `MotionController.command_joints()` call;
  - removed per-policy-tick measured-state polling and custom trajectory guard
    execution from this RTC path. The controller remains measured-state-seeded
    once at startup and performs 300 Hz linear interpolation/publishing;
  - preserved the existing FluxVLA model/checkpoint loader, task-ID keyboard
    state machine, action recorder and idle-only MoveJ reset integration.
- Checkpoint adaptation:
  - kept the selected fold-clothes checkpoint
    `step-018000-epoch-001-loss=0.0137.safetensors` and task ID 12;
  - synchronized runtime model settings with its checkpoint-local config:
    `n_action_steps=50`, raw/action dimensions 32/16,
    `rtc_training_config.max_delay=20`, uniform distribution and explicit
    delays `[0, 5, 10, 19]`; initial deployment delay is nine frames and local
    model execution uses BF16 mixed precision instead of the inherited FP32.
- Validation actually executed:
  - YAPF, Python bytecode compilation and `git diff --check` passed;
  - 47 hardware-free RTC runner/operator/CI tests passed with five existing
    third-party warnings;
  - regression coverage includes the upstream P95 calculation, raw 32-D versus
    executable 16-D queues, actual-index ActionQueue merge, shutdown-event
    consumer, six-frame recovery and native current-head/gripper behavior;
  - the complete test selection still reaches only the same two unrelated,
    independently reproduced DreamZero KV-cache failures documented above;
    all selected TRON2 tests pass.
- Safety/runtime boundary: no inference model was loaded, no Bridge/control
  WebSocket was contacted and no robot command was sent during this change.
  Native RTC execution intentionally omits the project-specific per-tick
  ServoJ guards; real-hardware smoothness and safety remain unvalidated.

## 2026-08-07 01:11 CST — Switch RTC branch to native Tron2Env and discrete checkpoint prefixes

- Purpose: remove the remaining locally reproduced TRON2 observation/control
  path from `tron2env-rtc-execution`, while retaining only the FluxVLA adapter
  needed to run the fold-clothes 18k train-time-RTC checkpoint.
- Upstream verification:
  - freshly cloned `limxdynamics/tron2_env` at
    `5b7b145229416f3731f61657e6fa71c89c37bc9d` and `tron2_openpi` at
    `fb1ca651bc0de96aef6a4d2d1445e98cb9a84ac5`;
  - `diff -qr` confirmed that the installed `tron2-env==0.1.0` package is
    identical to the pinned `tron2_env/src/tron2_env` source.
- Implementation:
  - added `Tron2NativeEnvOperator`, a thin adapter that directly constructs
    the public `Tron2Env` and delegates observation to `get_obs()`, each 16-D
    policy command to `step()`, and idle `r` reset to environment
    reconstruction so the upstream MoveJ bring-up runs again;
  - selected the public deployment topology: Bridge WebSocket images, robot
    WebSocket state, 30 Hz policy playback, and the upstream 300 Hz
    `MotionController`; head initialization remains unset for this robot;
  - retained the official RTC scheduling values `H=50`, `s=10`, initial
    `d=6`, `ceil(latency / 1/30s)`, ten-sample P95, actual-consumer-index
    `ActionQueue` replacement, six-frame stall recovery, and disabled action
    postprocessing;
  - adapted dynamic delay to the checkpoint's explicitly confirmed supported
    prefixes `{0, 5, 9, 19}` by rounding upward. Queue replacement still uses
    the actual consumer index. The model receives padded normalized raw
    previous actions with shape `[1, 50, 32]`; only denormalized 16-D actions
    reach `Tron2Env.step()`;
  - added `d`, quantized `prefix`, measured delay, and merge-used index to the
    producer diagnostics, and recorded supported prefixes in action-session
    metadata.
- Commands and validation actually executed:
  - YAPF, Python bytecode compilation, resolved-MMEngine-config inspection,
    and `git diff --check` passed;
  - project TRON2/RTC tests plus the freshly cloned upstream `tron2_env` tests:
    83 passed with five existing third-party warnings;
  - complete repository suite: 77 passed, 26 skipped, and the same two
    unrelated DreamZero KV-cache failures (`_create_kv_cache` absent and cache
    reuse output unchanged) already present in the branch baseline;
  - confirmed the selected checkpoint exists locally at 14,466,989,776 bytes.
- Runtime boundary: no checkpoint inference was launched, no Bridge or robot
  WebSocket was opened, and no control command was sent. Real-hardware
  smoothness remains unvalidated and requires an operator-supervised run.

## 2026-08-07 01:31 CST — Correct fold RTC prefixes and isolate action jitter

- Corrected the fold checkpoint deployment prefixes from the previously used
  `[0, 5, 9, 19]` to the checkpoint-local `[0, 5, 10, 19]`. Measured delays
  of six through ten frames now condition the model with prefix ten; actual
  queue replacement continues to use the consumer's observed index.
- Updated the RTC config regression fixtures and quantization expectations.
  `/home/limx/miniconda3/envs/fluxvla/bin/python -m pytest
  test/test_ci_smoke.py -q` passed all 35 tests with five existing third-party
  warnings; `git diff --check` also passed.
- Offline analysis of the operator-recorded 88-action RTC session found stable
  32.42 ms policy pacing but strong within-chunk oscillation after the first
  plain chunk. RTC-phase arm step P95 was 0.1233 rad versus 0.0315 rad in the
  initial plain segment, and the largest 0.1655 rad change occurred inside a
  generated chunk rather than at a queue replacement boundary.
- Read-only inspection of the training server confirmed that the fold dataset
  is 30 Hz. Thirty evenly sampled episodes (45,438 transitions) had a 0.0444
  rad arm-step P95, approximately 5% velocity-sign reversals, and strongly
  positive adjacent-velocity correlation; the deployed RTC actions had
  46%--70% reversals and negative adjacent-velocity correlation. This rules
  out an intended high-frequency training trajectory or a 30 Hz mismatch.
- The training server code confirms that this checkpoint sampled only the
  configured discrete delays, so prefix nine was an unseen conditioning value.
  A second code-level risk remains: training actions are 18-D and padded to
  32-D while the loss supervises only the first 16 dimensions, but deployment
  currently feeds all 32 raw predicted dimensions back as the next clean RTC
  prefix. The unsupervised tail may therefore be out of distribution. This is
  a diagnosis only; no masking behavior was changed in this entry.
- Hardware boundary: all investigation after the supplied recording was
  offline/read-only. No robot command was issued. The already running process
  predates the prefix correction and must be restarted to load it.

## 2026-08-07 01:38 CST — Restrict RTC prefix feedback to supervised actions

- Read-only inspection of the exact training-server source and dataset showed
  that each 18-D action is padded to 32-D before RTC conditioning, while the
  PI0.5 loss truncates both prediction and target to `ori_action_dim=16`.
  Thus dimensions 16--17 contain clean head data during training and 18--31
  are zero padding, but none of dimensions 16--31 receives action loss. The
  previous deployment fed all 32 model-predicted dimensions back as the next
  clean prefix, including that unsupervised tail.
- Added `rtc_config.prefix_action_dim=16`. `ActionQueue` still retains the
  complete raw 32-D chunk for scheduling and diagnostics, but the next model
  request receives only the supervised first 16 dimensions. PI0.5's existing
  inference path then zero-pads that tensor back to its 32-D internal width.
  Queue timing, measured-delay replacement, prefix length, postprocessing and
  robot command layout are unchanged.
- Added configuration and shape regression coverage, startup diagnostics and
  action-session metadata for the feedback width. Updated the RTC deployment
  documentation and corrected the checkpoint prefix set to
  `[0, 5, 10, 19]`.
- Validation actually executed: YAPF, Python bytecode compilation and
  `git diff --check` passed; `python -m pytest test/test_ci_smoke.py -q`
  passed all 35 tests, and the combined RTC/operator selection passed all 57
  tests, with five existing third-party warnings. No model was loaded and no
  robot/Bridge connection or command was made.

## 2026-08-07 01:48 CST — Match RTC head conditioning and fix prefix at 19

- Further inspection showed that training did not condition the complete
  16--31 feature tail with zeros. Dataset actions are 18-D: dimensions 16--17
  are head targets and only dimensions 18--31 are padding. A deterministic
  sample of 45,468 training frames found head 0 primarily at 0 or 1.0467 rad
  (min-max normalized near -1 or +1), while head 1 primarily used -0.014,
  0, or 0.0046 rad (normalized near -1, 0.505, or +1). Therefore zeroing both
  head dimensions at deployment did not match train-time RTC conditioning.
- The RTC runner now retains the latest measured two-joint head observation.
  Prefix construction uses the previous raw model chunk's supervised first
  16 dimensions, appends that measured locked head normalized with the exact
  checkpoint action min/max statistics, and relies on PI0.5 to zero-pad the
  remaining 14 features to its internal 32-D width. It fails explicitly if
  the head, normalization type, or statistics are unavailable, and logs the
  first raw/normalized head pair for operator verification.
- The selected deployment also sets `include_head_in_state=True`. Its model
  input now preserves the training dataset's full 18-D measured proprio
  `[left7, left_gripper, right7, right_gripper, head2]`, while action parsing
  and robot commands remain the established 16-D head-free layout. This fixes
  a second zero-padded-head mismatch without authorizing head control.
- Added `rtc_config.prefix_len=19` as a fixed trained model condition and
  validates it against the checkpoint's `[0, 5, 10, 19]` set. This does not
  alter ActionQueue timing: queue replacement still crops by actual consumer
  progress during inference.
- Updated configuration, action-record metadata and deployment documentation.
  YAPF, bytecode compilation and `git diff --check` passed; the combined
  RTC/config/operator selection passed all 59 tests with five existing
  third-party warnings. No model, Bridge, robot state or command connection
  was used during validation.

## 2026-08-07 01:51 CST — Stage fold RTC 32k checkpoint

- Pulled `step-032000-epoch-001-loss=0.0112.safetensors` from the authorized
  LIMX training server into the existing fold RTC work directory without
  replacing the 18k checkpoint. Transfer took 18 minutes 51 seconds at an
  average 12.20 MB/s.
- Verified the 14,466,989,776-byte file against the remote SHA-256
  `36380569f7ab6930bbf066f9c3f95ac55460bbad00111acb86e73a3d7953b48b`.
  Safetensors header inspection found 812 readable tensors.
- Synchronized and hash-checked the remote `config.json`, `config.yaml`,
  `dataset_statistics.json`, `adapter_config.json`, `README.md`, and
  `llm_backbone_config.json`. The remote work directory did not contain
  `deployment_metadata.json` or `tokenizer/`, so the already validated local
  copies were preserved. No model or robot process was started.

## 2026-08-07 07:17 CST — Reconstruct task1/task2 RTC 22k deployment files

- Completed the partially downloaded
  `pi05_task1_task2_rtc_lora_rank256_8gpu_bs8_40k_20260806_173542` work
  directory after its original SeetaCloud endpoint became unavailable. Per
  the training-run provenance supplied by the user, model structure,
  tokenizer and task metadata were copied from the matching task1/task2
  `...20260806_005318` run, while both model RTC configuration blocks were
  set to the fold-RTC training settings: uniform delays, `max_delay=20`, and
  `delay_values=[0, 5, 10, 19]`.
- Preserved the 22k checkpoint and its existing statistics. Its statistics
  SHA-256 matches the task1/task2 source run exactly. Deployment metadata
  advertises task IDs 1--11 with `action_layout=tron2_16`; it does not
  incorrectly advertise the fold-only task 12.
- Offline validation loaded the local tokenizer as `GemmaTokenizerFast`
  (257,153 tokens), verified JSON and YAML RTC blocks against the fold 32k
  files, and verified every non-RTC config field against the task1/task2
  source. Safetensors header inspection found 812 tensors in the
  14,466,989,776-byte checkpoint. No model inference, Bridge connection or
  robot command was started.

## 2026-08-07 07:32 CST — Stage fold RTC 40k and audit SeetaCloud latest checkpoint

- Pulled the fold RTC `step-040000-epoch-002-loss=0.0108.safetensors`
  checkpoint from the authorized LIMX server. The 14,466,989,776-byte local
  file matches the remote SHA-256
  `047a2dab06009dd4b56a6e7123be5e3cf28a68637ad3758653406341d6843b9e`;
  safetensors header, offsets and all 812 tensors are readable. Transfer took
  18 minutes 27 seconds at an average 12.46 MB/s. The existing fold 18k and
  32k checkpoints were preserved.
- Synchronized and hash-checked the six configuration files present beside
  that fold run. The remote run had no deployment metadata or tokenizer, so
  the validated local copies were preserved.
- Attempted the separately requested latest-checkpoint audit on the authorized
  SeetaCloud endpoint. SSH login succeeds, but `/dev/shm/fluxvla-runs` and the
  requested task1/task2 run no longer exist; `/dev/shm` is nearly empty and
  has a fresh 07:26 mtime, consistent with tmpfs being cleared after an
  instance restart. No newer checkpoint candidate could be listed or
  transferred. The existing local 22k checkpoint and reconstructed config
  were not modified. No training, inference or robot process was started.

## 2026-08-07 07:47 CST — Support non-RTC chunks in the native operator

- Diagnosed a deployment-only interface mismatch after selecting
  `Tron2InferenceRunner` to disable inference-time RTC. The ordinary runner
  submits complete chunks through `execute_trajectory`, while the new native
  `tron2_env` adapter only exposed the RTC runner's per-frame
  `execute_waypoint` method. Model loading and task metadata were valid; the
  first chunk failed before any action was submitted.
- Added synchronous native trajectory execution. The adapter validates every
  arm, gripper and optional-head array before issuing the first frame, then
  feeds each waypoint through the same upstream `Tron2Env.step()` path at the
  configured policy period. A 16-D chunk continues to omit head commands, and
  asynchronous trajectory mode is rejected explicitly because this local
  profile configures synchronous execution.
- Added hardware-free coverage for 30 Hz pacing, per-frame recording metadata,
  the 16-D head-free action layout, pre-execution shape rejection and the
  asynchronous-mode guard. Python compilation, `git diff --check`, and the
  combined operator/config regression selection passed all 61 tests with five
  existing third-party warnings. No model, Bridge or robot connection was
  used during validation.

## 2026-08-07 13:51 CST — Open grippers before idle reset MoveJ

- Confirmed that upstream `tron2_env` performs its construction-time reset as
  MoveJ first and gripper-open second. Changed only the idle `r` reset path:
  it now opens both grippers through the still-connected old native
  environment, waits the configured 0.5 seconds, then closes ServoJ and
  reconstructs `Tron2Env` for the original MoveJ bring-up. Startup
  initialization and running-task key restrictions are unchanged.
- A failed gripper-open request aborts reset before the old environment is
  closed or any new MoveJ environment is created. Added the native reset hook
  to the ordinary `Tron2InferenceRunner` as well, so RTC and non-RTC keyboard
  modes share the same order and neither relies on the legacy
  `move_to_targets` interface.
- Documented `reset_gripper_open_wait_s=0.5` and the new safety order. Python
  compilation, `git diff --check`, and the combined operator/config selection
  passed all 63 tests with five existing third-party warnings. Validation used
  fake environments only; no model, Bridge or robot connection was made.
