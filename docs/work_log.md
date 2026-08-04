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
