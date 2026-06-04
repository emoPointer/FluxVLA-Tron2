# Third-Party Dependency License Review

This document is a preliminary engineering review for open-source release
preparation. It is not legal approval. Before a formal public release, OSPO or
legal reviewers should confirm every item marked `Needs review`.

Source used for this draft:

- `requirements.txt`
- installed package metadata from the local `fluxvla` conda environment
- local third-party attribution files under `fluxvla/models/third_party_models`

Status values:

- `Pass`: package metadata indicates a permissive or otherwise expected license.
- `Needs review`: license metadata is missing, unclear, copyleft-like, or the
  dependency is pulled from a Git repository and must be checked from source.
- `Not packaged`: not included in this source repository, but referenced by
  docs or runtime setup.

## Direct Runtime and Training Dependencies

| Dependency            | Version / Source                                   | License from local metadata | Status       | Notes                                                                                     |
| --------------------- | -------------------------------------------------- | --------------------------- | ------------ | ----------------------------------------------------------------------------------------- |
| `torch`               | `2.6.0`                                            | BSD-3-Clause                | Pass         | PyTorch package.                                                                          |
| `torchvision`         | `0.21.0`                                           | BSD                         | Pass         | PyTorch vision package.                                                                   |
| `torchaudio`          | `2.6.0`                                            | Empty metadata              | Needs review | PyTorch audio package; confirm upstream license for release record.                       |
| `accelerate`          | `0.33.0`                                           | Apache                      | Pass         | Hugging Face package.                                                                     |
| `bddl`                | `1.0.1`                                            | Empty metadata              | Needs review | Confirm upstream license and whether bundled task assets are redistributed.               |
| `boto3`               | `1.38.32`                                          | Apache License 2.0          | Pass         | AWS SDK.                                                                                  |
| `botocore`            | `1.38.32`                                          | Apache License 2.0          | Pass         | AWS SDK core.                                                                             |
| `cloudpickle`         | installed `3.1.2`                                  | BSD-3-Clause                | Pass         | Requirement is unpinned.                                                                  |
| `datasets`            | `4.0.0`                                            | Apache 2.0                  | Pass         | Hugging Face package.                                                                     |
| `diffusers`           | `0.30.2`                                           | Apache 2.0 License          | Pass         | Hugging Face package.                                                                     |
| `dlimp`               | `git+https://github.com/kvablack/dlimp`            | Empty metadata              | Needs review | Editable Git dependency; confirm upstream license and commit/ref before release.          |
| `draccus`             | `0.10.0`                                           | MIT License                 | Pass         | Local metadata includes MIT text.                                                         |
| `easydict`            | `1.9`                                              | `LPGL, see LICENSE file.`   | Needs review | Metadata appears to mean LGPL; confirm exact license and policy compatibility.            |
| `einops`              | `0.4.1`                                            | UNKNOWN                     | Needs review | Confirm upstream license because package metadata is unclear.                             |
| `future`              | `0.18.2`                                           | MIT                         | Pass         | Python compatibility package.                                                             |
| `gcsfs`               | installed `2025.3.0`                               | BSD                         | Pass         | Requirement is unpinned.                                                                  |
| `gym`                 | `0.25.2`                                           | MIT                         | Pass         | Legacy Gym package.                                                                       |
| `hydra-core`          | `1.2.0`                                            | MIT                         | Pass         | Configuration framework.                                                                  |
| `imageio`             | installed `2.37.3`                                 | BSD-2-Clause                | Pass         | Requirement is unpinned.                                                                  |
| `imageio-ffmpeg`      | `0.6.0`                                            | BSD-2-Clause                | Pass         | FFmpeg wrapper; confirm binary redistribution scope if packaged.                          |
| `jsonlines`           | `4.0.0`                                            | BSD                         | Pass         | JSON Lines helper.                                                                        |
| `libero`              | `git+https://github.com/yinchimaoliang/LIBERO.git` | Empty metadata              | Needs review | Editable Git dependency; confirm license, assets, and dataset redistribution permissions. |
| `matplotlib`          | installed `3.10.9`                                 | Matplotlib license          | Pass         | Requirement is unpinned; license text includes bundled font notices.                      |
| `mmengine`            | `0.10.7`                                           | Empty metadata              | Needs review | Confirm upstream license because package metadata is empty locally.                       |
| `mujoco`              | `3.3.2`                                            | Apache License 2.0          | Pass         | Google DeepMind MuJoCo package.                                                           |
| `numpy`               | `1.26.4`                                           | BSD-style full text         | Pass         | Local metadata includes full license and bundled library notices.                         |
| `opencv-python`       | installed `4.11.0.86`                              | Apache 2.0                  | Pass         | Requirement is unpinned.                                                                  |
| `peft`                | `0.11.1`                                           | Apache                      | Pass         | Hugging Face package.                                                                     |
| `robomimic`           | `0.2.0`                                            | Empty metadata              | Needs review | Confirm upstream license and benchmark asset usage.                                       |
| `robosuite`           | `1.4.1`                                            | UNKNOWN                     | Needs review | Confirm upstream license and asset redistribution terms.                                  |
| `sentencepiece`       | `0.1.99`                                           | Apache                      | Pass         | Google SentencePiece wrapper.                                                             |
| `sentry-sdk`          | `2.27.0`                                           | MIT                         | Pass         | Pulled by `wandb`.                                                                        |
| `tensorflow`          | `2.15.0`                                           | Apache 2.0                  | Pass         | TensorFlow package.                                                                       |
| `tensorflow-graphics` | installed `2021.12.3`                              | Apache 2.0                  | Pass         | Requirement is unpinned.                                                                  |
| `thop`                | `0.1.1-2209072238`                                 | MIT                         | Pass         | FLOPs counter.                                                                            |
| `timm`                | `0.9.10`                                           | Empty metadata              | Needs review | Confirm upstream license because package metadata is empty locally.                       |
| `tqdm-loggable`       | `0.2`                                              | MIT                         | Pass         | Progress logging helper.                                                                  |
| `transformers`        | `4.53.2`                                           | Apache 2.0 License          | Pass         | Hugging Face package.                                                                     |
| `types-boto3-s3`      | `1.38.26`                                          | MIT License                 | Pass         | Type stubs.                                                                               |
| `wandb`               | `0.21.0`                                           | MIT License                 | Pass         | Experiment tracking client.                                                               |

## Optional Remote Inference Dependencies

| Dependency         | Version / Source                       | License from local metadata | Status       | Notes                                                                                      |
| ------------------ | -------------------------------------- | --------------------------- | ------------ | ------------------------------------------------------------------------------------------ |
| `pyzmq`            | installed `27.1.0`                     | BSD 3-Clause License        | Pass         | Used by ZMQ remote inference.                                                              |
| `msgpack`          | installed `1.1.2`                      | Apache-2.0                  | Pass         | Used by msgpack serializer.                                                                |
| `protobuf`         | installed `4.25.9`                     | 3-Clause BSD License        | Pass         | Optional serializer dependency.                                                            |
| `websocket-client` | optional, not installed in checked env | Not checked locally         | Needs review | Required for Tron2 WebSocket control; verify installed version and license before release. |

## Bundled or Adapted Third-Party Files

| Component                        | Local path                                          | Upstream source                            | License status                    | Release action                                                                                                       |
| -------------------------------- | --------------------------------------------------- | ------------------------------------------ | --------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| DreamZero modules                | `fluxvla/models/third_party_models/dreamzero`       | `https://github.com/dreamzero0/dreamzero`  | Local attribution says Apache-2.0 | Preserve attribution and confirm copied files match upstream license.                                                |
| Alibaba Wan2.1 modules           | `fluxvla/models/third_party_models/dreamzero`       | `https://github.com/Wan-Video/Wan2.1`      | Local attribution says Apache-2.0 | Preserve attribution and confirm copied files match upstream license.                                                |
| Hugging Face Diffusers scheduler | `fluxvla/models/third_party_models/dreamzero`       | `https://github.com/huggingface/diffusers` | Local attribution says Apache-2.0 | Preserve attribution and confirm copied file origin/ref.                                                             |
| NVIDIA Isaac-GR00T Eagle2 files  | `fluxvla/models/third_party_models/eagle2_hg_model` | `https://github.com/NVIDIA/Isaac-GR00T`    | Needs review                      | Local attribution lacks an explicit license field; confirm upstream license and redistribution terms before release. |
| X-VLA / Florence-2 components    | `fluxvla/models/third_party_models/xvla_models`     | `https://github.com/2toINF/X-VLA`          | Local attribution says Apache-2.0 | Preserve upstream Microsoft/Hugging Face notices and local X-VLA origin comments.                                    |

## Data, Checkpoints, and Generated Outputs

| Asset type        | Repository state                             | Status       | Notes                                                                  |
| ----------------- | -------------------------------------------- | ------------ | ---------------------------------------------------------------------- |
| Private datasets  | Ignored by `.gitignore` under `datasets/`    | Not packaged | Do not commit private robot data.                                      |
| Model checkpoints | Ignored by `.gitignore` under `checkpoints/` | Not packaged | Public checkpoint links remain subject to publisher terms.             |
| Training outputs  | Ignored by `.gitignore` under `work_dirs/`   | Not packaged | Do not commit logs containing internal paths, hostnames, or user data. |

## Open Items Before Public Release

- Confirm or replace dependencies with missing or unclear metadata:
  `bddl`, `dlimp`, `libero`, `mmengine`, `robomimic`, `robosuite`, `timm`,
  `torchaudio`, `websocket-client`, and `einops`.
- Confirm whether `easydict` is LGPL and whether that is acceptable under the
  release policy.
- Confirm the NVIDIA Isaac-GR00T Eagle2 copied assets/config files are
  redistributable under the intended public repository license.
- Pin Git dependencies to reviewed commits or replace them with released
  packages if possible.
- Re-run the sensitive information scan before publishing the final release
  branch.
