# Contributing to FluxVLA-Tron2

Thanks for contributing to FluxVLA-Tron2. This repository focuses on Tron2
training, remote inference, and real-robot deployment support built on top of
the upstream FluxVLA project.

[English](#contributing-to-fluxvla-tron2) | [简体中文](#贡献指南)

## Contribution Scope

Use this repository for changes related to:

- Tron2 dataset preparation, training configs, and LoRA fine-tuning.
- Tron2 ROS observation collection and WebSocket execution.
- Remote inference serving and robot-side client workflows.
- Documentation, safety notes, CI, tests, and open-source release metadata.

For generic FluxVLA model architecture changes that are not specific to Tron2,
consider opening an issue or pull request against the upstream project:
https://github.com/FluxVLA/FluxVLA.

## Issues

Open an issue in this repository:
https://github.com/clearlab-sustech/FluxVLA-Tron2/issues.

Use the bug report template for reproducible failures and the feature request
template for new functionality. Do not include private datasets, checkpoints,
robot account IDs, private server addresses, customer names, real site names,
tokens, SSH keys, or uncensored logs in public issues.

For security-sensitive reports, follow `SECURITY.md` instead of opening a
public issue.

## Pull Requests

1. Fork the repository and create a branch from `main`.
2. Keep each pull request focused on one issue or one coherent change.
3. Describe what changed, why it changed, and how it was tested.
4. Run the relevant local checks before requesting review.
5. Fill in the pull request checklist, especially the public-release and
   real-robot impact sections.

Recommended branch names:

| Prefix      | Purpose                                  |
| ----------- | ---------------------------------------- |
| `feat/`     | New feature or model/config support      |
| `fix/`      | Bug fix                                  |
| `docs/`     | Documentation only                       |
| `test/`     | Tests or CI                              |
| `refactor/` | Internal cleanup without behavior change |

## Development Checks

Install pre-commit:

```bash
pip install pre-commit
pre-commit install
```

Run checks on changed files:

```bash
pre-commit run --files <changed-files>
```

Run the lightweight CI smoke test:

```bash
python -m pytest test/test_ci_smoke.py
```

For robot-side changes, document whether the change affects dry-run only or
real robot execution. Real-robot execution paths must preserve a dry-run mode
and should include clear safety assumptions.

## License and Third-Party Code

By contributing, you agree that your contribution is provided under the
repository license in `LICENSE`.

New third-party code, configuration, model-support files, images, datasets, or
adapted snippets must include source and license information. Update `NOTICE`
and `docs/release_license_review.md` when adding or modifying third-party
materials.

## 贡献指南

感谢你参与 FluxVLA-Tron2。本仓库重点维护基于上游 FluxVLA 的 Tron2
训练、remote inference 和真实机器人部署能力。

本仓库适合提交：

- Tron2 数据集准备、训练配置和 LoRA 微调相关改动。
- Tron2 ROS 观测采集和 WebSocket 执行动作相关改动。
- remote inference 服务端和机器人侧客户端流程。
- 文档、安全说明、CI、测试和开源发布元数据。

如果改动是通用 FluxVLA 模型架构能力，且不专属于 Tron2，优先考虑向上游
FluxVLA 项目提交：https://github.com/FluxVLA/FluxVLA。

## Issue

请在本仓库提交 Issue：
https://github.com/clearlab-sustech/FluxVLA-Tron2/issues。

可复现问题使用 bug report 模板，新功能建议使用 feature request 模板。公开
Issue 中不要包含私有数据集、权重、机器人账号 ID、私有服务器地址、客户名称、
真实场地名称、token、SSH key 或未脱敏日志。

安全问题请按 `SECURITY.md` 处理，不要在公开 Issue 中披露漏洞细节。

## Pull Request

1. Fork 本仓库，并从 `main` 创建分支。
2. 每个 PR 聚焦一个问题或一组清晰相关的改动。
3. 在 PR 中说明改了什么、为什么改、如何测试。
4. 请求 review 前运行相关本地检查。
5. 完成 PR checklist，尤其是开源发布检查和真实机器人影响说明。

建议分支命名：

| 前缀        | 用途                   |
| ----------- | ---------------------- |
| `feat/`     | 新功能、模型或配置支持 |
| `fix/`      | Bug 修复               |
| `docs/`     | 文档改动               |
| `test/`     | 测试或 CI              |
| `refactor/` | 不改变行为的内部整理   |

## 开发检查

安装 pre-commit：

```bash
pip install pre-commit
pre-commit install
```

检查改动文件：

```bash
pre-commit run --files <changed-files>
```

运行轻量 CI smoke test：

```bash
python -m pytest test/test_ci_smoke.py
```

如果改动影响机器人侧逻辑，请说明它只影响 dry-run，还是影响真实机器人执行。
真实机器人执行路径必须保留 dry-run 模式，并说明安全假设。

## License 和第三方代码

提交贡献即表示你同意该贡献按本仓库 `LICENSE` 授权。

新增第三方代码、配置、模型支持文件、图片、数据集或改写片段时，必须记录来源和
License。涉及第三方材料的改动需要同步更新 `NOTICE` 和
`docs/release_license_review.md`。
