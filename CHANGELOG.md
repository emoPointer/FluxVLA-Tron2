# Changelog

All notable changes to FluxVLA should be documented in this file.

The format follows a simple release-note style. Dates use `YYYY-MM-DD`.

## Unreleased

### Added

- Added Tron2 PI0.5 LoRA fine-tuning configuration.
- Added Tron2 remote-inference deployment guide.
- Added robot-side lightweight remote client support for ZMQ-based remote
  inference.
- Added dry-run support for the Tron2 inference flow so observations can be
  collected, sent to the GPU server, inferred, and returned without executing
  robot actions.
- Added 16-dimensional Tron2 action layout support:
  `left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)`.
- Added repository governance files for open-source release review:
  `NOTICE`, `SECURITY.md`, GitHub issue templates, a pull request template, and
  a preliminary third-party dependency license review.

### Changed

- Updated Tron2 camera topics to match the common Power Computing Module ROS
  topic layout.
- Avoided `cv_bridge` for Tron2 image conversion to reduce Conda/ROS dynamic
  library conflicts.
- Sanitized the default Tron2 WebSocket account ID in the public config by
  leaving `ws_accid=None`.

### Known Limitations

- This repository does not include private datasets, private checkpoints, or
  training outputs.
- Real-robot execution requires user-side safety validation, stable ROS topics,
  a stable Tron2 WebSocket control service, and a physical emergency stop.
- The third-party dependency license table is a preliminary engineering review
  and still requires OSPO/legal confirmation before a formal public release.
- Git history may require cleanup before publishing a public release branch.
