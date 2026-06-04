# Security Policy

FluxVLA can be used with real robot systems. Security reports may involve robot
control, network services, credentials, model checkpoints, datasets, or
deployment infrastructure. Please do not disclose security-sensitive details in
public GitHub issues, pull requests, or discussions.

## Reporting a Vulnerability

Report security issues privately through one of these channels:

- GitHub private vulnerability reporting, if enabled for this repository.
- Email the maintainers listed in `README.md` with the subject prefix
  `[FluxVLA Security]`.

Please include:

- affected component, file, or deployment mode;
- FluxVLA commit or release version;
- environment details needed to reproduce the issue;
- impact assessment, especially if real robot control is affected;
- minimal reproduction steps or logs with secrets and private data removed.

Do not include private keys, tokens, robot account IDs, customer data, real site
names, or uncensored logs in the report.

## Response Expectations

The maintainers aim to acknowledge security reports within 72 hours. The fix
timeline depends on severity, reproducibility, and whether the issue requires
internal escalation or third-party coordination.

## Real Robot Safety

If a vulnerability or malfunction may affect a real robot:

- stop the experiment;
- keep a physical emergency stop available;
- disconnect external-control paths if needed;
- do not publish exploit details or unsafe control instructions publicly.

`Ctrl+C` or process termination should not be treated as a robot emergency stop.
Use the robot platform's official safety mechanisms.

## Supported Versions

Security fixes are handled for the latest public release and the active main
development branch unless otherwise stated in the release notes.
