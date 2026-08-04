#!/bin/bash
# Remote inference client launcher with optional SSH tunnel.
#
# Usage:
#   bash scripts/remote_inference_client.sh <CONFIG> [OPTIONS]
#
# Basic (tunnel already set up or on same LAN):
#   bash scripts/remote_inference_client.sh \
#       configs/pi05/pi05_paligemma_ur3_remote_inference.py
#
# With auto SSH tunnel:
#   bash scripts/remote_inference_client.sh \
#       configs/pi05/pi05_paligemma_ur3_remote_inference.py \
#       --ssh-host user@server.example.com \
#       --ssh-port 22 \
#       --ssh-key ~/.ssh/id_rsa \
#       --local-port 5555 \
#       --remote-port 3333

set -e

CONFIG=""
SSH_HOST=""
SSH_PORT="22"
SSH_KEY=""
LOCAL_PORT="5555"
REMOTE_PORT="3333"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ssh-host)     SSH_HOST="$2";    shift 2 ;;
        --ssh-port)     SSH_PORT="$2";    shift 2 ;;
        --ssh-key)      SSH_KEY="$2";     shift 2 ;;
        --local-port)   LOCAL_PORT="$2";  shift 2 ;;
        --remote-port)  REMOTE_PORT="$2"; shift 2 ;;
        --cfg-options)
            shift
            EXTRA_ARGS="--cfg-options"
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                EXTRA_ARGS="$EXTRA_ARGS $1"
                shift
            done
            ;;
        *)
            if [ -z "$CONFIG" ]; then
                CONFIG="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$CONFIG" ]; then
    cat <<'USAGE'
Usage: bash scripts/remote_inference_client.sh <CONFIG> [OPTIONS]

Options:
  --ssh-host USER@HOST    SSH destination (enables auto tunnel)
  --ssh-port PORT         SSH port (default: 22)
  --ssh-key PATH          SSH private key file
  --local-port PORT       Local tunnel port (default: 5555)
  --remote-port PORT      Remote ZMQ server port (default: 3333)
  --cfg-options K=V ...   Override config values

Examples:
  # LAN / tunnel already running
  bash scripts/remote_inference_client.sh \
      configs/pi05/pi05_paligemma_ur3_remote_inference.py

  # Auto SSH tunnel
  bash scripts/remote_inference_client.sh \
      configs/pi05/pi05_paligemma_ur3_remote_inference.py \
      --ssh-host user@server.example.com \
      --ssh-port 57705 \
      --ssh-key ~/.ssh/my_key \
      --local-port 5555 \
      --remote-port 3333
USAGE
    exit 1
fi

export FLUXVLA_REMOTE_CLIENT_ONLY=1
export PYTHONPATH="$(pwd):${PYTHONPATH}"

PYTHON_BIN="${FLUXVLA_PYTHON:-}"
if [ -z "$PYTHON_BIN" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    elif [ -x "${HOME}/.venvs/fluxvla-tron2/bin/python" ]; then
        PYTHON_BIN="${HOME}/.venvs/fluxvla-tron2/bin/python"
    else
        echo "[client] ERROR: no Python interpreter found."
        echo "[client] Activate the lightweight client environment or set FLUXVLA_PYTHON."
        exit 1
    fi
fi

if ! "$PYTHON_BIN" -c \
    "import tron2_env, websockets; from fluxvla.engines.operators import Tron2EnvOperator"; then
    echo "[client] ERROR: lightweight client imports failed with $PYTHON_BIN."
    echo "[client] Install the pinned tron2-env bridge runtime (websockets>=12)."
    exit 1
fi

TUNNEL_CONTROL_DIR=""
TUNNEL_CONTROL_SOCKET=""
cleanup() {
    if [ -n "$TUNNEL_CONTROL_SOCKET" ] && [ -S "$TUNNEL_CONTROL_SOCKET" ]; then
        echo "[client] Closing SSH tunnel..."
        ssh -S "$TUNNEL_CONTROL_SOCKET" -O exit "$SSH_HOST" >/dev/null 2>&1 || true
    fi
    if [ -n "$TUNNEL_CONTROL_DIR" ] && [ -d "$TUNNEL_CONTROL_DIR" ]; then
        rmdir "$TUNNEL_CONTROL_DIR" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [ -n "$SSH_HOST" ]; then
    if command -v ss >/dev/null 2>&1; then
        PORT_LISTENERS=$(ss -H -ltnp "( sport = :${LOCAL_PORT} )" 2>/dev/null || true)
        if [ -n "$PORT_LISTENERS" ]; then
            echo "[client] ERROR: local port ${LOCAL_PORT} is already in use:"
            echo "$PORT_LISTENERS"
            echo "[client] Stop the existing client/tunnel or choose another --local-port."
            exit 1
        fi
    fi

    TUNNEL_CONTROL_DIR="$(mktemp -d /tmp/fluxvla-ssh.XXXXXX)"
    TUNNEL_CONTROL_SOCKET="${TUNNEL_CONTROL_DIR}/control"

    SSH_CMD=(
        ssh
        -p "$SSH_PORT"
        -o ExitOnForwardFailure=yes
        -o ServerAliveInterval=30
        -o ServerAliveCountMax=3
    )
    if [ -n "$SSH_KEY" ]; then
        SSH_CMD+=(-i "$SSH_KEY")
    fi
    SSH_CMD+=(
        -M
        -S "$TUNNEL_CONTROL_SOCKET"
        -fN
        -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}"
        "$SSH_HOST"
    )

    echo "[client] Starting SSH tunnel..."
    echo "  ${SSH_CMD[*]}"
    echo "[client] Complete any host-key or password prompts below."
    "${SSH_CMD[@]}"

    if ! TUNNEL_STATUS=$(ssh -S "$TUNNEL_CONTROL_SOCKET" -O check "$SSH_HOST" 2>&1); then
        echo "[client] ERROR: SSH tunnel failed to start."
        echo "  $TUNNEL_STATUS"
        exit 1
    fi
    echo "[client] SSH tunnel running ($TUNNEL_STATUS)"
    echo "  local :${LOCAL_PORT} -> remote :${REMOTE_PORT}"
fi

echo "[client] Starting inference with config: $CONFIG"
"$PYTHON_BIN" scripts/inference.py --config "$CONFIG" $EXTRA_ARGS
