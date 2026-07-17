#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/artifacts/conda/runtime/bin/python}"

test -x "$PYTHON"
cd "$REPO_ROOT"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8127/v1}"
export VLLM_MODEL="${VLLM_MODEL:-gemma3-4b-temp0}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-127.0.0.1}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

exec "$PYTHON" -m gradio_project.interface.interface_v4_gradio
