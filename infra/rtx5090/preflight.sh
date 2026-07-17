#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-/home/liaan/Documentos/Luan/temp_vllm/conda_envs/gradio/bin/python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8127}"

echo "host=$(hostname)"
echo "repo=$REPO_ROOT"
echo "commit=$(git rev-parse HEAD)"
git status --short --branch
test -x "$PYTHON_BIN"
"$PYTHON_BIN" --version
curl -fsS "$BASE_URL/v1/models"
echo
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
tmux ls 2>&1 || true
test -L "$REPO_ROOT/artifacts/phase3_memory/RTX5090_liaan"
echo "preflight=PASS"
