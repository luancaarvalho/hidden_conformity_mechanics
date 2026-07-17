#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
CONDA_BIN="${CONDA_BIN:-/home/liaan/miniconda3/bin/conda}"
UV_BIN="${UV_BIN:-/home/liaan/.local/bin/uv}"
PREFIX="${HC_CONDA_PREFIX:-$REPO_ROOT/artifacts/conda/runtime}"

test -x "$CONDA_BIN"
test -x "$UV_BIN"

if [ ! -x "$PREFIX/bin/python" ]; then
  "$CONDA_BIN" create -y -p "$PREFIX" python=3.12
fi

"$UV_BIN" pip install --python "$PREFIX/bin/python" \
  -r "$REPO_ROOT/infra/rtx5090/requirements-runtime.txt"
"$PREFIX/bin/python" -c 'import httpx, matplotlib, numba, numpy, openai, pandas, psutil, yaml; print("runtime_imports=PASS")'
