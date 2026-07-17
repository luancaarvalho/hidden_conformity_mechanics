#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/artifacts/conda/runtime/bin/python}"

cd "$REPO_ROOT"
"$PYTHON" -m py_compile $(find gradio_project -name '*.py' -type f -print)
bash -n gradio_project/launchers/*.sh
"$PYTHON" -m unittest discover -s gradio_project/tests -p 'test_*.py' -v
