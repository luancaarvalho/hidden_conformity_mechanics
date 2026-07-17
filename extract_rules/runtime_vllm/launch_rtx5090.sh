#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/artifacts/conda/runtime/bin/python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8127}"
MODEL_ALIAS="${MODEL_ALIAS:-gemma3-4b-temp0}"
ARTIFACT_ROOT="${HC_ARTIFACT_ROOT:-$REPO_ROOT/artifacts}"
RUN_ID="phase1_gemma4b_canonical_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$ARTIFACT_ROOT/phase1_rule_extraction/RTX5090_liaan/$RUN_ID"
CANONICAL_MANIFEST="$SCRIPT_DIR/manifests/gemma4b_canonical_54_cells.csv"
RUN_MANIFEST="$RUN_DIR/inputs/experiments.csv"

if ! git -C "$REPO_ROOT" diff --quiet || ! git -C "$REPO_ROOT" diff --cached --quiet; then
  echo "Refusing scientific run from a dirty tracked worktree." >&2
  exit 2
fi

if tmux ls 2>/dev/null | grep -q 'gemma4b_standard_cot_r60'; then
  echo "The Phase 3 R60 batch is still active; queue Phase 1 instead of competing for the GPU." >&2
  exit 3
fi

test -x "$PYTHON_BIN"
curl -fsS "$BASE_URL/v1/models" | grep -q "$MODEL_ALIAS"
mkdir -p "$RUN_DIR/inputs" "$RUN_DIR/output" "$RUN_DIR/audit"

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_manifest.py" \
  --input "$CANONICAL_MANIFEST" \
  --output "$RUN_MANIFEST" \
  --model-alias "$MODEL_ALIAS"

test "$(($(wc -l < "$RUN_MANIFEST") - 1))" -eq 54
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/audit/git_commit.txt"
git -C "$REPO_ROOT" status --short --branch > "$RUN_DIR/audit/git_status.txt"
sha256sum "$SCRIPT_DIR/execucao_simultanea_vllm.py" "$SCRIPT_DIR/prompt_strategies.py" \
  "$SCRIPT_DIR/prompt_templates.yaml" "$RUN_MANIFEST" > "$RUN_DIR/audit/source_hashes.sha256"
curl -fsS "$BASE_URL/v1/models" > "$RUN_DIR/audit/models.json"
nvidia-smi > "$RUN_DIR/audit/nvidia_smi.txt"

export LLM_API_FORMAT=vllm
export LMSTUDIO_BASE_URL="$BASE_URL"
export AVAILABLE_MODELS_OVERRIDE="$MODEL_ALIAS"
export ACTIVE_POOLS_OVERRIDE=gemma4b-pool
export EXPERIMENTOS_CSV_PATH="$RUN_MANIFEST"
export EXPERIMENTOS_DB_PATH="$RUN_DIR/inputs/experiments.db"
export OUTPUT_BASE_DIR="$RUN_DIR/output"
export LLM_SEED=42
export LLM_MAX_OUTPUT_TOKENS="${LLM_MAX_OUTPUT_TOKENS:-3000}"
export GLOBAL_LLM_MAX_INFLIGHT="${GLOBAL_LLM_MAX_INFLIGHT:-1}"
export SYNC_DB_FROM_CSV=false
unset LLM_TOP_K LLM_TOP_P LLM_MIN_P LLM_REPEAT_PENALTY

"$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); import db_sqlite; db_sqlite.init_db(from_csv='$RUN_MANIFEST')"
exec "$PYTHON_BIN" "$SCRIPT_DIR/execucao_simultanea_vllm.py" --auto-orchestrate \
  2>&1 | tee "$RUN_DIR/orchestrator.log"
