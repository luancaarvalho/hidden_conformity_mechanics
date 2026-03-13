#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${OUT_ROOT:-$HOME/luan/projeto_final/streamlit_test/batch_outputs/A100/qwen4b_dual_prompt_v9_seed1_20}"
BASE_URL="${BASE_URL:-http://127.0.0.1:1234/v1}"
PROMPT_VARIANT="${PROMPT_VARIANT:-v9_lista_completa_meio_kz}"
AGENTS_LIST="${AGENTS_LIST:-30,60,90}"
MEMORY_WINDOWS="${MEMORY_WINDOWS:-3,5}"
NEIGHBORS="${NEIGHBORS:-7}"
INITIAL_MAJORITY_RATIO="${INITIAL_MAJORITY_RATIO:-0.51}"
INITIAL_DISTRIBUTION_MODE="${INITIAL_DISTRIBUTION_MODE:-auto}"
SEEDS_DISTRIBUTION="${SEEDS_DISTRIBUTION:-1-20}"
REQUEST_SEED="${REQUEST_SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-50}"
CONFORMITY_MODE="${CONFORMITY_MODE:-Normative DCT Convergence}"
STANDARD_RUN_MODE="${STANDARD_RUN_MODE:-skip}"
CONFORMITY_RUN_MODE="${CONFORMITY_RUN_MODE:-skip}"

export PATH="$HOME/.lmstudio/bin:$PATH"
export FORCE_QWEN_NO_THINK=1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate luan_conformidade

mkdir -p "$OUT_ROOT/standard" "$OUT_ROOT/conformity_game"

lms unload --all || true
lms load --exact -y --identifier qwen4b_std qwen/qwen3-4b
lms load --exact -y --identifier qwen4b_cg qwen/qwen3-4b
lms ps

python "$ROOT_DIR/run_qwen4b_a100_worker.py" \
  --base-url "$BASE_URL" \
  --model qwen4b_std \
  --output-root "$OUT_ROOT/standard" \
  --prompt-variant "$PROMPT_VARIANT" \
  --agents-list "$AGENTS_LIST" \
  --memory-windows "$MEMORY_WINDOWS" \
  --neighbors "$NEIGHBORS" \
  --initial-majority-ratio "$INITIAL_MAJORITY_RATIO" \
  --initial-distribution-mode "$INITIAL_DISTRIBUTION_MODE" \
  --seeds-distribution "$SEEDS_DISTRIBUTION" \
  --request-seed "$REQUEST_SEED" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --worker-index 0 \
  --worker-count 1 \
  --mode "$STANDARD_RUN_MODE" \
  > "$OUT_ROOT/standard_worker.log" 2>&1 &
STD_PID=$!

python "$ROOT_DIR/run_qwen4b_a100_worker.py" \
  --base-url "$BASE_URL" \
  --model qwen4b_cg \
  --output-root "$OUT_ROOT/conformity_game" \
  --prompt-variant "$PROMPT_VARIANT" \
  --agents-list "$AGENTS_LIST" \
  --memory-windows "$MEMORY_WINDOWS" \
  --neighbors "$NEIGHBORS" \
  --initial-majority-ratio "$INITIAL_MAJORITY_RATIO" \
  --initial-distribution-mode "$INITIAL_DISTRIBUTION_MODE" \
  --seeds-distribution "$SEEDS_DISTRIBUTION" \
  --request-seed "$REQUEST_SEED" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --worker-index 0 \
  --worker-count 1 \
  --mode "$CONFORMITY_RUN_MODE" \
  --conformity-game \
  --conformity-game-mode "$CONFORMITY_MODE" \
  > "$OUT_ROOT/conformity_worker.log" 2>&1 &
CG_PID=$!

wait "$STD_PID"
wait "$CG_PID"
