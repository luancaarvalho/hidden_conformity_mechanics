#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${OUT_ROOT:-$HOME/luan/projeto_final/streamlit_test/batch_outputs/A100/qwen4b_workers_seed2}"
BASE_URL="${BASE_URL:-http://127.0.0.1:1234/v1}"
PROMPT_VARIANT="${PROMPT_VARIANT:-v21_zero_shot_cot}"
AGENTS_LIST="${AGENTS_LIST:-30,60,90}"
MEMORY_WINDOWS="${MEMORY_WINDOWS:-3,5}"
NEIGHBORS="${NEIGHBORS:-7}"
INITIAL_MAJORITY_RATIO="${INITIAL_MAJORITY_RATIO:-0.51}"
INITIAL_DISTRIBUTION_MODE="${INITIAL_DISTRIBUTION_MODE:-half_split}"
SEEDS_DISTRIBUTION="${SEEDS_DISTRIBUTION:-2}"
REQUEST_SEED="${REQUEST_SEED:-2}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-3000}"
CONFORMITY_MODE="${CONFORMITY_MODE:-Normative DCT Convergence}"

export PATH="$HOME/.lmstudio/bin:$PATH"
export FORCE_QWEN_NO_THINK=1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate luan_conformidade

mkdir -p "$OUT_ROOT"

load_aliases() {
  local count="$1"
  lms unload --all || true
  local i=1
  while [[ "$i" -le "$count" ]]; do
    lms load --exact -y --identifier "qwen4b_w${i}" qwen/qwen3-4b
    i=$((i + 1))
  done
  lms ps
}

run_suite() {
  local tag="$1"
  local models="$2"
  shift 2
  local suite_dir="${OUT_ROOT}/${tag}"
  mkdir -p "$suite_dir"

  python "$ROOT_DIR/run_qwen4b_a100_suite.py" \
    --base-url "$BASE_URL" \
    --models "$models" \
    --output-root "$suite_dir" \
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
    --mode overwrite \
    "$@"

  python "$ROOT_DIR/summarize_qwen4b_a100_suite.py" "$suite_dir" | tee "$suite_dir/suite_summary.pretty.json"
}

load_aliases 2
run_suite "v21_standard_memory/2x" "qwen4b_w1,qwen4b_w2"
load_aliases 4
run_suite "v21_standard_memory/4x" "qwen4b_w1,qwen4b_w2,qwen4b_w3,qwen4b_w4"
load_aliases 2
run_suite "v21_conformity_game_memory/2x" "qwen4b_w1,qwen4b_w2" --conformity-game --conformity-game-mode "$CONFORMITY_MODE"
load_aliases 4
run_suite "v21_conformity_game_memory/4x" "qwen4b_w1,qwen4b_w2,qwen4b_w3,qwen4b_w4" --conformity-game --conformity-game-mode "$CONFORMITY_MODE"
