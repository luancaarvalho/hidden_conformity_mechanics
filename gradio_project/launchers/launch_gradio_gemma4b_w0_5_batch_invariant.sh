#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/artifacts/conda/runtime/bin/python}"
TIMESTAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/artifacts/phase3_memory/RTX5090_liaan}"
BATCH_ROOT="$RESULTS_DIR/gemma4b_01_n30_neigh7_W0_5_seed1_10_batch_invariant_${TIMESTAMP}"
RUNNER="$REPO_ROOT/gradio_project/memory/run_gradio_gemma4b_01_matrix.py"
COMPARATOR="$REPO_ROOT/gradio_project/memory/compare_gradio_gemma4b_w_replays.py"
FINALIZER="$REPO_ROOT/gradio_project/memory/finalize_gradio_gemma4b_w0_5.py"
MODES=(standard_only_token standard_cot conformity_only_token conformity_cot)

mkdir -p "$BATCH_ROOT"
exec > >(tee -a "$BATCH_ROOT/orchestrator.log") 2>&1

echo "BATCH_ROOT=$BATCH_ROOT"
echo "TIMESTAMP=$TIMESTAMP"

{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader
  curl -fsS http://127.0.0.1:8127/v1/models
  echo
  ps -eo pid,ppid,etime,%cpu,%mem,args | grep -E '[v]llm serve.*--port 8127|[E]ngineCore'
} | tee "$BATCH_ROOT/preflight.txt"

VLLM_PID="$(ps -eo pid,args | awk '/python.*vllm.*serve.*--port 8127/ {print $1; exit}')"
test -n "$VLLM_PID"
tr '\0' '\n' < "/proc/$VLLM_PID/environ" | grep -qx 'VLLM_BATCH_INVARIANT=1'
tr '\0' '\n' < "/proc/$VLLM_PID/environ" | grep -qx 'VLLM_USE_FLASHINFER_SAMPLER=0'
"$PYTHON" -m py_compile "$RUNNER" "$COMPARATOR" "$FINALIZER"

run_smoke() {
  local mode="$1"
  local window="$2"
  local output="$BATCH_ROOT/smoke/${mode}_W${window}"
  "$PYTHON" "$RUNNER" \
    --output-root "$output" \
    --agents 30 \
    --rounds 10 \
    --neighbors 7 \
    --memory-window "$window" \
    --initial-majority 51 \
    --temperature 0 \
    --seeds 1 \
    --workers 1 \
    --server-batch-invariant \
    --modes "$mode"
}

run_smoke standard_only_token 0
run_smoke conformity_cot 5

W0_SMOKE="$BATCH_ROOT/smoke/standard_only_token_W0/standard_only_token/cells/seed_01/prompt_log_normalized.txt"
W5_SMOKE="$BATCH_ROOT/smoke/conformity_cot_W5/conformity_cot/cells/seed_01/prompt_log_normalized.txt"
if grep -q '=== MEMORY (Previous Rounds) ===' "$W0_SMOKE"; then
  echo "SMOKE_FAIL: W=0 contains MEMORY"
  exit 10
fi
grep -q '=== MEMORY (Previous Rounds) ===' "$W5_SMOKE"
echo "SMOKE_PASS"

run_replay() {
  local window="$1"
  local replay="$2"
  local replay_root="$BATCH_ROOT/staging/W${window}/${replay}"
  local log_dir="$replay_root/logs"
  local pids=()
  local modes=()
  mkdir -p "$log_dir"

  for mode in "${MODES[@]}"; do
    local output="$replay_root/$mode"
    "$PYTHON" "$RUNNER" \
      --output-root "$output" \
      --agents 30 \
      --rounds 10 \
      --neighbors 7 \
      --memory-window "$window" \
      --initial-majority 51 \
      --temperature 0 \
      --seeds 1-10 \
      --workers 8 \
      --server-batch-invariant \
      --modes "$mode" \
      >"$log_dir/${mode}.log" 2>&1 &
    pids+=("$!")
    modes+=("$mode")
  done

  local failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "RUN_FAIL W=$window replay=$replay mode=${modes[$index]}"
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    return 1
  fi

  for mode in "${MODES[@]}"; do
    "$PYTHON" - "$replay_root/$mode/summary.json" "$mode" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = sys.argv[2]
item = summary["modes"][mode]
assert summary["completed_cells"] == 10, summary
assert summary["error_cells"] == 0, summary
assert summary["png_count"] == 10, summary
assert item["completed"] == 10, item
assert item["parse_failures"] == 0, item
PY
  done
}

promote_passed_replay() {
  local window="$1"
  local replay1_root="$BATCH_ROOT/staging/W${window}/replay01"
  local replay2_root="$BATCH_ROOT/staging/W${window}/replay02"
  local proof="$BATCH_ROOT/determinism_proof/W${window}"

  for mode in "${MODES[@]}"; do
    cp "$replay2_root/$mode/manifest.json" "$proof/replay02_${mode}_manifest.json"
    cp "$replay2_root/$mode/manifest.jsonl" "$proof/replay02_${mode}_manifest.jsonl"
    cp "$replay2_root/$mode/scoreboard.csv" "$proof/replay02_${mode}_scoreboard.csv"
    cp "$replay2_root/$mode/summary.json" "$proof/replay02_${mode}_summary.json"
    cp "$replay2_root/logs/${mode}.log" "$proof/replay02_${mode}_runner.log"
  done

  "$PYTHON" - "$proof" <<'PY'
import csv
import json
import sys
from pathlib import Path

proof = Path(sys.argv[1])
summary = json.loads((proof / "summary.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader((proof / "paired_determinism_comparison.csv").open()))
hashes = (proof / "replay02_hashes.jsonl").read_text(encoding="utf-8").splitlines()
assert summary["determinism_gate_pass"] is True, summary
assert summary["exact_equal"] == 40, summary
assert len(rows) == 40 and all(row["exact_equal"] == "True" for row in rows)
assert len(hashes) == 40
PY

  for mode in "${MODES[@]}"; do
    local outer="$replay1_root/$mode"
    local final="$BATCH_ROOT/strategies/$mode/W${window}"
    mkdir -p "$(dirname "$final")"
    mv "$outer/$mode" "$final"
    mv "$outer/manifest.json" "$final/manifest.json"
    mv "$outer/manifest.jsonl" "$final/manifest.jsonl"
    mv "$outer/progress.jsonl" "$final/progress.jsonl"
    mv "$outer/scoreboard.csv" "$final/scoreboard.csv"
    mv "$outer/summary.json" "$final/summary.json"
    mv "$replay1_root/logs/${mode}.log" "$final/runner.log"
  done

  rm -rf "$replay2_root"
  rm -rf "$replay1_root"
  rmdir "$BATCH_ROOT/staging/W${window}" 2>/dev/null || true
}

for window in 0 1 2 3 4 5; do
  echo "=== W=$window REPLAY 1 ==="
  run_replay "$window" replay01
  echo "=== W=$window REPLAY 2 ==="
  run_replay "$window" replay02

  proof="$BATCH_ROOT/determinism_proof/W${window}"
  set +e
  "$PYTHON" "$COMPARATOR" \
    --replay1-root "$BATCH_ROOT/staging/W${window}/replay01" \
    --replay2-root "$BATCH_ROOT/staging/W${window}/replay02" \
    --memory-window "$window" \
    --output-dir "$proof"
  comparison_status=$?
  set -e
  if [[ "$comparison_status" -ne 0 ]]; then
    echo "DETERMINISM_FAIL W=$window status=$comparison_status"
    exit "$comparison_status"
  fi

  promote_passed_replay "$window"
  printf '{"memory_window":%s,"status":"PASS","retention":"replay01_plus_proof"}\n' "$window" \
    >> "$BATCH_ROOT/orchestrator_state.jsonl"
  echo "=== W=$window PASS ==="
done

"$PYTHON" "$FINALIZER" --batch-root "$BATCH_ROOT"
echo "BATCH_COMPLETE=$BATCH_ROOT"
