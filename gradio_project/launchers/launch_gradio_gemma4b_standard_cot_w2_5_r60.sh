#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/artifacts/conda/runtime/bin/python}"
TIMESTAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
BATCH_NAME="gemma4b_01_n30_neigh7_standard_cot_W2_5_seed1_10_r60_batch_invariant_${TIMESTAMP}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/artifacts/phase3_memory/RTX5090_liaan}"
BATCH_ROOT="$RESULTS_DIR/$BATCH_NAME"
RUNNER="$REPO_ROOT/gradio_project/memory/run_gradio_gemma4b_01_matrix.py"
COMPARATOR="$REPO_ROOT/gradio_project/memory/compare_gradio_gemma4b_w_replays.py"
FINALIZER="$REPO_ROOT/gradio_project/memory/finalize_gradio_gemma4b_w0_5.py"
INTERFACE="$REPO_ROOT/gradio_project/interface/interface_v4_gradio.py"
MODE="standard_cot"
WINDOWS=(2 3 4 5)
MAX_ROUNDS=60
MAX_OUTPUT_TOKENS=3000
REQUEST_TIMEOUT_S=600

mkdir -p "$BATCH_ROOT"
exec > >(tee -a "$BATCH_ROOT/orchestrator.log") 2>&1

echo "BATCH_NAME=$BATCH_NAME"
echo "BATCH_ROOT=$BATCH_ROOT"
echo "TIMESTAMP=$TIMESTAMP"

{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader
  curl -fsS http://127.0.0.1:8127/v1/models
  echo
  ps -eo pid,ppid,etime,%cpu,%mem,args | grep -E '[v]llm.*serve.*8127|[E]ngineCore'
} | tee "$BATCH_ROOT/preflight.txt"

VLLM_PID="$(
  ps -eo pid=,args= |
    awk '$0 ~ /python/ && $0 ~ /vllm/ && $0 ~ /serve/ && $0 ~ /--port 8127/ {print $1; exit}'
)"
test -n "$VLLM_PID"
tr '\0' '\n' < "/proc/$VLLM_PID/environ" | grep -qx 'VLLM_BATCH_INVARIANT=1'
tr '\0' '\n' < "/proc/$VLLM_PID/environ" | grep -qx 'VLLM_USE_FLASHINFER_SAMPLER=0'
VLLM_ARGS="$(ps -p "$VLLM_PID" -o args=)"
grep -q -- '--generation-config vllm' <<<"$VLLM_ARGS"
grep -q -- '--seed 42' <<<"$VLLM_ARGS"
printf '%s\n' "$VLLM_ARGS" > "$BATCH_ROOT/vllm_command.txt"

curl -fsS http://127.0.0.1:8127/v1/models > "$BATCH_ROOT/models.json"
"$PYTHON" - "$BATCH_ROOT/models.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
models = payload.get("data", [])
assert len(models) == 1, models
model = models[0]
assert model["id"] == "gemma3-4b-temp0", model
assert str(model.get("root", "")).endswith("google-gemma-3-4b-it"), model
assert int(model.get("max_model_len", 0)) >= 4096, model
PY

"$PYTHON" -m py_compile "$RUNNER" "$COMPARATOR" "$FINALIZER" "$INTERFACE"

"$PYTHON" - "$BATCH_ROOT/experiment_contract.json" "$TIMESTAMP" <<'PY'
import json
import sys
from pathlib import Path

contract = {
    "created_at_utc": sys.argv[2],
    "model": "google/gemma-3-4b-it",
    "served_model": "gemma3-4b-temp0",
    "mode": "standard_cot",
    "prompt_variant": "v21_zero_shot_cot_01",
    "agents": 30,
    "neighbors": 7,
    "tokens": ["0", "1"],
    "initial_majority_percent": 51,
    "seeds": list(range(1, 11)),
    "memory_windows": [2, 3, 4, 5],
    "max_rounds": 60,
    "stop_rules": ["consensus", "max_rounds"],
    "sampling": {
        "temperature": 0,
        "seed": 42,
        "max_output_tokens": 3000,
        "top_k": "omitted",
        "top_p": "omitted",
        "min_p": "omitted",
        "repeat_penalty": "omitted",
    },
    "server": {
        "base_url": "http://127.0.0.1:8127/v1",
        "batch_invariant": True,
        "flashinfer_sampler": False,
        "generation_config": "vllm",
        "seed": 42,
    },
    "replays": 2,
    "expected_scientific_cells": 40,
    "expected_total_executions": 80,
}
Path(sys.argv[1]).write_text(
    json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

run_matrix() {
  local output_root="$1"
  local window="$2"
  local seeds="$3"
  local workers="$4"

  "$PYTHON" "$RUNNER" \
    --output-root "$output_root" \
    --agents 30 \
    --rounds "$MAX_ROUNDS" \
    --neighbors 7 \
    --memory-window "$window" \
    --initial-majority 51 \
    --temperature 0 \
    --seeds "$seeds" \
    --workers "$workers" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --request-timeout-s "$REQUEST_TIMEOUT_S" \
    --server-batch-invariant \
    --modes "$MODE"
}

validate_run() {
  local output_root="$1"
  local window="$2"
  local expected_cells="$3"

  "$PYTHON" - "$output_root" "$window" "$expected_cells" <<'PY'
import csv
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
window = int(sys.argv[2])
expected = int(sys.argv[3])
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader((root / "scoreboard.csv").open(encoding="utf-8")))
assert summary["completed_cells"] == expected, summary
assert summary["error_cells"] == 0, summary
assert summary["png_count"] == expected, summary
assert len(rows) == expected, rows
assert summary["modes"]["standard_cot"]["parse_failures"] == 0, summary
assert summary["modes"]["standard_cot"]["request_failures"] == 0, summary
for row in rows:
    assert row["mode"] == "standard_cot", row
    assert int(row["memory_window"]) == window, row
    assert int(row["rounds"]) == 60, row
    assert int(row["max_output_tokens"]) == 3000, row
    assert int(row["request_timeout_s"]) == 600, row
    assert int(row["parse_failure_count"]) == 0, row
    assert int(row["request_failure_count"]) == 0, row
    assert row["stop_reason"] in {"consensus", "max_rounds"}, row
    assert not row["status"].startswith("Erro"), row
    state_path = root / "standard_cot" / "cells" / f"seed_{int(row['seed']):02d}" / "states.npy"
    states = np.load(state_path)
    assert states.shape == (60, 30), (state_path, states.shape)
PY
}

validate_memory_log() {
  local log_path="$1"
  local window="$2"
  "$PYTHON" - "$log_path" "$window" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
window = int(sys.argv[2])
marker = "=== MEMORY (Previous Rounds) ==="
assert marker in text, (window, "missing memory")
for block in text.split(marker)[1:]:
    memory_text = block.split("\n\n", 1)[0]
    count = len(re.findall(r"^Round \d+:$", memory_text, flags=re.MULTILINE))
    assert 1 <= count <= window, (window, count)
PY
}

for window in 2 5; do
  smoke_root="$BATCH_ROOT/smoke/standard_cot_W${window}"
  echo "=== SMOKE W=$window ==="
  run_matrix "$smoke_root" "$window" 1 1
  validate_run "$smoke_root" "$window" 1
  validate_memory_log \
    "$smoke_root/standard_cot/cells/seed_01/prompt_log_normalized.txt" \
    "$window"
done
echo "SMOKE_PASS"

run_replay() {
  local window="$1"
  local replay="$2"
  local replay_root="$BATCH_ROOT/staging/W${window}/${replay}"
  mkdir -p "$replay_root/logs"
  run_matrix "$replay_root/$MODE" "$window" 1-10 10 \
    >"$replay_root/logs/${MODE}.log" 2>&1
  validate_run "$replay_root/$MODE" "$window" 10
  test "$(wc -l < "$replay_root/$MODE/manifest.jsonl")" -eq 10
}

promote_passed_replay() {
  local window="$1"
  local replay1_root="$BATCH_ROOT/staging/W${window}/replay01"
  local replay2_root="$BATCH_ROOT/staging/W${window}/replay02"
  local proof="$BATCH_ROOT/determinism_proof/W${window}"
  local outer="$replay1_root/$MODE"
  local final="$BATCH_ROOT/strategies/$MODE/W${window}"

  cp "$replay2_root/$MODE/manifest.json" "$proof/replay02_${MODE}_manifest.json"
  cp "$replay2_root/$MODE/manifest.jsonl" "$proof/replay02_${MODE}_manifest.jsonl"
  cp "$replay2_root/$MODE/scoreboard.csv" "$proof/replay02_${MODE}_scoreboard.csv"
  cp "$replay2_root/$MODE/summary.json" "$proof/replay02_${MODE}_summary.json"
  cp "$replay2_root/logs/${MODE}.log" "$proof/replay02_${MODE}_runner.log"

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
assert summary["exact_equal"] == 10, summary
assert len(rows) == 10 and all(row["exact_equal"] == "True" for row in rows)
assert all(int(row["request_failures_replay1"]) == 0 for row in rows)
assert all(int(row["request_failures_replay2"]) == 0 for row in rows)
assert len(hashes) == 10
PY

  mkdir -p "$(dirname "$final")"
  mv "$outer/$MODE" "$final"
  mv "$outer/manifest.json" "$final/manifest.json"
  mv "$outer/manifest.jsonl" "$final/manifest.jsonl"
  mv "$outer/progress.jsonl" "$final/progress.jsonl"
  mv "$outer/scoreboard.csv" "$final/scoreboard.csv"
  mv "$outer/summary.json" "$final/summary.json"
  mv "$replay1_root/logs/${MODE}.log" "$final/runner.log"

  rm -rf "$replay2_root" "$replay1_root"
  rmdir "$BATCH_ROOT/staging/W${window}" 2>/dev/null || true
}

for window in "${WINDOWS[@]}"; do
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
    --output-dir "$proof" \
    --modes "$MODE"
  comparison_status=$?
  set -e
  if [[ "$comparison_status" -ne 0 ]]; then
    echo "DETERMINISM_FAIL W=$window status=$comparison_status"
    exit "$comparison_status"
  fi

  promote_passed_replay "$window"
  printf '{"memory_window":%s,"status":"PASS","exact":"10/10","retention":"replay01_plus_proof"}\n' "$window" \
    >> "$BATCH_ROOT/orchestrator_state.jsonl"
  echo "=== W=$window PASS ==="
done

"$PYTHON" "$FINALIZER" \
  --batch-root "$BATCH_ROOT" \
  --modes "$MODE" \
  --windows 2,3,4,5 \
  --expected-rounds "$MAX_ROUNDS"

test "$(find "$BATCH_ROOT/strategies/$MODE" -path '*/cells/seed_*/states.npy' | wc -l)" -eq 40
test "$(find "$BATCH_ROOT/strategies/$MODE" -maxdepth 2 -name 'seed_*.png' | wc -l)" -eq 40
test "$(find "$BATCH_ROOT/slides_4k/$MODE" -maxdepth 1 -name 'W*.png' | wc -l)" -eq 4

printf 'canonical_artifact_path=%s\nresolved_artifact_path=%s\n' \
  "$BATCH_ROOT" "$(realpath "$BATCH_ROOT")" \
  > "$BATCH_ROOT/artifact_verification.txt"

echo "BATCH_COMPLETE=$BATCH_ROOT"
echo "CANONICAL_ARTIFACT=$BATCH_ROOT"
