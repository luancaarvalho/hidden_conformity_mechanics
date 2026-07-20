#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PYTHON="${PYTHON:-$REPO_ROOT/artifacts/conda/runtime/bin/python}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8127/v1}"
MODEL="${MODEL:-gemma3-4b-temp0}"
CROSS_ROOT="$REPO_ROOT/artifacts/cross_phase_validation/rtx5090/n=7/gemma-3-4b-it/tokens=0-1/$RUN_ID"
ORCHESTRATOR="$CROSS_ROOT/orchestrator"
PREFLIGHT="$CROSS_ROOT/preflight"
STATE="$ORCHESTRATOR/stage_status.jsonl"

mkdir -p "$ORCHESTRATOR" "$PREFLIGHT"
exec > >(tee -a "$ORCHESTRATOR/orchestrator.log") 2>&1

stage() {
  local name="$1"
  local status="$2"
  printf '{"utc":"%s","stage":"%s","status":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" "$status" | tee -a "$STATE"
}

terminal_status() {
  local status="$1"
  local stage_name="$2"
  printf '{"utc":"%s","status":"%s","complete":true,"run_id":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" "$RUN_ID" > "$CROSS_ROOT/status.json"
  stage "$stage_name" "$status"
  stage pipeline COMPLETE
  trap - ERR
  exit 0
}

run_determinism_gate() {
  local stage_name="$1"
  shift
  stage "$stage_name" STARTED
  set +e
  "$@"
  local status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    stage "$stage_name" PASS
  elif [[ "$status" -eq 2 ]]; then
    terminal_status FAIL_NONDETERMINISTIC "$stage_name"
  else
    return "$status"
  fi
}

on_error() {
  local status=$?
  printf '{"utc":"%s","status":"INCOMPLETE","exit_code":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" > "$CROSS_ROOT/status.json"
  stage pipeline FAIL
  exit "$status"
}
trap on_error ERR

cd "$REPO_ROOT"
test -x "$PYTHON"
test -z "$(git status --porcelain)"

stage preflight STARTED
git rev-parse HEAD > "$PREFLIGHT/git_commit.txt"
git status --short --branch > "$PREFLIGHT/git_status.txt"
"$PYTHON" --version > "$PREFLIGHT/python.txt" 2>&1
nvidia-smi > "$PREFLIGHT/nvidia_smi.txt"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader \
  > "$PREFLIGHT/gpu_summary.csv"
ps -eo pid,ppid,etime,%cpu,%mem,args > "$PREFLIGHT/processes.txt"
ss -ltnp > "$PREFLIGHT/ports.txt"
tmux ls > "$PREFLIGHT/tmux.txt" 2>&1 || true
curl -fsS "$BASE_URL/models" > "$PREFLIGHT/vllm_model.json"

VLLM_PID="$(ps -eo pid,args | awk '/python.*vllm.*serve.*--port 8127/ {print $1; exit}')"
test -n "$VLLM_PID"
tr '\0' '\n' < "/proc/$VLLM_PID/environ" \
  | grep -E '^VLLM_(BATCH_INVARIANT|USE_FLASHINFER_SAMPLER)=' \
  > "$PREFLIGHT/vllm_environment.txt"
grep -qx 'VLLM_BATCH_INVARIANT=1' "$PREFLIGHT/vllm_environment.txt"
grep -qx 'VLLM_USE_FLASHINFER_SAMPLER=0' "$PREFLIGHT/vllm_environment.txt"
ps -p "$VLLM_PID" -o pid,ppid,lstart,args > "$PREFLIGHT/vllm_process.txt"
grep -q -- '--generation-config vllm' "$PREFLIGHT/vllm_process.txt"
grep -q -- '--seed 42' "$PREFLIGHT/vllm_process.txt"
grep -q -- '--max-num-seqs 32' "$PREFLIGHT/vllm_process.txt"

"$PYTHON" -m py_compile \
  utils/w0_parity_contract.py \
  utils/parity_artifacts.py \
  utils/parity_render.py \
  extract_rules/runtime_vllm/run_w0_parity_extraction.py \
  experimentos_automatos/runtime/run_w0_parity_automata.py \
  gradio_project/memory/run_w0_parity_online.py \
  infra/rtx5090/compare_w0_rule_ca_parity.py
"$PYTHON" -m unittest discover -s extract_rules/runtime_vllm/tests -p 'test_*.py'
"$PYTHON" -m unittest discover -s experimentos_automatos/runtime/tests -p 'test_*.py'
bash gradio_project/tests/run_tests.sh

"$PYTHON" - "$BASE_URL" "$MODEL" "$PREFLIGHT/canary.json" <<'PY'
import json
import sys
from pathlib import Path

from utils.w0_parity_contract import (
    build_responses_payload,
    query_responses_with_retries,
    render_w0_prompt,
)

base_url, model, output = sys.argv[1:]
system_prompt, user_prompt = render_w0_prompt(
    "v9_lista_completa_meio_parity_01", [0, 1, 0, 1, 0, 1, 0]
)
payload = build_responses_payload(
    model=model,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    variant="v9_lista_completa_meio_parity_01",
)
results = [
    query_responses_with_retries(
        base_url=base_url, payload=payload, timeout_s=120, max_attempts=2
    )
    for _ in range(2)
]
assert results[0]["raw_response"] == results[1]["raw_response"]
assert results[0]["choice"] == results[1]["choice"]
Path(output).write_text(json.dumps({"payload": payload, "results": results}, indent=2) + "\n")
PY
stage preflight PASS

run_determinism_gate phase1_rule_extraction \
  "$PYTHON" extract_rules/runtime_vllm/run_w0_parity_extraction.py \
  --run-id "$RUN_ID" --base-url "$BASE_URL" --model "$MODEL" \
  --neighbors 3 5 7 --workers 32

run_determinism_gate phase2_cellular_automata \
  "$PYTHON" experimentos_automatos/runtime/run_w0_parity_automata.py \
  --run-id "$RUN_ID" --agents 30 --seeds 1-20 --majority-ratio 0.51 --max-transitions 60

run_determinism_gate phase3_online_w0 \
  "$PYTHON" gradio_project/memory/run_w0_parity_online.py \
  --run-id "$RUN_ID" --base-url "$BASE_URL" --model "$MODEL" \
  --agents 30 --seeds 1-20 --majority-ratio 0.51 --max-transitions 60 \
  --request-workers 30

stage cross_phase_proofread STARTED
set +e
"$PYTHON" infra/rtx5090/compare_w0_rule_ca_parity.py \
  --run-id "$RUN_ID" --seeds 1-20
comparison_status=$?
set -e
if [[ "$comparison_status" -eq 0 ]]; then
  stage cross_phase_proofread PASS
elif [[ "$comparison_status" -eq 3 ]]; then
  terminal_status FAIL_PARITY cross_phase_proofread
else
  false
fi

printf '{"utc":"%s","status":"PASS","run_id":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ID" > "$CROSS_ROOT/status.json"
stage pipeline PASS
trap - ERR
echo "RUN_COMPLETE=$RUN_ID"
