#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PYTHON="${PYTHON:-$REPO_ROOT/artifacts/conda/runtime/bin/python}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
BASELINE_RUN_ID="${BASELINE_RUN_ID:?BASELINE_RUN_ID is required}"
TOKEN_PAIR="${TOKEN_PAIR:-01}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8127/v1}"
MODEL="${MODEL:-gemma3-4b-temp0}"
case "$TOKEN_PAIR" in
  01) TOKEN_FOLDER="tokens=0-1" ;;
  kz) TOKEN_FOLDER="tokens=k-z" ;;
  triangle_circle) TOKEN_FOLDER="tokens=triangle-circle" ;;
  *) echo "unsupported TOKEN_PAIR=$TOKEN_PAIR" >&2; exit 1 ;;
esac
CROSS_ROOT="$REPO_ROOT/artifacts/cross_phase_validation/rtx5090/n=7/gemma-3-4b-it/$TOKEN_FOLDER/$RUN_ID"
ORCHESTRATOR="$CROSS_ROOT/orchestrator"
PREFLIGHT="$CROSS_ROOT/preflight"
STATE="$ORCHESTRATOR/stage_status.jsonl"

mkdir -p "$ORCHESTRATOR" "$PREFLIGHT"
exec > >(tee -a "$ORCHESTRATOR/orchestrator.log") 2>&1

stage() {
  printf '{"utc":"%s","stage":"%s","status":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" | tee -a "$STATE"
}

terminal_status() {
  printf '{"utc":"%s","status":"%s","complete":true,"run_id":"%s","token_pair":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$RUN_ID" "$TOKEN_PAIR" > "$CROSS_ROOT/status.json"
  stage "$2" "$1"
  stage pipeline COMPLETE
  trap - ERR
  exit 0
}

on_error() {
  local status=$?
  printf '{"utc":"%s","status":"INCOMPLETE","complete":true,"exit_code":%s,"run_id":"%s","token_pair":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" "$RUN_ID" "$TOKEN_PAIR" > "$CROSS_ROOT/status.json"
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
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free,memory.total \
  --format=csv,noheader > "$PREFLIGHT/gpu_summary.csv"
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
  gradio_project/memory/run_w0_parity_online.py \
  infra/rtx5090/compare_memory_window_to_w0.py
"$PYTHON" -m unittest discover -s extract_rules/runtime_vllm/tests -p 'test_*.py'
bash gradio_project/tests/run_tests.sh

"$PYTHON" - "$BASE_URL" "$MODEL" "$TOKEN_PAIR" "$PREFLIGHT/canary.json" <<'PY'
import json
import sys
from pathlib import Path

from utils.w0_parity_contract import (
    build_responses_payload,
    query_responses_with_retries,
    render_memory_prompt,
    variants_for_pair,
)

base_url, model, token_pair, output = sys.argv[1:]
neighborhood = [0, 1, 0, 0, 0, 1, 1]
canaries = []
for variant in variants_for_pair(token_pair):
    system_prompt, user_prompt = render_memory_prompt(
        variant,
        neighborhood,
        memory_snapshots=[(0, neighborhood)],
    )
    payload = build_responses_payload(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        variant=variant,
    )
    results = [
        query_responses_with_retries(
            base_url=base_url,
            payload=payload,
            variant=variant,
            timeout_s=300,
            max_attempts=2,
        )
        for _ in range(2)
    ]
    assert results[0]["raw_response"] == results[1]["raw_response"]
    assert results[0]["choice"] == results[1]["choice"]
    assert results[0]["choice"] in (0, 1)
    assert results[0]["request_failures"] == 0
    assert results[0]["parse_failures"] == 0
    canaries.append(
        {
            "variant": variant,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "payload": payload,
            "results": results,
        }
    )
Path(output).write_text(
    json.dumps(
        {"token_pair": token_pair, "memory_window": 1, "canaries": canaries},
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)
PY
stage preflight PASS

stage phase3_online_w1 STARTED
set +e
"$PYTHON" gradio_project/memory/run_w0_parity_online.py \
  --run-id "$RUN_ID" --base-url "$BASE_URL" --model "$MODEL" \
  --token-pair "$TOKEN_PAIR" \
  --agents 30 --seeds 1-20 --majority-ratio 0.51 --memory-window 1 \
  --max-transitions 60 --request-workers 30
run_status=$?
set -e
if [[ "$run_status" -eq 0 ]]; then
  stage phase3_online_w1 PASS
elif [[ "$run_status" -eq 2 ]]; then
  terminal_status FAIL_NONDETERMINISTIC phase3_online_w1
else
  false
fi

stage compare_w1_to_w0 STARTED
"$PYTHON" infra/rtx5090/compare_memory_window_to_w0.py \
  --baseline-run-id "$BASELINE_RUN_ID" --run-id "$RUN_ID" \
  --memory-window 1 --token-pair "$TOKEN_PAIR" --seeds 1-20
stage compare_w1_to_w0 PASS

printf '{"utc":"%s","status":"PASS","complete":true,"run_id":"%s","token_pair":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ID" "$TOKEN_PAIR" > "$CROSS_ROOT/status.json"
find "$REPO_ROOT/artifacts/work/rtx5090" -depth -type d -empty -path "*$RUN_ID*" -delete
stage pipeline PASS
trap - ERR
echo "RUN_COMPLETE=$RUN_ID"
