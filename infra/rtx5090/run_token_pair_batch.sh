#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PYTHON="${PYTHON:-$REPO_ROOT/artifacts/conda/runtime/bin/python}"
BATCH_ID="${BATCH_ID:?BATCH_ID is required}"
TIMESTAMP="${TIMESTAMP:?TIMESTAMP is required}"
RUN_01_W0="${RUN_01_W0:-rules-ca-parity-w0-seed1-20_20260720T143020Z}"
RUN_01_W1="${RUN_01_W1:-memory-W1-seed1-20_20260720T174027Z}"
ORCHESTRATOR="$REPO_ROOT/artifacts/orchestration/rtx5090/$BATCH_ID"
STATE="$ORCHESTRATOR/stage_status.jsonl"

mkdir -p "$ORCHESTRATOR"
exec > >(tee -a "$ORCHESTRATOR/orchestrator.log") 2>&1

stage() {
  printf '{"utc":"%s","stage":"%s","status":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" | tee -a "$STATE"
}

token_folder() {
  case "$1" in
    kz) printf '%s\n' 'tokens=k-z' ;;
    triangle_circle) printf '%s\n' 'tokens=triangle-circle' ;;
    *) return 1 ;;
  esac
}

status_for() {
  "$PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(json.loads(path.read_text(encoding="utf-8"))["status"] if path.is_file() else "MISSING")
PY
}

record_pair() {
  "$PYTHON" - "$ORCHESTRATOR/pair_status.jsonl" "$1" "$2" "$3" "$4" "$5" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, pair, w0_run, w0_status, w1_run, w1_status = sys.argv[1:]
row = {
    "utc": datetime.now(timezone.utc).isoformat(),
    "token_pair": pair,
    "w0_run_id": w0_run,
    "w0_status": w0_status,
    "w1_run_id": w1_run,
    "w1_status": w1_status,
}
with Path(path).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
PY
}

on_error() {
  local rc=$?
  printf '{"utc":"%s","status":"INCOMPLETE","exit_code":%s,"batch_id":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$BATCH_ID" > "$ORCHESTRATOR/status.json"
  stage batch FAIL
  exit "$rc"
}
trap on_error ERR

cd "$REPO_ROOT"
test -x "$PYTHON"
test -z "$(git status --porcelain)"

stage batch_preflight STARTED
git rev-parse HEAD > "$ORCHESTRATOR/git_commit.txt"
git status --short --branch > "$ORCHESTRATOR/git_status.txt"
nvidia-smi > "$ORCHESTRATOR/nvidia_smi.txt"
ps -eo pid,ppid,etime,%cpu,%mem,args > "$ORCHESTRATOR/processes.txt"
ss -ltnp > "$ORCHESTRATOR/ports.txt"
curl -fsS http://127.0.0.1:8127/v1/models > "$ORCHESTRATOR/vllm_model.json"
"$PYTHON" -m py_compile \
  utils/w0_parity_contract.py \
  extract_rules/runtime_vllm/run_w0_parity_extraction.py \
  experimentos_automatos/runtime/run_w0_parity_automata.py \
  gradio_project/memory/run_w0_parity_online.py \
  infra/rtx5090/compare_w0_rule_ca_parity.py \
  infra/rtx5090/compare_memory_window_to_w0.py \
  infra/rtx5090/compare_token_pairs.py
bash -n \
  infra/rtx5090/run_w0_rule_ca_parity_pipeline.sh \
  infra/rtx5090/run_memory_w1_pipeline.sh \
  infra/rtx5090/run_token_pair_batch.sh \
  infra/rtx5090/launch_token_pair_batch_tmux.sh
stage batch_preflight PASS

declare -A W0_RUNS
declare -A W1_RUNS
declare -A W0_STATUSES
declare -A W1_STATUSES

for pair in kz triangle_circle; do
  folder="$(token_folder "$pair")"
  w0_run="rules-ca-parity-w0-${pair}-seed1-20_${TIMESTAMP}"
  w1_run="memory-W1-${pair}-seed1-20_${TIMESTAMP}"
  W0_RUNS[$pair]="$w0_run"
  W1_RUNS[$pair]="$w1_run"

  stage "${pair}_w0" STARTED
  set +e
  TOKEN_PAIR="$pair" RUN_ID="$w0_run" \
    bash infra/rtx5090/run_w0_rule_ca_parity_pipeline.sh
  w0_rc=$?
  set -e
  w0_status_path="$REPO_ROOT/artifacts/cross_phase_validation/rtx5090/n=7/gemma-3-4b-it/$folder/$w0_run/status.json"
  w0_status="$(status_for "$w0_status_path")"
  W0_STATUSES[$pair]="$w0_status"
  if [[ "$w0_rc" -ne 0 || "$w0_status" != PASS ]]; then
    W1_STATUSES[$pair]="SKIPPED_DEPENDENCY"
    stage "${pair}_w0" "$w0_status"
    stage "${pair}_w1" SKIPPED_DEPENDENCY
    record_pair "$pair" "$w0_run" "$w0_status" "$w1_run" SKIPPED_DEPENDENCY
    continue
  fi
  stage "${pair}_w0" PASS

  stage "${pair}_w1" STARTED
  set +e
  TOKEN_PAIR="$pair" RUN_ID="$w1_run" BASELINE_RUN_ID="$w0_run" \
    bash infra/rtx5090/run_memory_w1_pipeline.sh
  w1_rc=$?
  set -e
  w1_status_path="$REPO_ROOT/artifacts/cross_phase_validation/rtx5090/n=7/gemma-3-4b-it/$folder/$w1_run/status.json"
  w1_status="$(status_for "$w1_status_path")"
  W1_STATUSES[$pair]="$w1_status"
  if [[ "$w1_rc" -ne 0 || "$w1_status" != PASS ]]; then
    stage "${pair}_w1" "$w1_status"
    record_pair "$pair" "$w0_run" "$w0_status" "$w1_run" "$w1_status"
    continue
  fi
  stage "${pair}_w1" PASS
  record_pair "$pair" "$w0_run" "$w0_status" "$w1_run" "$w1_status"
done

if [[ "${W0_STATUSES[kz]}" == PASS && "${W1_STATUSES[kz]}" == PASS \
   && "${W0_STATUSES[triangle_circle]}" == PASS \
   && "${W1_STATUSES[triangle_circle]}" == PASS ]]; then
  stage cross_token_proofread STARTED
  "$PYTHON" infra/rtx5090/compare_token_pairs.py \
    --batch-id "$BATCH_ID" \
    --run-01-w0 "$RUN_01_W0" --run-01-w1 "$RUN_01_W1" \
    --run-kz-w0 "${W0_RUNS[kz]}" --run-kz-w1 "${W1_RUNS[kz]}" \
    --run-triangle-circle-w0 "${W0_RUNS[triangle_circle]}" \
    --run-triangle-circle-w1 "${W1_RUNS[triangle_circle]}"
  stage cross_token_proofread PASS
  batch_status=PASS
else
  stage cross_token_proofread SKIPPED_DEPENDENCY
  batch_status=TERMINAL_PARTIAL
fi

printf '{"utc":"%s","status":"%s","complete":true,"batch_id":"%s","kz_w0":"%s","kz_w1":"%s","triangle_circle_w0":"%s","triangle_circle_w1":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$batch_status" "$BATCH_ID" \
  "${W0_RUNS[kz]}" "${W1_RUNS[kz]}" \
  "${W0_RUNS[triangle_circle]}" "${W1_RUNS[triangle_circle]}" \
  > "$ORCHESTRATOR/status.json"
stage batch "$batch_status"
trap - ERR
echo "BATCH_COMPLETE=$BATCH_ID"
