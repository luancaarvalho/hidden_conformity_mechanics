#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASELINE_RUN_ID="${BASELINE_RUN_ID:-rules-ca-parity-w0-seed1-20_20260720T143020Z}"
TIMESTAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="memory-W1-seed1-20_${TIMESTAMP}"
SESSION="memory_w1_${TIMESTAMP}"
CROSS_ROOT="$REPO_ROOT/artifacts/cross_phase_validation/rtx5090/n=7/gemma-3-4b-it/tokens=0-1/$RUN_ID"

test -z "$(git -C "$REPO_ROOT" status --porcelain)"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$CROSS_ROOT/orchestrator"
printf '%s\n' "$SESSION" > "$CROSS_ROOT/orchestrator/tmux_session.txt"
printf '%s\n' "$RUN_ID" > "$CROSS_ROOT/orchestrator/run_id.txt"
printf '%s\n' "$BASELINE_RUN_ID" > "$CROSS_ROOT/orchestrator/baseline_run_id.txt"

tmux new-session -d -s "$SESSION" \
  "cd '$REPO_ROOT'; RUN_ID='$RUN_ID' BASELINE_RUN_ID='$BASELINE_RUN_ID' bash infra/rtx5090/run_memory_w1_pipeline.sh"

echo "session=$SESSION"
echo "run_id=$RUN_ID"
echo "baseline_run_id=$BASELINE_RUN_ID"
echo "cross_root=$CROSS_ROOT"
echo "attach=tmux attach -t $SESSION"
