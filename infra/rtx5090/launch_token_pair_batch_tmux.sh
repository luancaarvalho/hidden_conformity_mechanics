#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
BATCH_ID="token-pair-kz-triangle-circle_${TIMESTAMP}"
SESSION="token_pair_batch_${TIMESTAMP}"
ORCHESTRATOR="$REPO_ROOT/artifacts/orchestration/rtx5090/$BATCH_ID"

test -z "$(git -C "$REPO_ROOT" status --porcelain)"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$ORCHESTRATOR"
printf '%s\n' "$SESSION" > "$ORCHESTRATOR/tmux_session.txt"
printf '%s\n' "$BATCH_ID" > "$ORCHESTRATOR/batch_id.txt"
printf '%s\n' "$TIMESTAMP" > "$ORCHESTRATOR/timestamp.txt"

tmux new-session -d -s "$SESSION" \
  "cd '$REPO_ROOT'; BATCH_ID='$BATCH_ID' TIMESTAMP='$TIMESTAMP' bash infra/rtx5090/run_token_pair_batch.sh"

echo "session=$SESSION"
echo "batch_id=$BATCH_ID"
echo "orchestrator=$ORCHESTRATOR"
echo "attach=tmux attach -t $SESSION"
