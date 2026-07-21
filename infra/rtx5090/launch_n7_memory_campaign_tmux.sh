#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/artifacts/conda/runtime/bin/python}"
TIMESTAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
CAMPAIGN_ID="gemma4b_n7_w0_5_seed50_${TIMESTAMP}"
SESSION="$CAMPAIGN_ID"
CONTROL_ROOT="$REPO_ROOT/artifacts/orchestration/rtx5090/$CAMPAIGN_ID"
LOG="$CONTROL_ROOT/orchestrator.log"

cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to launch from a dirty Git worktree." >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Conda runtime not found: $PYTHON" >&2
  exit 2
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 2
fi

mkdir -p "$CONTROL_ROOT"
printf '%s\n' "$SESSION" > "$CONTROL_ROOT/tmux_session.txt"
printf '%s\n' "$CAMPAIGN_ID" > "$CONTROL_ROOT/campaign_id.txt"

tmux new-session -d -s "$SESSION" \
  "cd '$REPO_ROOT' && exec '$PYTHON' -u infra/rtx5090/run_n7_memory_campaign.py --campaign-id '$CAMPAIGN_ID' >> '$LOG' 2>&1"

printf 'CAMPAIGN_ID=%s\n' "$CAMPAIGN_ID"
printf 'TMUX_SESSION=%s\n' "$SESSION"
printf 'LOG=%s\n' "$LOG"
printf 'MONITOR=tmux attach -t %s\n' "$SESSION"

