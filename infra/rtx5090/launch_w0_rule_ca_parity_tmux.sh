#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="rules-ca-parity-w0-seed1-20_${TIMESTAMP}"
SESSION="w0_rule_ca_parity_${TIMESTAMP}"
CROSS_ROOT="$REPO_ROOT/artifacts/cross_phase_validation/rtx5090/n=7/gemma-3-4b-it/tokens=0-1/$RUN_ID"

test -z "$(git -C "$REPO_ROOT" status --porcelain)"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$CROSS_ROOT/orchestrator"
printf '%s\n' "$SESSION" > "$CROSS_ROOT/orchestrator/tmux_session.txt"
printf '%s\n' "$RUN_ID" > "$CROSS_ROOT/orchestrator/run_id.txt"

tmux new-session -d -s "$SESSION" \
  "cd '$REPO_ROOT'; RUN_ID='$RUN_ID' bash infra/rtx5090/run_w0_rule_ca_parity_pipeline.sh; rc=\$?; sleep 1; '$REPO_ROOT/artifacts/conda/runtime/bin/python' infra/rtx5090/compare_w0_rule_ca_parity.py --run-id '$RUN_ID' --refresh-index-only >/dev/null; index_rc=\$?; if [ \$rc -eq 0 ] && [ \$index_rc -ne 0 ]; then rc=\$index_rc; fi; exit \$rc"

echo "session=$SESSION"
echo "run_id=$RUN_ID"
echo "cross_root=$CROSS_ROOT"
echo "attach=tmux attach -t $SESSION"
