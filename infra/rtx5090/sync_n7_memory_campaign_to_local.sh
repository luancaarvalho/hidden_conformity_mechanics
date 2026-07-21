#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <CAMPAIGN_ID>" >&2
  exit 2
fi

CAMPAIGN_ID="$1"
REMOTE="${REMOTE:-rtx5090}"
REMOTE_REPO="${REMOTE_REPO:-/home/liaan/Documentos/Luan/hidden_conformity_mechanics}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
LOCAL_ROOT="$WORKSPACE_ROOT/projeto_final/resultados_simulacoes_conformidade/rtx5090/n=7"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

REMOTE_STATUS="$REMOTE_REPO/artifacts/orchestration/rtx5090/$CAMPAIGN_ID/terminal_status.json"
STATUS="$(ssh "$REMOTE" "python3 -c \"import json; print(json.load(open('$REMOTE_STATUS'))['status'])\"")"
if [[ "$STATUS" != "PASS" && "$STATUS" != FAIL_* ]]; then
  echo "Campaign is not terminal: $STATUS" >&2
  exit 3
fi

mkdir -p "$LOCAL_ROOT"
rsync -a --partial --info=progress2 \
  "$REMOTE:$REMOTE_REPO/artifacts/orchestration/rtx5090/$CAMPAIGN_ID/" \
  "$LOCAL_ROOT/orchestration/$CAMPAIGN_ID/"

for phase in phase1_rule_extraction phase2_cellular_automata phase3_memory; do
  mkdir -p "$LOCAL_ROOT/$phase"
  rsync -a --partial --info=progress2 \
    "$REMOTE:$REMOTE_REPO/artifacts/$phase/rtx5090/n=7/gemma-3-4b-it/" \
    "$LOCAL_ROOT/$phase/gemma-3-4b-it/"
done

TIMESTAMP="${CAMPAIGN_ID##*_}"
AGGREGATE="W0-5_seed1-50_$TIMESTAMP"
mkdir -p "$LOCAL_ROOT/cross_window_validation/gemma-3-4b-it"
rsync -a --partial --info=progress2 \
  "$REMOTE:$REMOTE_REPO/artifacts/cross_window_validation/rtx5090/n=7/gemma-3-4b-it/$AGGREGATE/" \
  "$LOCAL_ROOT/cross_window_validation/gemma-3-4b-it/$AGGREGATE/"

hash_local() {
  local root="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$root" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
  else
    (cd "$root" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256)
  fi
}

verify_tree() {
  local remote_root="$1"
  local local_root="$2"
  local name="$3"
  ssh "$REMOTE" "cd '$remote_root' && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum" \
    > "$TMP_DIR/$name.remote"
  hash_local "$local_root" > "$TMP_DIR/$name.local"
  diff -u "$TMP_DIR/$name.remote" "$TMP_DIR/$name.local"
}

verify_tree \
  "$REMOTE_REPO/artifacts/orchestration/rtx5090/$CAMPAIGN_ID" \
  "$LOCAL_ROOT/orchestration/$CAMPAIGN_ID" orchestration
for phase in phase1_rule_extraction phase2_cellular_automata phase3_memory; do
  verify_tree \
    "$REMOTE_REPO/artifacts/$phase/rtx5090/n=7/gemma-3-4b-it" \
    "$LOCAL_ROOT/$phase/gemma-3-4b-it" "$phase"
done
verify_tree \
  "$REMOTE_REPO/artifacts/cross_window_validation/rtx5090/n=7/gemma-3-4b-it/$AGGREGATE" \
  "$LOCAL_ROOT/cross_window_validation/gemma-3-4b-it/$AGGREGATE" aggregate

printf 'SYNC_PASS=%s\n' "$LOCAL_ROOT"

