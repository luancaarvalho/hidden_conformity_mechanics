#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
LEGACY_MEMORY_ROOT="${LEGACY_MEMORY_ROOT:-/home/liaan/Documentos/Luan/temp_vllm/gradio_project/results}"

mkdir -p "$REPO_ROOT/artifacts/phase1_rule_extraction/RTX5090_liaan"
mkdir -p "$REPO_ROOT/artifacts/phase2_cellular_automata/RTX5090_liaan"
mkdir -p "$REPO_ROOT/artifacts/phase3_memory"
for n in 3 5 7; do
  mkdir -p "$REPO_ROOT/artifacts/phase1_rule_extraction/rtx5090/n=$n"
done
mkdir -p "$REPO_ROOT/artifacts/phase2_cellular_automata/rtx5090/n=7"
mkdir -p "$REPO_ROOT/artifacts/phase3_memory/rtx5090/n=7"
mkdir -p "$REPO_ROOT/artifacts/cross_phase_validation/rtx5090/n=7"
mkdir -p "$REPO_ROOT/artifacts/work/rtx5090"

LINK="$REPO_ROOT/artifacts/phase3_memory/RTX5090_liaan"
if [ ! -e "$LINK" ] && [ ! -L "$LINK" ]; then
  ln -s "$LEGACY_MEMORY_ROOT" "$LINK"
fi

test -d "$LINK"
printf 'phase1=%s\nphase2=%s\nphase3=%s\n' \
  "$REPO_ROOT/artifacts/phase1_rule_extraction/RTX5090_liaan" \
  "$REPO_ROOT/artifacts/phase2_cellular_automata/RTX5090_liaan" \
  "$(readlink "$LINK")"
