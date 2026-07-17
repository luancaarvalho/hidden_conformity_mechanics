#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/artifacts/conda/runtime/bin/python}"
TIMESTAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/artifacts/phase3_memory/RTX5090_liaan}"
REPLAY1="$RESULTS_DIR/gemma4b_01_n30_neigh7_W3_batch_invariant_replay01_${TIMESTAMP}"
REPLAY2="$RESULTS_DIR/gemma4b_01_n30_neigh7_W3_batch_invariant_replay02_${TIMESTAMP}"
COMPARISON="$RESULTS_DIR/gemma4b_01_n30_neigh7_W3_batch_invariant_comparison_${TIMESTAMP}"
RUNNER="$REPO_ROOT/gradio_project/memory/run_gradio_gemma4b_01_matrix.py"
COMPARATOR="$REPO_ROOT/gradio_project/memory/compare_gradio_gemma4b_replays.py"

echo "REPLAY1=$REPLAY1"
echo "REPLAY2=$REPLAY2"
echo "COMPARISON=$COMPARISON"

cd "$REPO_ROOT"

"$PYTHON" "$RUNNER" \
  --output-root "$REPLAY1" \
  --agents 30 \
  --rounds 10 \
  --neighbors 7 \
  --memory-window 3 \
  --initial-majority 51 \
  --temperature 0 \
  --seeds 1-10 \
  --workers 8 \
  --server-batch-invariant

"$PYTHON" "$RUNNER" \
  --output-root "$REPLAY2" \
  --agents 30 \
  --rounds 10 \
  --neighbors 7 \
  --memory-window 3 \
  --initial-majority 51 \
  --temperature 0 \
  --seeds 1-10 \
  --workers 8 \
  --server-batch-invariant

set +e
"$PYTHON" "$COMPARATOR" \
  --replay1 "$REPLAY1" \
  --replay2 "$REPLAY2" \
  --output-dir "$COMPARISON"
COMPARISON_STATUS=$?
set -e

echo "COMPARISON_STATUS=$COMPARISON_STATUS"
exit "$COMPARISON_STATUS"
