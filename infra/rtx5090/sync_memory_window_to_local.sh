#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?usage: $0 RUN_ID MEMORY_WINDOW [LOCAL_RESULTS_ROOT]}"
MEMORY_WINDOW="${2:?usage: $0 RUN_ID MEMORY_WINDOW [LOCAL_RESULTS_ROOT]}"
LOCAL_RESULTS_ROOT="${3:-/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/resultados_simulacoes_conformidade}"
REMOTE_REPO="${REMOTE_REPO:-/home/liaan/Documentos/Luan/hidden_conformity_mechanics}"
DEST_ROOT="$LOCAL_RESULTS_ROOT/rtx5090"
PROOF_ROOT="$DEST_ROOT/n=7/sync_proof/$RUN_ID"
VARIANTS=(v9_lista_completa_meio_parity_01 v21_zero_shot_cot_parity_01)

mkdir -p "$PROOF_ROOT"
: > "$PROOF_ROOT/sync_status.txt"

sync_tree() {
  local label="$1"
  local remote_path="$2"
  local local_path="$3"
  test ! -e "$local_path"
  ssh rtx5090 "test -d '$remote_path'"
  mkdir -p "$local_path"
  rsync -a "rtx5090:$remote_path/" "$local_path/"
  ssh rtx5090 "cd '$remote_path' && find . -type f -print0 | sort -z | xargs -0 sha256sum" \
    > "$PROOF_ROOT/${label}.remote.sha256"
  (cd "$local_path" && find . -type f -print0 | sort -z | xargs -0 shasum -a 256) \
    > "$PROOF_ROOT/${label}.local.sha256"
  diff -u "$PROOF_ROOT/${label}.remote.sha256" "$PROOF_ROOT/${label}.local.sha256" \
    > "$PROOF_ROOT/${label}.diff"
  printf 'PASS %s %s\n' "$label" "$local_path" | tee -a "$PROOF_ROOT/sync_status.txt"
}

for variant in "${VARIANTS[@]}"; do
  source="$REMOTE_REPO/artifacts/phase3_memory/rtx5090/n=7/gemma-3-4b-it/tokens=0-1/W=$MEMORY_WINDOW/$variant/$RUN_ID"
  destination="$DEST_ROOT/n=7/phase3/gemma-3-4b-it/tokens=0-1/W=$MEMORY_WINDOW/$variant/$RUN_ID"
  sync_tree "phase3_n7_w${MEMORY_WINDOW}_${variant}" "$source" "$destination"
done

source="$REMOTE_REPO/artifacts/cross_phase_validation/rtx5090/n=7/gemma-3-4b-it/tokens=0-1/$RUN_ID"
destination="$DEST_ROOT/n=7/cross_phase/gemma-3-4b-it/tokens=0-1/$RUN_ID"
sync_tree cross_phase_n7 "$source" "$destination"

echo "local_root=$DEST_ROOT"
echo "proof=$PROOF_ROOT"
