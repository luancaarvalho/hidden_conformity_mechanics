#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?usage: $0 RUN_ID [LOCAL_RESULTS_ROOT]}"
LOCAL_RESULTS_ROOT="${2:-/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/resultados_simulacoes_conformidade}"
REMOTE_REPO="${REMOTE_REPO:-/home/liaan/Documentos/Luan/hidden_conformity_mechanics}"
DEST_ROOT="$LOCAL_RESULTS_ROOT/rtx5090"
PROOF_ROOT="$DEST_ROOT/n=7/sync_proof/$RUN_ID"
VARIANTS=(v9_lista_completa_meio_parity_01 v21_zero_shot_cot_parity_01)

mkdir -p "$PROOF_ROOT"

sync_tree() {
  local label="$1"
  local remote_path="$2"
  local local_path="$3"
  if ! ssh rtx5090 "test -d '$remote_path'"; then
    printf 'SKIPPED missing %s\n' "$remote_path" | tee -a "$PROOF_ROOT/sync_status.txt"
    return
  fi
  mkdir -p "$local_path"
  rsync -a --delete "rtx5090:$remote_path/" "$local_path/"
  ssh rtx5090 "cd '$remote_path' && find . -type f -print0 | sort -z | xargs -0 sha256sum" \
    > "$PROOF_ROOT/${label}.remote.sha256"
  (cd "$local_path" && find . -type f -print0 | sort -z | xargs -0 shasum -a 256) \
    > "$PROOF_ROOT/${label}.local.sha256"
  diff -u "$PROOF_ROOT/${label}.remote.sha256" "$PROOF_ROOT/${label}.local.sha256" \
    > "$PROOF_ROOT/${label}.diff"
  printf 'PASS %s %s\n' "$label" "$local_path" | tee -a "$PROOF_ROOT/sync_status.txt"
}

for n in 3 5 7; do
  for variant in "${VARIANTS[@]}"; do
    source="$REMOTE_REPO/artifacts/phase1_rule_extraction/rtx5090/n=$n/gemma-3-4b-it/tokens=0-1/$variant/$RUN_ID"
    destination="$DEST_ROOT/n=$n/phase1/gemma-3-4b-it/tokens=0-1/$variant/$RUN_ID"
    sync_tree "phase1_n${n}_${variant}" "$source" "$destination"
  done
done

for variant in "${VARIANTS[@]}"; do
  source="$REMOTE_REPO/artifacts/phase2_cellular_automata/rtx5090/n=7/gemma-3-4b-it/tokens=0-1/$variant/$RUN_ID"
  destination="$DEST_ROOT/n=7/phase2/gemma-3-4b-it/tokens=0-1/$variant/$RUN_ID"
  sync_tree "phase2_n7_${variant}" "$source" "$destination"

  source="$REMOTE_REPO/artifacts/phase3_memory/rtx5090/n=7/gemma-3-4b-it/tokens=0-1/W=0/$variant/$RUN_ID"
  destination="$DEST_ROOT/n=7/phase3/gemma-3-4b-it/tokens=0-1/W=0/$variant/$RUN_ID"
  sync_tree "phase3_n7_w0_${variant}" "$source" "$destination"
done

source="$REMOTE_REPO/artifacts/cross_phase_validation/rtx5090/n=7/gemma-3-4b-it/tokens=0-1/$RUN_ID"
destination="$DEST_ROOT/n=7/cross_phase/gemma-3-4b-it/tokens=0-1/$RUN_ID"
sync_tree cross_phase_n7 "$source" "$destination"

echo "local_root=$DEST_ROOT"
echo "proof=$PROOF_ROOT"
