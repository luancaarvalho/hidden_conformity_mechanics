#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados"
AUTOMATON_DIR="$ROOT/projeto_final/experimentos_automatos"
SRC_BASE="$ROOT/projeto_final/extract_rules/A100/gemma27b"
LOG_DIR="$AUTOMATON_DIR/logs"
LOG_FILE="$LOG_DIR/rerun_gemma27b_kz_only.log"

source /Users/luancarvalho/opt/miniconda3/etc/profile.d/conda.sh
conda activate luan_conformidade

cd "$AUTOMATON_DIR"
mkdir -p "$LOG_DIR"

# Replace previous output tree by moving it aside.
TS="$(date +%Y%m%d_%H%M%S)"
if [ -d "resultados/gemma27b" ]; then
  mv "resultados/gemma27b" "resultados/gemma27b_backup_${TS}"
fi
mkdir -p "resultados/gemma27b"

PYTHONPATH="$ROOT:$ROOT/experimentos_automatos" python -u - <<'PY' > "$LOG_FILE" 2>&1
import glob
import os
import re
import time
import traceback
from run_automaton_numba import run_simulations_from_csv

base = "/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/extract_rules/A100/gemma27b"
all_csvs = sorted(glob.glob(os.path.join(base, "n*", "csv", "basic", "*.csv")))

sel = []
rx_v9 = re.compile(r"/dados_combinados_v9_lista_completa_meio_kz_n(7|9|11)_exp\d+_iter\d+_.*\.csv$")
rx_v21 = re.compile(r"/dados_combinados_v21_zero_shot_cot_n(7|9|11)_exp\d+_iter\d+_.*\.csv$")
for p in all_csvs:
    if rx_v9.search(p) or rx_v21.search(p):
        sel.append(p)
sel = sorted(sel)

agents_list = [30, 60, 90, 120]
print(f"ALL_CSV_COUNT={len(all_csvs)}")
print(f"SELECTED_CSV_COUNT={len(sel)}")
for x in sel:
    print(f"SELECTED={os.path.basename(x)}")
if len(sel) != 6:
    raise SystemExit(f"Esperado 6 CSVs kz (v9/v21,n7/n9/n11), encontrado {len(sel)}")

total = len(sel) * len(agents_list)
idx = 0
failures = []
start_all = time.time()

for csv_path in sel:
    for agents in agents_list:
        idx += 1
        t0 = time.time()
        print("=" * 100, flush=True)
        print(f"[{idx}/{total}] START csv={os.path.basename(csv_path)} agents={agents}", flush=True)
        try:
            run_simulations_from_csv(
                csv_path=csv_path,
                n_simulations=50,
                n_agents=agents,
                max_iterations=2 * agents,
                generate_plot=True,
                force_extract=False,
                majority_ratio=0.51,
                initial_distribution_mode="auto",
                source_tag="A100",
            )
            dt = time.time() - t0
            print(f"[{idx}/{total}] DONE  csv={os.path.basename(csv_path)} agents={agents} sec={dt:.2f}", flush=True)
        except Exception as e:
            dt = time.time() - t0
            failures.append((csv_path, agents, repr(e)))
            print(f"[{idx}/{total}] FAIL  csv={os.path.basename(csv_path)} agents={agents} sec={dt:.2f} err={e!r}", flush=True)
            traceback.print_exc()

elapsed = time.time() - start_all
print("=" * 100)
print(f"FINISHED total={total} failures={len(failures)} elapsed_sec={elapsed:.2f}")
if failures:
    for c, a, err in failures:
        print(f"FAIL_ITEM csv={c} agents={a} err={err}")
    raise SystemExit(1)
PY

echo "RUNNING_LOG=$LOG_FILE"
tail -n 40 "$LOG_FILE"
