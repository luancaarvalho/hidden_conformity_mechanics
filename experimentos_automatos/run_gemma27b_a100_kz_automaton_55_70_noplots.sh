#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados"
AUTOMATON_DIR="$ROOT/projeto_final/experimentos_automatos"
LOG_DIR="$AUTOMATON_DIR/logs"
SRC_BASE="$ROOT/projeto_final/extract_rules/A100/gemma27b"
LOG_FILE="$LOG_DIR/run_gemma27b_kz_v9_v21_55_70_noplots.log"

source /Users/luancarvalho/opt/miniconda3/etc/profile.d/conda.sh
conda activate luan_conformidade

cd "$AUTOMATON_DIR"
mkdir -p "$LOG_DIR"

PYTHONPATH="$ROOT:$ROOT/experimentos_automatos" python -u - <<'PY' > "$LOG_FILE" 2>&1
import glob
import os
import re
import time
import traceback

from run_automaton_numba import run_simulations_from_csv

base = "/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/extract_rules/A100/gemma27b"
all_csvs = sorted(glob.glob(os.path.join(base, "n*", "csv", "basic", "*.csv")))

rx_v9 = re.compile(r"/dados_combinados_v9_lista_completa_meio_kz_n(7|9|11)_exp\d+_iter\d+_.*\.csv$")
rx_v21 = re.compile(r"/dados_combinados_v21_zero_shot_cot_n(7|9|11)_exp\d+_iter\d+_.*\.csv$")
selected = [p for p in all_csvs if rx_v9.search(p) or rx_v21.search(p)]

def _sort_key(path: str):
    m = re.search(r"/n(\d+)/csv/basic/", path)
    n_val = int(m.group(1)) if m else 999
    n_ord = {7: 0, 9: 1, 11: 2}.get(n_val, 999)
    prompt_ord = 0 if "v9_lista_completa_meio_kz" in path else 1
    return (n_ord, prompt_ord, os.path.basename(path))

selected = sorted(selected, key=_sort_key)

if len(selected) != 6:
    raise SystemExit(f"Esperado 6 CSVs (v9/v21, n7/n9/n11), encontrado {len(selected)}")

agents_list = [30, 60, 90, 120]
majority_ratios = [0.55, 0.60, 0.65, 0.70]

total = len(selected) * len(agents_list) * len(majority_ratios)
idx = 0
failures = []
start_all = time.time()

print(f"ALL_CSV_COUNT={len(all_csvs)}")
print(f"SELECTED_CSV_COUNT={len(selected)}")
for p in selected:
    print(f"SELECTED={os.path.basename(p)}")
print(f"AGENTS={agents_list}")
print(f"MAJORITY_RATIOS={majority_ratios}")
print(f"TOTAL_JOBS={total}")

for csv_path in selected:
    for agents in agents_list:
        for ratio in majority_ratios:
            idx += 1
            t0 = time.time()
            print("=" * 110, flush=True)
            print(
                f"[{idx}/{total}] START csv={os.path.basename(csv_path)} agents={agents} ratio={ratio:.2f}",
                flush=True,
            )
            try:
                run_simulations_from_csv(
                    csv_path=csv_path,
                    n_simulations=50,
                    n_agents=agents,
                    max_iterations=2 * agents,
                    generate_plot=False,
                    force_extract=False,
                    majority_ratio=ratio,
                    initial_distribution_mode="auto",
                    source_tag="A100",
                )
                dt = time.time() - t0
                print(
                    f"[{idx}/{total}] DONE  csv={os.path.basename(csv_path)} agents={agents} ratio={ratio:.2f} sec={dt:.2f}",
                    flush=True,
                )
            except Exception as e:
                dt = time.time() - t0
                failures.append((csv_path, agents, ratio, repr(e)))
                print(
                    f"[{idx}/{total}] FAIL  csv={os.path.basename(csv_path)} agents={agents} ratio={ratio:.2f} sec={dt:.2f} err={e!r}",
                    flush=True,
                )
                traceback.print_exc()

elapsed = time.time() - start_all
print("=" * 110)
print(f"FINISHED total={total} failures={len(failures)} elapsed_sec={elapsed:.2f}")
if failures:
    for c, a, r, err in failures:
        print(f"FAIL_ITEM csv={c} agents={a} ratio={r:.2f} err={err}")
    raise SystemExit(1)
PY

echo "RUNNING_LOG=$LOG_FILE"
tail -n 40 "$LOG_FILE"
