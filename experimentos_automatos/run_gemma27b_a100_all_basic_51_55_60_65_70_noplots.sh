#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados"
AUTOMATON_DIR="$ROOT/projeto_final/experimentos_automatos"
LOG_DIR="$AUTOMATON_DIR/logs"
LOG_FILE="$LOG_DIR/run_gemma27b_all_basic_51_55_60_65_70_noplots.log"
SRC_BASE="$ROOT/projeto_final/extract_rules/A100/gemma27b"

source /Users/luancarvalho/opt/miniconda3/etc/profile.d/conda.sh
conda activate luan_conformidade

cd "$AUTOMATON_DIR"
mkdir -p "$LOG_DIR"

# Start from clean output tree while preserving previous runs.
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
csvs = glob.glob(os.path.join(base, "n*", "csv", "basic", "*.csv"))

def n_sort_key(path: str):
    m = re.search(r"/n(\d+)/csv/basic/", path)
    n = int(m.group(1)) if m else 999
    order = {7: 0, 9: 1, 11: 2}.get(n, 999)
    return (order, os.path.basename(path))

csvs = sorted(csvs, key=n_sort_key)
agents_list = [30, 60, 90, 120]
majority_list = [0.51, 0.55, 0.60, 0.65, 0.70]

print(f"SELECTED_CSV_COUNT={len(csvs)}")
if len(csvs) != 54:
    raise SystemExit(f"Esperado 54 CSVs basic, encontrado {len(csvs)}")

total = len(csvs) * len(agents_list) * len(majority_list)
idx = 0
failures = []
start_all = time.time()

print(f"AGENTS={agents_list}")
print(f"MAJORITY_LIST={majority_list}")
print(f"TOTAL_JOBS={total}")

for csv_path in csvs:
    for agents in agents_list:
        for majority_ratio in majority_list:
            idx += 1
            t0 = time.time()
            print("=" * 110, flush=True)
            print(
                f"[{idx}/{total}] START csv={os.path.basename(csv_path)} agents={agents} majority={majority_ratio:.2f}",
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
                    majority_ratio=majority_ratio,
                    initial_distribution_mode="auto",
                    source_tag="A100",
                )
                dt = time.time() - t0
                print(
                    f"[{idx}/{total}] DONE  csv={os.path.basename(csv_path)} agents={agents} majority={majority_ratio:.2f} sec={dt:.2f}",
                    flush=True,
                )
            except Exception as e:
                dt = time.time() - t0
                failures.append((csv_path, agents, majority_ratio, repr(e)))
                print(
                    f"[{idx}/{total}] FAIL  csv={os.path.basename(csv_path)} agents={agents} majority={majority_ratio:.2f} sec={dt:.2f} err={e!r}",
                    flush=True,
                )
                traceback.print_exc()

elapsed = time.time() - start_all
print("=" * 110)
print(f"FINISHED total={total} failures={len(failures)} elapsed_sec={elapsed:.2f}")
if failures:
    for c, a, mr, err in failures:
        print(f"FAIL_ITEM csv={c} agents={a} majority={mr:.2f} err={err}")
    raise SystemExit(1)
PY

echo "RUNNING_LOG=$LOG_FILE"
tail -n 40 "$LOG_FILE"
