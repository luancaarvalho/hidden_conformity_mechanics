#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados"
AUTOMATON_DIR="$ROOT/projeto_final/experimentos_automatos"
RESULTS_SERVER="$ROOT/resultados_servidor"
LOG_DIR="$AUTOMATON_DIR/logs"
RUN_TAG="srv54_20260312"
RUN_LOG="$LOG_DIR/run_all6_models_${RUN_TAG}.log"
FINAL_EN_OUT="$ROOT/projeto_final/tabelas_rules_automaton/resultados_finais_en_${RUN_TAG}"

source /Users/luancarvalho/opt/miniconda3/etc/profile.d/conda.sh
conda activate luan_conformidade

mkdir -p "$LOG_DIR"
cd "$AUTOMATON_DIR"

{
  echo "[$(date '+%F %T')] START run_all6_models_${RUN_TAG}"
  echo "ROOT=$ROOT"
  echo "AUTOMATON_DIR=$AUTOMATON_DIR"
  echo "RESULTS_SERVER=$RESULTS_SERVER"
  echo "FINAL_EN_OUT=$FINAL_EN_OUT"
  echo
} | tee "$RUN_LOG"

# 1) Limpa resultados anteriores para evitar mistura de runs
python - <<'PY' | tee -a "$RUN_LOG"
import os
import shutil

root = "/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados"
base = os.path.join(root, "projeto_final", "experimentos_automatos", "resultados")
targets = ["gemma4b", "gemma27b", "llama8b", "llama70b", "qwen4b", "qwen32b"]

print("=== CLEAN OLD AUTOMATON RESULTS ===")
for m in targets:
    p = os.path.join(base, m)
    if os.path.isdir(p):
        shutil.rmtree(p)
        print(f"REMOVED: {p}")
    else:
        print(f"SKIP (not found): {p}")
PY

# limpa pasta de saída final EN da rodada anterior com a mesma tag
python - <<'PY' | tee -a "$RUN_LOG"
import os
import shutil

out_dir = "/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/tabelas_rules_automaton/resultados_finais_en_srv54_20260312"
if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
    print(f"REMOVED: {out_dir}")
else:
    print(f"SKIP (not found): {out_dir}")
PY

# 2) Roda autômato (mesmos parâmetros do gemma27b com plots)
PYTHONPATH="$ROOT:$ROOT/experimentos_automatos:$ROOT/projeto_final/experimentos_automatos" python -u - <<'PY' | tee -a "$RUN_LOG"
import glob
import os
import re
import time
import traceback

import run_automaton_numba as ra

ROOT = "/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados"
RESULTS_SERVER = os.path.join(ROOT, "resultados_servidor")
RUN_TAG = "srv54_20260312"

MODEL_SOURCES = {
    "gemma4b": ("gemma4b", "Gemma 4B"),
    "gemma27b": ("gemma27b", "Gemma 27B"),
    "llama8b": ("llama8b", "Llama 8B"),
    "llama70b": ("llama70b", "Llama 70B"),
    "qwen3_4b": ("qwen4b", "Qwen 4B"),
    "qwen_qwen3_32b": ("qwen32b", "Qwen 32B"),
}

AGENTS_LIST = [30, 60, 90, 120]
N_SIMULATIONS = 50
MAJORITY_RATIOS = [0.51, 0.55, 0.60, 0.65, 0.70]

original_parse = ra.parse_csv_metadata

def patched_parse(csv_path: str):
    meta = original_parse(csv_path)
    p = csv_path.replace("\\\\", "/")
    if "/resultados_servidor/gemma4b/" in p:
        meta["model"] = "gemma4b"
    elif "/resultados_servidor/gemma27b/" in p:
        meta["model"] = "gemma27b"
    elif "/resultados_servidor/llama8b/" in p:
        meta["model"] = "llama8b"
    elif "/resultados_servidor/llama70b/" in p:
        meta["model"] = "llama70b"
    elif "/resultados_servidor/qwen3_4b/" in p:
        meta["model"] = "qwen4b"
    elif "/resultados_servidor/qwen_qwen3_32b/" in p:
        meta["model"] = "qwen32b"
    return meta

ra.parse_csv_metadata = patched_parse

def n_sort_key(path: str):
    m = re.search(r"/n(\d+)/csv/basic/", path.replace("\\\\", "/"))
    n = int(m.group(1)) if m else 999
    order = {7: 0, 9: 1, 11: 2}.get(n, 999)
    return (order, os.path.basename(path))

all_jobs = []
for src_folder, (slug, nice) in MODEL_SOURCES.items():
    csvs = []
    for n in ("n7", "n9", "n11"):
        pattern = os.path.join(RESULTS_SERVER, src_folder, n, "csv", "basic", "*.csv")
        csvs.extend(glob.glob(pattern))
    csvs = sorted(csvs, key=n_sort_key)
    print(f"{nice}: csv_count={len(csvs)}")
    if len(csvs) != 54:
        raise SystemExit(f"[FATAL] {nice}: esperado 54 CSVs basic, encontrado {len(csvs)}")
    for csv_path in csvs:
        for agents in AGENTS_LIST:
            for majority_ratio in MAJORITY_RATIOS:
                all_jobs.append((nice, csv_path, agents, majority_ratio))

total = len(all_jobs)
print(f"TOTAL_JOBS={total} (models={len(MODEL_SOURCES)}, csvs/model=54, agents={AGENTS_LIST})")
print(f"SETTINGS: n_simulations={N_SIMULATIONS}, majority_ratios={MAJORITY_RATIOS}, max_iterations=2N, plots=True")

failures = []
start_all = time.time()

for idx, (model_name, csv_path, agents, majority_ratio) in enumerate(all_jobs, 1):
    t0 = time.time()
    csv_name = os.path.basename(csv_path)
    print("=" * 120, flush=True)
    print(
        f"[{idx}/{total}] START model={model_name} csv={csv_name} agents={agents} majority={majority_ratio:.2f}",
        flush=True,
    )
    try:
        ra.run_simulations_from_csv(
            csv_path=csv_path,
            n_simulations=N_SIMULATIONS,
            n_agents=agents,
            max_iterations=2 * agents,
            generate_plot=True,
            force_extract=False,
            majority_ratio=majority_ratio,
            initial_distribution_mode="auto",
            source_tag=RUN_TAG,
        )
        dt = time.time() - t0
        print(
            f"[{idx}/{total}] DONE model={model_name} csv={csv_name} agents={agents} majority={majority_ratio:.2f} sec={dt:.2f}",
            flush=True,
        )
    except Exception as e:
        dt = time.time() - t0
        failures.append((model_name, csv_path, agents, majority_ratio, repr(e)))
        print(
            f"[{idx}/{total}] FAIL model={model_name} csv={csv_name} agents={agents} majority={majority_ratio:.2f} sec={dt:.2f} err={e!r}",
            flush=True,
        )
        traceback.print_exc()

elapsed = time.time() - start_all
print("=" * 120)
print(f"FINISHED_AUTOMATON total={total} failures={len(failures)} elapsed_sec={elapsed:.2f}")
if failures:
    for model_name, csv_path, agents, majority_ratio, err in failures:
        print(
            f"FAIL_ITEM model={model_name} csv={csv_path} agents={agents} majority={majority_ratio:.2f} err={err}"
        )
    raise SystemExit(1)
PY

# 3) Gera resultados finais EN para os 6 modelos sem editar o script original
PYTHONPATH="$ROOT" python -u - <<'PY' | tee -a "$RUN_LOG"
import glob
import importlib.util
import os
import re
import numpy as np
import pandas as pd

ROOT = "/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados"
RUN_TAG = "srv54_20260312"
OUTPUT_DIR = os.path.join(ROOT, "projeto_final", "tabelas_rules_automaton", f"resultados_finais_en_{RUN_TAG}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

module_path = os.path.join(ROOT, "projeto_final", "tabelas_rules_automaton", "gerar_resultados_finais_en.py")
spec = importlib.util.spec_from_file_location("grf_en", module_path)
grf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grf)

sources_ici = {
    "Gemma 4B": os.path.join(ROOT, "resultados_servidor", "gemma4b"),
    "Gemma 27B": os.path.join(ROOT, "resultados_servidor", "gemma27b"),
    "Llama 8B": os.path.join(ROOT, "resultados_servidor", "llama8b"),
    "Llama 70B": os.path.join(ROOT, "resultados_servidor", "llama70b"),
    "Qwen 4B": os.path.join(ROOT, "resultados_servidor", "qwen3_4b"),
    "Qwen 32B": os.path.join(ROOT, "resultados_servidor", "qwen_qwen3_32b"),
}

print("[0] Loading ICI data (6 models, resultados_servidor)...")
all_data = []
for name, base in sources_ici.items():
    for n in ("n7", "n9", "n11"):
        d = grf.collect_ici_data(os.path.join(base, n, "csv", "basic"), name)
        all_data.extend(d)
        print(f"  {name} ({n}): {len(d)} experiments")
print(f"  Total: {len(all_data)} experiments")

print("\n[1] Basic ranking tables...")
grf.generate_basic_ranking_tables(all_data, OUTPUT_DIR)

print("\n[2] Extended ranking tables with efficiencies (using automaton outputs from this run tag)...")
model_slug_to_name = {
    "gemma4b": "Gemma 4B",
    "gemma27b": "Gemma 27B",
    "llama8b": "Llama 8B",
    "llama70b": "Llama 70B",
    "qwen4b": "Qwen 4B",
    "qwen32b": "Qwen 32B",
}
token_map = {
    "kz": "k/z", "ab": "a/b", "01": "0/1", "pq": "p/q",
    "łþ": "ł/þ", "αβ": "α/β", "△○": "△/○", "⊕⊖": "⊕/⊖",
    "yesno": "no/yes", "noyes": "no/yes",
}

rx = re.compile(
    r"/resultados/(?P<model>[^/]+)/(?P<token>[^/]+)/n(?P<n>\d+)/(?P<fmt>[^/]+)/(?P<tag>[^/]+)/"
    r"(?P<agents>\d+)agents_(?P<pct>\d+)pct/resultados_individuais\.csv$"
)

rows = []
pattern = os.path.join(ROOT, "projeto_final", "experimentos_automatos", "resultados", "**", "resultados_individuais.csv")
for p in glob.glob(pattern, recursive=True):
    m = rx.search(p.replace("\\\\", "/"))
    if not m:
        continue
    d = m.groupdict()
    if d["tag"] != RUN_TAG:
        continue
    model_name = model_slug_to_name.get(d["model"])
    if model_name is None:
        continue
    token_type = token_map.get(d["token"], d["token"])
    n_window = int(d["n"])
    n_agents = int(d["agents"])
    majority_pct = int(d["pct"])

    rdf = pd.read_csv(p)
    if "correct_consensus" in rdf.columns:
        ok = grf._to_bool_series(rdf["correct_consensus"])
    elif "success" in rdf.columns:
        ok = grf._to_bool_series(rdf["success"])
    else:
        continue

    success_count = int(ok.sum())
    total_simulations = int(len(rdf))
    success_rate = float((success_count / total_simulations) * 100.0) if total_simulations > 0 else np.nan
    fmt_raw = d["fmt"]
    fmt_norm = "with_cot" if fmt_raw == "com_cot" else fmt_raw

    rows.append({
        "model": model_name,
        "token_type": token_type,
        "n_window": n_window,
        "format": fmt_norm,
        "success_rate_percent": success_rate,
        "success_count": success_count,
        "total_simulations": total_simulations,
        "n_agents_config": n_agents,
        "majority_pct_config": majority_pct,
    })

automaton_df = pd.DataFrame(rows) if rows else pd.DataFrame()
print(f"  Automaton records loaded for efficiency: {len(automaton_df)}")
if automaton_df.empty:
    print("  [warn] No automaton records found for efficiency.")

orig_loader = grf.load_automaton_data
grf.load_automaton_data = lambda: automaton_df.copy()
try:
    grf.generate_extended_ranking_tables(all_data, OUTPUT_DIR)
finally:
    grf.load_automaton_data = orig_loader

print("\n[3] Token bias plots for all 6 models...")
tb_sources = {
    "Gemma 4B": os.path.join(ROOT, "resultados_servidor", "gemma4b"),
    "Gemma 27B": os.path.join(ROOT, "resultados_servidor", "gemma27b"),
    "Llama 8B": os.path.join(ROOT, "resultados_servidor", "llama8b"),
    "Llama 70B": os.path.join(ROOT, "resultados_servidor", "llama70b"),
    "Qwen 4B": os.path.join(ROOT, "resultados_servidor", "qwen3_4b"),
    "Qwen 32B": os.path.join(ROOT, "resultados_servidor", "qwen_qwen3_32b"),
}
tb_results = grf.collect_token_bias_data(tb_sources)
print(f"  Token-bias points: {len(tb_results)}")
if tb_results:
    grf.plot_all_tokens(tb_results, os.path.join(OUTPUT_DIR, "token_bias_all_tokens.png"))
    token_deltas = {}
    for r in tb_results:
        token_deltas.setdefault(r["token"], []).append(r["delta"])
    means = {t: float(np.mean(v)) for t, v in token_deltas.items()}
    token_pos = max(means, key=means.get)
    token_neg = min(means, key=means.get)
    grf.plot_focused_tokens(
        tb_results,
        token_pos,
        token_neg,
        os.path.join(OUTPUT_DIR, f"token_bias_focused_{token_pos}_{token_neg}.png"),
    )

all_data_csv = os.path.join(OUTPUT_DIR, "ici_all_models_raw.csv")
pd.DataFrame(all_data).to_csv(all_data_csv, index=False)
print(f"  Saved: {all_data_csv}")
if not automaton_df.empty:
    eff_csv = os.path.join(OUTPUT_DIR, "efficiency_automaton_raw.csv")
    automaton_df.to_csv(eff_csv, index=False)
    print(f"  Saved: {eff_csv}")

images = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])
print("\nDONE.")
print(f"OUTPUT_DIR={OUTPUT_DIR}")
print(f"IMAGES={len(images)}")
for img in images:
    print(f"  {img}")
PY

echo "[$(date '+%F %T')] END run_all6_models_${RUN_TAG}" | tee -a "$RUN_LOG"
echo "RUN_LOG=$RUN_LOG"
echo "FINAL_EN_OUT=$FINAL_EN_OUT"
