#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados"
AUTOMATON_DIR="$ROOT/projeto_final/experimentos_automatos"
RESULTS_SERVER="$ROOT/resultados_servidor"
LOG_DIR="$AUTOMATON_DIR/logs"
RUN_TAG="srv54_parallel_20260313"
RUN_LOG="$LOG_DIR/run_all6_models_parallel_srv54_20260313.log"
FINAL_EN_OUT="$ROOT/projeto_final/tabelas_rules_automaton/resultados_finais_en_${RUN_TAG}"

source /Users/luancarvalho/opt/miniconda3/etc/profile.d/conda.sh
conda activate luan_conformidade
mkdir -p "$LOG_DIR"
cd "$AUTOMATON_DIR"

{
  echo "[$(date '+%F %T')] START run_all6_models_parallel_${RUN_TAG}"
  echo "ROOT=$ROOT"
  echo "AUTOMATON_DIR=$AUTOMATON_DIR"
  echo "RESULTS_SERVER=$RESULTS_SERVER"
  echo "FINAL_EN_OUT=$FINAL_EN_OUT"
  echo "PARALLELISM=1 worker per model (6 workers)"
  echo
} | tee "$RUN_LOG"

PYTHONPATH="$ROOT:$ROOT/experimentos_automatos:$ROOT/projeto_final/experimentos_automatos" \
  python -u "$AUTOMATON_DIR/run_all6_parallel_runner.py" | tee -a "$RUN_LOG"

PYTHONPATH="$ROOT" python -u - <<'PY' | tee -a "$RUN_LOG"
import glob
import importlib.util
import os
import re
import numpy as np
import pandas as pd

ROOT = "/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados"
RUN_TAG = "srv54_parallel_20260313"
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

print("\n[2] Extended ranking tables with efficiencies (from this run tag)...")
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
    m = rx.search(p.replace("\\", "/"))
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

orig_loader = grf.load_automaton_data
grf.load_automaton_data = lambda: automaton_df.copy()
try:
    grf.generate_extended_ranking_tables(all_data, OUTPUT_DIR)
finally:
    grf.load_automaton_data = orig_loader

print("\n[3] Token bias plots for all 6 models...")
tb_results = grf.collect_token_bias_data(sources_ici)
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

pd.DataFrame(all_data).to_csv(os.path.join(OUTPUT_DIR, "ici_all_models_raw.csv"), index=False)
if not automaton_df.empty:
    automaton_df.to_csv(os.path.join(OUTPUT_DIR, "efficiency_automaton_raw.csv"), index=False)

images = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])
print(f"DONE_FINAL_EN images={len(images)} out={OUTPUT_DIR}")
PY

echo "[$(date '+%F %T')] END run_all6_models_parallel_${RUN_TAG}" | tee -a "$RUN_LOG"
echo "RUN_LOG=$RUN_LOG"
echo "FINAL_EN_OUT=$FINAL_EN_OUT"
