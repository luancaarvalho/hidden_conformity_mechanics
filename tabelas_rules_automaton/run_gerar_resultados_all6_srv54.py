#!/usr/bin/env python3
"""
Wrapper: roda gerar_resultados_finais_en.py para todos os 6 modelos.
Output: resultados_finais_en_srv54_parallel_20260313/
Não modifica o script original.
"""
import importlib.util
import os
import sys
import re
import glob

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(os.path.dirname(HERE))           # repo root
RESULTS_SRV  = os.path.join(ROOT, "resultados_servidor")
AUTOMATON    = os.path.join(ROOT, "projeto_final", "experimentos_automatos", "resultados")
OUTPUT_DIR   = os.path.join(HERE, "resultados_finais_en_srv54_parallel_20260313")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# Load original script as module (without executing main)
# ──────────────────────────────────────────────────────────────────
orig_path = os.path.join(HERE, "gerar_resultados_finais_en.py")
spec = importlib.util.spec_from_file_location("gen_en", orig_path)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

# Override output dir in the module namespace
gen.OUTPUT_DIR = OUTPUT_DIR

# ──────────────────────────────────────────────────────────────────
# Model mapping: resultados_servidor_dir -> (display_name, automaton_slug)
# ──────────────────────────────────────────────────────────────────
MODELS = {
    "gemma4b":        ("Gemma 4B",  "gemma4b"),
    "gemma27b":       ("Gemma 27B", "gemma27b"),
    "llama8b":        ("Llama 8B",  "llama8b"),
    "llama70b":       ("Llama 70B", "llama70b"),
    "qwen3_4b":       ("Qwen 4B",   "qwen4b"),
    "qwen_qwen3_32b": ("Qwen 32B",  "qwen32b"),
}

TARGET_PCTS = {51, 55, 60, 65, 70}

TOKEN_MAP = {
    "kz": "k/z", "ab": "a/b", "01": "0/1", "pq": "p/q",
    "łþ": "ł/þ", "αβ": "α/β", "△○": "△/○", "⊕⊖": "⊕/⊖",
    "yesno": "no/yes", "noyes": "no/yes",
}

# ──────────────────────────────────────────────────────────────────
# 0. Collect ICI data for all 6 models
# ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("GENERATING FINAL RESULTS (English) — ALL 6 MODELS")
print(f"Output → {OUTPUT_DIR}")
print("=" * 70)

print("\n[0] Loading ICI data from resultados_servidor...")
all_data = []
for srv_dir, (model_name, _) in MODELS.items():
    base = os.path.join(RESULTS_SRV, srv_dir)
    if not os.path.exists(base):
        print(f"  [warn] Missing: {base}")
        continue
    for n in ("n7", "n9", "n11"):
        csv_dir = os.path.join(base, n, "csv", "basic")
        if not os.path.exists(csv_dir):
            print(f"  [warn] Missing csv/basic: {csv_dir}")
            continue
        d = gen.collect_ici_data(csv_dir, model_name)
        all_data.extend(d)
        print(f"  {model_name} ({n}): {len(d)} experiments")

print(f"  Total ICI rows: {len(all_data)}")

# ──────────────────────────────────────────────────────────────────
# 1. Basic ICI ranking tables
# ──────────────────────────────────────────────────────────────────
gen.generate_basic_ranking_tables(all_data, OUTPUT_DIR)

# ──────────────────────────────────────────────────────────────────
# 2. Extended tables — patch load_automaton_data for all 6 models
# ──────────────────────────────────────────────────────────────────
RX_AUTO = re.compile(
    r"/(?P<slug>[^/]+)/(?P<token>[^/]+)/n(?P<n>\d+)/(?P<fmt>[^/]+)/(?P<tag>[^/]+)/"
    r"(?P<agents>\d+)agents_(?P<pct>\d+)pct/resultados_individuais\.csv$"
)
SLUG_TO_MODEL = {v[1]: v[0] for v in MODELS.values()}   # e.g. "gemma4b" -> "Gemma 4B"


def load_automaton_data_all6():
    print("  Loading automaton data for all 6 models...")
    rows = []
    for srv_dir, (model_name, slug) in MODELS.items():
        pattern = os.path.join(AUTOMATON, slug, "**", "resultados_individuais.csv")
        paths = glob.glob(pattern, recursive=True)
        for p in paths:
            m = RX_AUTO.search(p.replace("\\", "/"))
            if not m:
                continue
            try:
                d = m.groupdict()
                majority_pct = int(d["pct"])
                if majority_pct not in TARGET_PCTS:
                    continue
                rdf = pd.read_csv(p)
                if "correct_consensus" in rdf.columns:
                    ok = gen._to_bool_series(rdf["correct_consensus"])
                elif "success" in rdf.columns:
                    ok = gen._to_bool_series(rdf["success"])
                else:
                    continue
                success_count = int(ok.sum())
                total_sims    = int(len(rdf))
                success_rate  = float(success_count / total_sims * 100) if total_sims > 0 else np.nan
                fmt_raw  = d.get("fmt", "")
                fmt_norm = "with_cot" if fmt_raw == "com_cot" else fmt_raw
                tok = TOKEN_MAP.get(d["token"], d["token"])
                rows.append({
                    "model":                model_name,
                    "token_type":           tok,
                    "n_window":             int(d["n"]),
                    "format":               fmt_norm,
                    "success_rate_percent": success_rate,
                    "success_count":        success_count,
                    "total_simulations":    total_sims,
                    "n_agents_config":      int(d["agents"]),
                    "majority_pct_config":  majority_pct,
                })
            except Exception:
                continue

    if not rows:
        print("  [error] No automaton CSV files found.")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} automaton records across {df['model'].nunique()} models.")
    return df


# Patch in module namespace so generate_extended_ranking_tables uses our version
gen.load_automaton_data = load_automaton_data_all6
gen.generate_extended_ranking_tables(all_data, OUTPUT_DIR)

# ──────────────────────────────────────────────────────────────────
# 3. Gemma 27B aggregated CSV (keeps original behaviour)
# ──────────────────────────────────────────────────────────────────
# Point gemma27b automaton dir to the new results
gen.GEMMA27B_AUTOMATON_RESULTS_DIR = os.path.join(AUTOMATON, "gemma27b")
gen.generate_gemma27b_aggregated_results(OUTPUT_DIR)

# ──────────────────────────────────────────────────────────────────
# 4. Token bias plots — all 6 models
# ──────────────────────────────────────────────────────────────────
print("\n[4] Generating token bias plots (all 6 models)...")
sources = {}
for srv_dir, (model_name, _) in MODELS.items():
    p = os.path.join(RESULTS_SRV, srv_dir)
    if os.path.exists(p):
        sources[model_name] = p

bias_results = gen.collect_token_bias_data(sources)
print(f"  Collected {len(bias_results)} token-bias data points.")

if bias_results:
    per_model = {}
    for r in bias_results:
        per_model.setdefault(r["model"], 0)
        per_model[r["model"]] += 1
    for m, c in sorted(per_model.items()):
        print(f"    {m}: {c}")

    gen.plot_all_tokens(bias_results, os.path.join(OUTPUT_DIR, "token_bias_all_tokens.png"))

    token_deltas = {}
    for r in bias_results:
        token_deltas.setdefault(r["token"], []).append(r["delta"])
    means = {t: np.mean(v) for t, v in token_deltas.items()}
    token_pos = max(means, key=means.get)
    token_neg = min(means, key=means.get)
    print(f"  Most biased: pos={token_pos} ({means[token_pos]:+.4f}), "
          f"neg={token_neg} ({means[token_neg]:+.4f})")
    gen.plot_focused_tokens(
        bias_results, token_pos, token_neg,
        os.path.join(OUTPUT_DIR, f"token_bias_focused_{token_pos}_{token_neg}.png")
    )
else:
    print("  [skip] No token-bias data.")

# ──────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────
images = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png"))
csvs   = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv"))
print("\n" + "=" * 70)
print(f"DONE.  {len(images)} PNGs + {len(csvs)} CSVs → {OUTPUT_DIR}")
print("=" * 70)
for f in images + csvs:
    print(f"  {f}")
