#!/usr/bin/env python3
import csv
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from run_automaton_numba import run_simulations_from_csv


ROOT = Path("/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados")
PROJECT_FINAL = ROOT / "projeto_final"
EXTRACT_A100 = PROJECT_FINAL / "extract_rules" / "A100"
OUTPUT_DIR = PROJECT_FINAL / "resultados_qualidade_execucao_simultanea_automaton_20260308"

AGENTS = [30, 60, 90, 120]
MAJORITIES = [0.51, 0.55, 0.60, 0.65, 0.70]
N_SIMULATIONS = 50
MAX_WORKERS = min(12, max(2, os.cpu_count() or 4))


def list_basic_csvs(model_slug: str) -> List[Path]:
    files = list((EXTRACT_A100 / model_slug).glob("n*/csv/basic/*.csv"))

    def sort_key(p: Path):
        m_n = re.search(r"/n(\d+)/csv/basic/", p.as_posix())
        n = int(m_n.group(1)) if m_n else 999
        m_exp = re.search(r"exp(\d+)_", p.name)
        exp = int(m_exp.group(1)) if m_exp else 9999
        return (n, exp, p.name)

    return sorted(files, key=sort_key)


def merge_basic_csvs(model_slug: str, csv_files: List[Path], out_path: Path) -> None:
    rows: List[Dict] = []
    for p in csv_files:
        df = pd.read_csv(p)
        df["source_csv"] = p.name
        df["source_path"] = str(p)
        rows.extend(df.to_dict(orient="records"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_one(task: Dict) -> Dict:
    csv_path = task["csv_path"]
    try:
        metrics = run_simulations_from_csv(
            csv_path=csv_path,
            n_simulations=N_SIMULATIONS,
            n_agents=task["agents"],
            max_iterations=2 * task["agents"],
            generate_plot=False,
            force_extract=False,
            majority_ratio=task["majority"],
            initial_distribution_mode="ratio",
            source_tag="A100",
        )
        return {
            **task,
            **metrics,
            "status": "ok",
            "error": "",
        }
    except Exception as e:
        return {
            **task,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def run_model(model_slug: str, csv_files: List[Path], out_csv: Path) -> None:
    tasks = []
    for p in csv_files:
        for maj in MAJORITIES:
            for agents in AGENTS:
                tasks.append(
                    {
                        "model": model_slug,
                        "csv_path": str(p),
                        "csv_name": p.name,
                        "agents": agents,
                        "majority": maj,
                    }
                )

    results: List[Dict] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        total = len(futs)
        done = 0
        for fut in as_completed(futs):
            done += 1
            res = fut.result()
            results.append(res)
            if done % 20 == 0 or done == total:
                print(f"[{model_slug}] progresso {done}/{total}", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"OUTPUT_DIR={OUTPUT_DIR}", flush=True)
    print(f"MAX_WORKERS={MAX_WORKERS}", flush=True)
    print(
        f"SETTINGS: n_simulations={N_SIMULATIONS}, majorities={MAJORITIES}, agents={AGENTS}",
        flush=True,
    )

    models = ["gemma27b", "gemma4b"]
    merged_paths = {}
    model_csvs = {}

    for model in models:
        csv_files = list_basic_csvs(model)
        if len(csv_files) != 54:
            raise RuntimeError(f"{model}: esperado 54 CSVs basic, encontrado {len(csv_files)}")
        model_csvs[model] = csv_files
        merged = OUTPUT_DIR / f"{model}_basic_54_unificado_execucao_simultanea.csv"
        merge_basic_csvs(model, csv_files, merged)
        merged_paths[model] = merged
        print(f"[{model}] merged_csv={merged}", flush=True)

    out_27 = (
        OUTPUT_DIR
        / "gemma27b_automaton_resultados_54csv_agents30_60_90_120_majorias51_55_60_65_70_sem_plot.csv"
    )
    out_4 = (
        OUTPUT_DIR
        / "gemma4b_automaton_resultados_54csv_agents30_60_90_120_majorias51_55_60_65_70_sem_plot.csv"
    )

    run_model("gemma27b", model_csvs["gemma27b"], out_27)
    run_model("gemma4b", model_csvs["gemma4b"], out_4)

    print("DONE_ALL", flush=True)
    print(f"FILES:\n- {merged_paths['gemma27b']}\n- {merged_paths['gemma4b']}\n- {out_27}\n- {out_4}", flush=True)


if __name__ == "__main__":
    start = datetime.now()
    print(f"START={start.isoformat(sep=' ', timespec='seconds')}", flush=True)
    main()
    end = datetime.now()
    print(f"END={end.isoformat(sep=' ', timespec='seconds')}", flush=True)
