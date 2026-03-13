#!/usr/bin/env python3
import concurrent.futures as cf
import glob
import os
import re
import time
import traceback

import run_automaton_numba as ra

ROOT = "/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados"
RESULTS_SERVER = os.path.join(ROOT, "resultados_servidor")
RUN_TAG = "srv54_parallel_20260313"

MODEL_SOURCES = {
    "gemma4b": "Gemma 4B",
    "gemma27b": "Gemma 27B",
    "llama8b": "Llama 8B",
    "llama70b": "Llama 70B",
    "qwen3_4b": "Qwen 4B",
    "qwen_qwen3_32b": "Qwen 32B",
}
MODEL_SLUG_MAP = {
    "gemma4b": "gemma4b",
    "gemma27b": "gemma27b",
    "llama8b": "llama8b",
    "llama70b": "llama70b",
    "qwen3_4b": "qwen4b",
    "qwen_qwen3_32b": "qwen32b",
}

AGENTS_LIST = [30, 60, 90, 120]
MAJORITY_RATIOS = [0.51, 0.55, 0.60, 0.65, 0.70]
N_SIMULATIONS = 50


def n_sort_key(path: str):
    m = re.search(r"/n(\d+)/csv/basic/", path.replace("\\", "/"))
    n = int(m.group(1)) if m else 999
    order = {7: 0, 9: 1, 11: 2}.get(n, 999)
    return (order, os.path.basename(path))


def model_jobs(src_folder: str):
    csvs = []
    for n in ("n7", "n9", "n11"):
        pattern = os.path.join(RESULTS_SERVER, src_folder, n, "csv", "basic", "*.csv")
        csvs.extend(glob.glob(pattern))
    csvs = sorted(csvs, key=n_sort_key)
    if len(csvs) != 54:
        raise RuntimeError(f"{src_folder}: esperado 54 CSVs basic, encontrado {len(csvs)}")
    jobs = []
    for csv_path in csvs:
        for agents in AGENTS_LIST:
            for majority_ratio in MAJORITY_RATIOS:
                jobs.append((csv_path, agents, majority_ratio))
    return jobs


def run_one_model(src_folder: str):
    nice_name = MODEL_SOURCES[src_folder]
    model_slug = MODEL_SLUG_MAP[src_folder]
    jobs = model_jobs(src_folder)

    original_parse = ra.parse_csv_metadata

    def patched_parse(csv_path: str):
        meta = original_parse(csv_path)
        p = csv_path.replace("\\", "/")
        if f"/resultados_servidor/{src_folder}/" in p:
            meta["model"] = model_slug
        return meta

    ra.parse_csv_metadata = patched_parse

    total = len(jobs)
    failures = []
    start = time.time()
    print(f"[{nice_name}] START total_jobs={total}", flush=True)

    for idx, (csv_path, agents, majority_ratio) in enumerate(jobs, 1):
        csv_name = os.path.basename(csv_path)
        t0 = time.time()
        try:
            ra.run_simulations_from_csv(
                csv_path=csv_path,
                n_simulations=N_SIMULATIONS,
                n_agents=agents,
                max_iterations=2 * agents,
                generate_plot=False,
                force_extract=False,
                majority_ratio=majority_ratio,
                initial_distribution_mode="auto",
                source_tag=RUN_TAG,
            )
            dt = time.time() - t0
            print(
                f"[{nice_name}] [{idx}/{total}] DONE csv={csv_name} agents={agents} majority={majority_ratio:.2f} sec={dt:.2f}",
                flush=True,
            )
        except Exception as e:
            dt = time.time() - t0
            failures.append((csv_path, agents, majority_ratio, repr(e)))
            print(
                f"[{nice_name}] [{idx}/{total}] FAIL csv={csv_name} agents={agents} majority={majority_ratio:.2f} sec={dt:.2f} err={e!r}",
                flush=True,
            )
            traceback.print_exc()

    elapsed = time.time() - start
    print(
        f"[{nice_name}] FINISHED total_jobs={total} failures={len(failures)} elapsed_sec={elapsed:.2f}",
        flush=True,
    )
    return {"model": nice_name, "source": src_folder, "total_jobs": total, "failures": failures, "elapsed_sec": elapsed}


def main():
    print("MODEL COUNTS:")
    for src, nice in MODEL_SOURCES.items():
        print(f"  {nice}: csv_count={len(model_jobs(src)) // (len(AGENTS_LIST) * len(MAJORITY_RATIOS))}", flush=True)

    max_workers = len(MODEL_SOURCES)
    print(f"PARALLEL_WORKERS={max_workers} (one per model)")
    print(
        f"SETTINGS: n_simulations={N_SIMULATIONS}, agents={AGENTS_LIST}, majorities={MAJORITY_RATIOS}, max_iterations=2N, plots=True",
        flush=True,
    )

    start_all = time.time()
    results = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(run_one_model, src): src for src in MODEL_SOURCES.keys()}
        for fut in cf.as_completed(fut_map):
            src = fut_map[fut]
            try:
                res = fut.result()
                results.append(res)
                print(f"[MAIN] completed model={res['model']} failures={len(res['failures'])}", flush=True)
            except Exception as e:
                print(f"[MAIN] FATAL worker crash for source={src}: {e!r}", flush=True)
                raise

    total_failures = sum(len(r["failures"]) for r in results)
    elapsed_all = time.time() - start_all
    print(f"FINISHED_AUTOMATON total_models={len(results)} total_failures={total_failures} elapsed_sec={elapsed_all:.2f}")
    if total_failures:
        for r in results:
            for csv_path, agents, majority_ratio, err in r["failures"]:
                print(
                    f"FAIL_ITEM model={r['model']} csv={csv_path} agents={agents} majority={majority_ratio:.2f} err={err}",
                    flush=True,
                )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
