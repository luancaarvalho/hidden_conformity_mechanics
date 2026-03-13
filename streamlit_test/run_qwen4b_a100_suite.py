#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence


def _parse_models(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _expand_base_urls(raw: str, worker_count: int) -> List[str]:
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not items:
        raise ValueError("No base URLs provided")
    if len(items) == 1:
        return items * worker_count
    if len(items) != worker_count:
        raise ValueError(f"--base-urls must have 1 or {worker_count} entries")
    return items


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run fixed-worker Qwen 4B experiments on A100 and measure wall time.")
    p.add_argument("--base-url", required=True)
    p.add_argument("--base-urls", default="")
    p.add_argument("--models", required=True, help="Comma-separated alias list, one per worker.")
    p.add_argument("--output-root", required=True)
    p.add_argument("--prompt-variant", required=True)
    p.add_argument("--agents-list", default="30,50,100,120")
    p.add_argument("--memory-windows", default="3,5")
    p.add_argument("--neighbors", type=int, default=7)
    p.add_argument("--initial-majority-ratio", type=float, default=0.51)
    p.add_argument("--initial-distribution-mode", default="half_split")
    p.add_argument("--seeds-distribution", default="2")
    p.add_argument("--request-seed", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--mode", choices=["skip", "overwrite"], default="overwrite")
    p.add_argument("--conformity-game", action="store_true")
    p.add_argument("--conformity-game-mode")
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    worker_script = repo_root / "streamlit_test" / "run_qwen4b_a100_worker.py"
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    models = _parse_models(args.models)
    if not models:
        raise ValueError("No models provided in --models")
    worker_base_urls = _expand_base_urls(args.base_urls or args.base_url, len(models))

    suite_meta = {
        "base_url": args.base_url,
        "base_urls": worker_base_urls,
        "models": models,
        "prompt_variant": args.prompt_variant,
        "agents_list": args.agents_list,
        "memory_windows": args.memory_windows,
        "neighbors": args.neighbors,
        "initial_majority_ratio": args.initial_majority_ratio,
        "initial_distribution_mode": args.initial_distribution_mode,
        "seeds_distribution": args.seeds_distribution,
        "request_seed": args.request_seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "mode": args.mode,
        "conformity_game": bool(args.conformity_game),
        "conformity_game_mode": args.conformity_game_mode,
        "worker_count": len(models),
    }
    (output_root / "suite_config.json").write_text(
        json.dumps(suite_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    processes = []
    t0 = time.time()
    for idx, model in enumerate(models):
        cmd = [
            sys.executable,
            str(worker_script),
            "--base-url",
            worker_base_urls[idx],
            "--model",
            model,
            "--output-root",
            str(output_root),
            "--prompt-variant",
            args.prompt_variant,
            "--agents-list",
            args.agents_list,
            "--memory-windows",
            args.memory_windows,
            "--neighbors",
            str(args.neighbors),
            "--initial-majority-ratio",
            str(args.initial_majority_ratio),
            "--initial-distribution-mode",
            args.initial_distribution_mode,
            "--seeds-distribution",
            args.seeds_distribution,
            "--request-seed",
            str(args.request_seed),
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--worker-index",
            str(idx),
            "--worker-count",
            str(len(models)),
            "--mode",
            args.mode,
        ]
        if args.conformity_game:
            cmd.append("--conformity-game")
            if args.conformity_game_mode:
                cmd.extend(["--conformity-game-mode", str(args.conformity_game_mode)])

        stdout_path = output_root / f"worker_{idx:02d}.stdout.log"
        stderr_path = output_root / f"worker_{idx:02d}.stderr.log"
        stdout_f = stdout_path.open("w", encoding="utf-8")
        stderr_f = stderr_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, text=True)
        processes.append((idx, model, proc, stdout_f, stderr_f))

    failed = False
    worker_status = []
    for idx, model, proc, stdout_f, stderr_f in processes:
        rc = proc.wait()
        stdout_f.close()
        stderr_f.close()
        worker_status.append(
            {
                "worker_index": idx,
                "model": model,
                "base_url": worker_base_urls[idx],
                "returncode": rc,
            }
        )
        if rc != 0:
            failed = True

    elapsed_s = time.time() - t0
    result = {
        **suite_meta,
        "wall_time_s": elapsed_s,
        "failed": failed,
        "workers": worker_status,
    }
    (output_root / "suite_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
