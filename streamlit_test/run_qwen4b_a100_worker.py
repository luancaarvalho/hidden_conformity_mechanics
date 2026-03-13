#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple


def _parse_csv_ints(raw: str) -> List[int]:
    values: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def _build_tasks(agents: Sequence[int], memory_windows: Sequence[int]) -> List[Tuple[int, int]]:
    return [(agent_count, memory_window) for agent_count in agents for memory_window in memory_windows]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fixed-model worker for Qwen 4B batch experiments on A100.")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--prompt-variant", required=True)
    p.add_argument("--agents-list", required=True, help="Comma-separated list, e.g. 30,50,100,120")
    p.add_argument("--memory-windows", required=True, help="Comma-separated list, e.g. 3,5")
    p.add_argument("--neighbors", type=int, default=7)
    p.add_argument("--initial-majority-ratio", type=float, default=0.51)
    p.add_argument("--initial-distribution-mode", default="half_split")
    p.add_argument("--seeds-distribution", default="2")
    p.add_argument("--request-seed", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--worker-index", type=int, required=True)
    p.add_argument("--worker-count", type=int, required=True)
    p.add_argument("--mode", choices=["skip", "overwrite"], default="overwrite")
    p.add_argument("--conformity-game", action="store_true")
    p.add_argument("--conformity-game-mode")
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    run_batch_png = repo_root / "streamlit_test" / "run_batch_png.py"

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    worker_dir = output_root / f"worker_{args.worker_index:02d}_{args.model.replace('/', '_')}"
    worker_dir.mkdir(parents=True, exist_ok=True)

    agents = _parse_csv_ints(args.agents_list)
    memory_windows = _parse_csv_ints(args.memory_windows)
    all_tasks = _build_tasks(agents, memory_windows)
    assigned = [task for idx, task in enumerate(all_tasks) if (idx % args.worker_count) == args.worker_index]

    manifest = {
        "model": args.model,
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "tasks": [{"agents": a, "memory_window": m} for a, m in assigned],
        "prompt_variant": args.prompt_variant,
        "conformity_game": bool(args.conformity_game),
        "conformity_game_mode": args.conformity_game_mode,
        "request_seed": args.request_seed,
        "seeds_distribution": args.seeds_distribution,
    }
    (worker_dir / "worker_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path = worker_dir / "worker_runs.jsonl"
    with summary_path.open("w", encoding="utf-8") as summary_f:
        for agent_count, memory_window in assigned:
            task_dir = worker_dir / f"agents_{agent_count}" / f"memory_w_{memory_window}"
            cmd = [
                sys.executable,
                str(run_batch_png),
                "--base-url",
                args.base_url,
                "--model",
                args.model,
                "--temperature",
                str(args.temperature),
                "--max-tokens",
                str(args.max_tokens),
                "--request-seed",
                str(args.request_seed),
                "--prompt-variant",
                args.prompt_variant,
                "--agents",
                str(agent_count),
                "--neighbors",
                str(args.neighbors),
                "--initial-majority-ratio",
                str(args.initial_majority_ratio),
                "--initial-distribution-mode",
                args.initial_distribution_mode,
                "--seeds-distribution",
                args.seeds_distribution,
                "--memory-windows",
                str(memory_window),
                "--output-dir",
                str(task_dir),
                "--mode",
                args.mode,
            ]
            if args.conformity_game:
                cmd.append("--conformity-game")
                if args.conformity_game_mode:
                    cmd.extend(["--conformity-game-mode", str(args.conformity_game_mode)])

            t0 = time.time()
            env = dict(os.environ)
            env["FORCE_QWEN_NO_THINK"] = "1"
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            elapsed_s = time.time() - t0
            rec = {
                "agents": agent_count,
                "memory_window": memory_window,
                "returncode": result.returncode,
                "elapsed_s": elapsed_s,
                "task_dir": str(task_dir),
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            }
            summary_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            summary_f.flush()
            if result.returncode != 0:
                return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
