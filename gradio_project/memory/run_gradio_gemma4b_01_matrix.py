#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import re
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

TIMESTAMP_RE = re.compile(r"^TIMESTAMP: .*? \| ROUND:", flags=re.MULTILINE)
SAMPLING_OVERRIDE_ENV = (
    "LLM_TOP_K",
    "LLM_TOP_P",
    "LLM_MIN_P",
    "LLM_REPEAT_PENALTY",
)
MODE_CONFIGS = {
    "standard_only_token": {
        "prompt_variant": "v9_lista_completa_meio_01",
        "conformity_game": False,
        "max_output_tokens": 8,
    },
    "standard_cot": {
        "prompt_variant": "v21_zero_shot_cot_01",
        "conformity_game": False,
        "max_output_tokens": 768,
    },
    "conformity_only_token": {
        "prompt_variant": "v9_lista_completa_meio_01",
        "conformity_game": True,
        "max_output_tokens": 8,
    },
    "conformity_cot": {
        "prompt_variant": "v21_zero_shot_cot_01",
        "conformity_game": True,
        "max_output_tokens": 768,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_rounds(states: np.ndarray) -> int:
    return sum(not np.isnan(row).all() for row in states)


def run_cell(task: dict[str, Any]) -> dict[str, Any]:
    for name in SAMPLING_OVERRIDE_ENV:
        os.environ.pop(name, None)
    os.environ["LLM_MAX_OUTPUT_TOKENS"] = str(task["max_output_tokens"])
    os.environ["LLM_REQUEST_TIMEOUT_S"] = str(task["request_timeout_s"])

    from gradio_project.interface.interface_v4_gradio import (
        CONFORMITY_GAME_MODE_B,
        SimulationRunner,
    )

    output_root = Path(task["output_root"])
    mode_dir = output_root / task["mode"]
    cell_dir = mode_dir / "cells" / f"seed_{task['seed']:02d}"
    image_path = mode_dir / f"seed_{task['seed']:02d}.png"
    cell_dir.mkdir(parents=True, exist_ok=False)

    runner = SimulationRunner(
        n_agents=task["agents"],
        n_rounds=task["rounds"],
        n_neighbors=task["neighbors"],
        temperature=task["temperature"],
        prompt_variant=task["prompt_variant"],
        base_url=task["base_url"],
        model=task["model"],
        delay_ms=0,
        initial_majority=task["initial_majority"],
        stop_event=threading.Event(),
        seed=task["seed"],
        initial_config_json=None,
        memory_window=task["memory_window"],
        memory_format="timeline",
        jogo_conformidade=task["conformity_game"],
        jogo_conformidade_modo=CONFORMITY_GAME_MODE_B,
        token0="0",
        token1="1",
    )
    runner.log_filepath = str(cell_dir / "prompt_log.txt")

    final_status = "not_started"
    updates = 0
    for payload in runner.run_stream():
        updates += 1
        final_status = str(payload[2])

    if runner.current_image is None:
        raise RuntimeError(f"seed {task['seed']} produced no PNG: {final_status}")

    states_path = cell_dir / "states.npy"
    np.save(states_path, runner.states)
    image_path.write_bytes(runner.current_image)

    prompt_log_path = Path(runner.log_filepath)
    prompt_log = (
        prompt_log_path.read_text(encoding="utf-8")
        if prompt_log_path.exists()
        else ""
    )
    normalized_log = TIMESTAMP_RE.sub("TIMESTAMP: <normalized> | ROUND:", prompt_log)
    normalized_log_path = cell_dir / "prompt_log_normalized.txt"
    normalized_log_path.write_text(normalized_log, encoding="utf-8")

    valid_rounds = completed_rounds(runner.states)
    initial_row = runner.states[0]
    count_0 = int(np.sum(initial_row == 0))
    count_1 = int(np.sum(initial_row == 1))
    expected_token = "0" if count_0 > count_1 else "1"
    final_row = runner.states[valid_rounds - 1]
    final_valid = final_row[~np.isnan(final_row)]
    consensus = bool(
        len(final_valid) == task["agents"]
        and np.all(final_valid == final_valid[0])
    )
    consensus_token = str(int(final_valid[0])) if consensus else None
    stop_reason = "consensus" if consensus else "max_rounds"
    if valid_rounds != task["rounds"] and not consensus:
        stop_reason = "interrupted_or_error"
    row = {
        "mode": task["mode"],
        "prompt_variant": task["prompt_variant"],
        "conformity_game": task["conformity_game"],
        "conformity_mode": CONFORMITY_GAME_MODE_B if task["conformity_game"] else None,
        "seed": task["seed"],
        "agents": task["agents"],
        "rounds": task["rounds"],
        "neighbors": task["neighbors"],
        "memory_window": task["memory_window"],
        "initial_majority_percent": task["initial_majority"],
        "initial_count_0": count_0,
        "initial_count_1": count_1,
        "expected_token": expected_token,
        "temperature": task["temperature"],
        "request_seed": 42,
        "max_output_tokens": task["max_output_tokens"],
        "request_timeout_s": task["request_timeout_s"],
        "server_batch_invariant": task["server_batch_invariant"],
        "resolved_model": runner.model,
        "completed_rounds": valid_rounds,
        "status": final_status,
        "stop_reason": stop_reason,
        "consensus": consensus,
        "consensus_token": consensus_token,
        "correct_consensus": consensus_token == expected_token,
        "raw_response_count": prompt_log.count(">>> RESPONSE (RAW)"),
        "parse_failure_count": prompt_log.count(">>> PARSE_ERROR:"),
        "request_failure_count": int(runner.request_failure_count),
        "states_sha256": sha256_file(states_path),
        "png_sha256": sha256_file(image_path),
        "normalized_prompt_log_sha256": sha256_file(normalized_log_path),
        "image_path": str(image_path),
    }
    (cell_dir / "result_summary.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8127/v1")
    parser.add_argument("--model", default="gemma3-4b-temp0")
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=7)
    parser.add_argument("--memory-window", type=int, default=3)
    parser.add_argument("--initial-majority", type=int, default=51)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seeds", default="1-10")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--request-timeout-s", type=int, default=120)
    parser.add_argument("--server-batch-invariant", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(MODE_CONFIGS),
        default=list(MODE_CONFIGS),
    )
    return parser.parse_args()


def parse_seed_range(value: str) -> list[int]:
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item) for item in value.split(",")]


def main() -> int:
    args = parse_args()
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        raise ValueError("--max-output-tokens must be a positive integer")
    if args.request_timeout_s < 1:
        raise ValueError("--request-timeout-s must be a positive integer")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)

    seeds = parse_seed_range(args.seeds)
    tasks: list[dict[str, Any]] = []
    for mode in args.modes:
        config = MODE_CONFIGS[mode]
        max_output_tokens = (
            args.max_output_tokens
            if args.max_output_tokens is not None
            else config["max_output_tokens"]
        )
        (args.output_root / mode / "cells").mkdir(parents=True)
        for seed in seeds:
            tasks.append(
                {
                    "output_root": str(args.output_root),
                    "mode": mode,
                    "seed": seed,
                    "base_url": args.base_url,
                    "model": args.model,
                    "agents": args.agents,
                    "rounds": args.rounds,
                    "neighbors": args.neighbors,
                    "memory_window": args.memory_window,
                    "initial_majority": args.initial_majority,
                    "temperature": args.temperature,
                    "request_timeout_s": args.request_timeout_s,
                    "server_batch_invariant": args.server_batch_invariant,
                    "prompt_variant": config["prompt_variant"],
                    "conformity_game": config["conformity_game"],
                    "max_output_tokens": max_output_tokens,
                }
            )

    source_files = [
        Path(__file__).resolve(),
        WORKSPACE_ROOT / "gradio_project" / "prompts" / "prompt_strategies.py",
        WORKSPACE_ROOT / "gradio_project" / "prompts" / "prompt_templates.yaml",
        WORKSPACE_ROOT / "gradio_project" / "interface" / "interface_v4_gradio.py",
        WORKSPACE_ROOT / "gradio_project" / "utils" / "conformity_game_prompts.py",
    ]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Gemma 3 4B 0/1 matrix: selected prompt modes",
        "sampling": {
            "temperature": args.temperature,
            "request_seed": 42,
            "sampling_overrides": "omitted",
            "max_output_tokens_by_mode": {
                mode: next(
                    task["max_output_tokens"]
                    for task in tasks
                    if task["mode"] == mode
                )
                for mode in args.modes
            },
            "request_timeout_s": args.request_timeout_s,
        },
        "engine": {
            "server_batch_invariant": args.server_batch_invariant,
        },
        "protocol": {
            "agents": args.agents,
            "rounds": args.rounds,
            "neighbors": args.neighbors,
            "memory_window": args.memory_window,
            "initial_majority_percent": args.initial_majority,
            "tokens": ["0", "1"],
            "seeds": seeds,
        },
        "source_hashes": {
            str(path): sha256_file(path) for path in source_files
        },
        "expected_png_count": len(tasks),
        "tasks": tasks,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    progress_path = args.output_root / "progress.jsonl"
    context = multiprocessing.get_context("spawn")

    # Modes run in sequence so only-token and CoT token budgets cannot leak across cells.
    for mode in args.modes:
        mode_tasks = [task for task in tasks if task["mode"] == mode]
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(mode_tasks)),
            mp_context=context,
        ) as pool:
            future_to_task = {pool.submit(run_cell, task): task for task in mode_tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    row = future.result()
                    rows.append(row)
                    event = {"event": "completed", **row}
                except Exception as exc:
                    error = {
                        "mode": task["mode"],
                        "seed": task["seed"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    errors.append(error)
                    event = {"event": "error", **error}
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                print(json.dumps(event, ensure_ascii=False), flush=True)

    rows.sort(key=lambda row: (row["mode"], row["seed"]))
    write_csv(args.output_root / "scoreboard.csv", rows)
    summary = {
        "completed_cells": len(rows),
        "error_cells": len(errors),
        "png_count": len(list(args.output_root.glob("*/*.png"))),
        "expected_png_count": len(tasks),
        "modes": {
            mode: {
                "completed": sum(row["mode"] == mode for row in rows),
                "correct_consensus": sum(
                    row["mode"] == mode and row["correct_consensus"] for row in rows
                ),
                "wrong_consensus": sum(
                    row["mode"] == mode
                    and row["consensus"]
                    and not row["correct_consensus"]
                    for row in rows
                ),
                "no_consensus": sum(
                    row["mode"] == mode and not row["consensus"] for row in rows
                ),
                "parse_failures": sum(
                    row["parse_failure_count"] for row in rows if row["mode"] == mode
                ),
                "request_failures": sum(
                    row["request_failure_count"] for row in rows if row["mode"] == mode
                ),
            }
            for mode in args.modes
        },
        "errors": errors,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0 if not errors and summary["png_count"] == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
