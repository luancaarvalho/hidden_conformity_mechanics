#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from gradio_project.interface.interface_v4_gradio import (  # noqa: E402
    CONFORMITY_GAME_MODE_B,
    SimulationRunner,
)


TIMESTAMP_RE = re.compile(r"^TIMESTAMP: .*? \| ROUND:", flags=re.MULTILINE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalize_prompt_log(text: str) -> str:
    return TIMESTAMP_RE.sub("TIMESTAMP: <normalized> | ROUND:", text)


def completed_rounds(states: np.ndarray) -> int:
    return sum(not np.isnan(row).all() for row in states)


def run_once(args: argparse.Namespace, output_dir: Path, run_id: str) -> dict:
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    runner = SimulationRunner(
        n_agents=args.agents,
        n_rounds=args.rounds,
        n_neighbors=args.neighbors,
        temperature=args.temperature,
        prompt_variant=args.prompt_variant,
        base_url=args.base_url,
        model=args.model,
        delay_ms=0,
        initial_majority=args.initial_majority,
        stop_event=threading.Event(),
        seed=args.distribution_seed,
        initial_config_json=None,
        memory_window=args.memory_window,
        memory_format="timeline",
        jogo_conformidade=True,
        jogo_conformidade_modo=CONFORMITY_GAME_MODE_B,
        token0=args.token0,
        token1=args.token1,
    )
    runner.log_filepath = str(run_dir / "prompt_log.txt")

    final_status = "not_started"
    updates = 0
    for payload in runner.run_stream():
        updates += 1
        final_status = str(payload[2])

    if runner.current_image is None:
        raise RuntimeError(f"{run_id} completed without a figure: {final_status}")

    states_path = run_dir / "states.npy"
    figure_path = run_dir / "final_figure.png"
    np.save(states_path, runner.states)
    figure_path.write_bytes(runner.current_image)

    prompt_log_path = Path(runner.log_filepath)
    prompt_log = prompt_log_path.read_text(encoding="utf-8") if prompt_log_path.exists() else ""
    normalized_log = normalize_prompt_log(prompt_log)
    normalized_log_path = run_dir / "prompt_log_normalized.txt"
    normalized_log_path.write_text(normalized_log, encoding="utf-8")

    valid_round_count = completed_rounds(runner.states)
    final_row = runner.states[valid_round_count - 1]
    final_valid = final_row[~np.isnan(final_row)]
    consensus = bool(len(final_valid) == args.agents and np.all(final_valid == final_valid[0]))
    summary = {
        "run_id": run_id,
        "status": final_status,
        "updates": updates,
        "resolved_model": runner.model,
        "max_output_tokens": runner.max_output_tokens,
        "request_seed": 42,
        "completed_rounds": valid_round_count,
        "consensus": consensus,
        "consensus_token": (
            runner.opinion_pair[int(final_valid[0])] if consensus else None
        ),
        "states_sha256": sha256_file(states_path),
        "figure_sha256": sha256_file(figure_path),
        "normalized_prompt_log_sha256": sha256_file(normalized_log_path),
        "prompt_log_requests": prompt_log.count(">>> RESPONSE (RAW)"),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if final_status.startswith("Erro"):
        raise RuntimeError(f"{run_id} failed: {final_status}")
    return summary


def compare_runs(output_dir: Path, run1: dict, run2: dict) -> dict:
    states1 = np.load(output_dir / "run1" / "states.npy")
    states2 = np.load(output_dir / "run2" / "states.npy")
    pixels1 = np.asarray(Image.open(output_dir / "run1" / "final_figure.png").convert("RGBA"))
    pixels2 = np.asarray(Image.open(output_dir / "run2" / "final_figure.png").convert("RGBA"))

    states_equal = bool(np.array_equal(states1, states2, equal_nan=True))
    pixel_shapes_equal = pixels1.shape == pixels2.shape
    pixels_equal = bool(pixel_shapes_equal and np.array_equal(pixels1, pixels2))
    differing_pixels = (
        int(np.count_nonzero(np.any(pixels1 != pixels2, axis=-1)))
        if pixel_shapes_equal
        else None
    )
    png_bytes_equal = run1["figure_sha256"] == run2["figure_sha256"]
    normalized_logs_equal = (
        run1["normalized_prompt_log_sha256"]
        == run2["normalized_prompt_log_sha256"]
    )
    comparison = {
        "gate_pass": states_equal and pixels_equal and normalized_logs_equal,
        "states_equal": states_equal,
        "png_bytes_equal": png_bytes_equal,
        "pixels_equal": pixels_equal,
        "pixel_shapes": [list(pixels1.shape), list(pixels2.shape)],
        "differing_pixels": differing_pixels,
        "normalized_prompt_logs_equal": normalized_logs_equal,
        "run1": run1,
        "run2": run2,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8127/v1")
    parser.add_argument("--model", default="gemma3-4b-temp0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=9)
    parser.add_argument("--memory-window", type=int, default=3)
    parser.add_argument("--initial-majority", type=int, default=51)
    parser.add_argument("--distribution-seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=8)
    parser.add_argument("--prompt-variant", default="v9_lista_completa_meio_kz")
    parser.add_argument("--token0", default="k")
    parser.add_argument("--token1", default="z")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    os.environ["LLM_MAX_OUTPUT_TOKENS"] = str(args.max_output_tokens)

    interface_path = (
        WORKSPACE_ROOT
        / "gradio_project"
        / "interface"
        / "interface_v4_gradio.py"
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Two-run deterministic replay through Gradio v4 SimulationRunner",
        "interface_path": str(interface_path),
        "interface_sha256": sha256_file(interface_path),
        "configuration": {
            "base_url": args.base_url,
            "model": args.model,
            "agents": args.agents,
            "rounds": args.rounds,
            "neighbors": args.neighbors,
            "memory_window": args.memory_window,
            "initial_majority": args.initial_majority,
            "distribution_seed": args.distribution_seed,
            "temperature": args.temperature,
            "request_seed": 42,
            "max_output_tokens": args.max_output_tokens,
            "prompt_variant": args.prompt_variant,
            "conformity_mode": CONFORMITY_GAME_MODE_B,
            "tokens": [args.token0, args.token1],
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    run1 = run_once(args, args.output_dir, "run1")
    run2 = run_once(args, args.output_dir, "run2")
    comparison = compare_runs(args.output_dir, run1, run2)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0 if comparison["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
