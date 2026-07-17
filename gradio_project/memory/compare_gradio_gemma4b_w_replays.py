#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


MODES = (
    "standard_only_token",
    "standard_cot",
    "conformity_only_token",
    "conformity_cot",
)
SEEDS = range(1, 11)


def classification(summary: dict[str, Any]) -> str:
    if not summary["consensus"]:
        return "no_consensus"
    return "correct" if summary["correct_consensus"] else "wrong"


def cell_dir(root: Path, mode: str, seed: int) -> Path:
    return root / mode / mode / "cells" / f"seed_{seed:02d}"


def load_summary(root: Path, mode: str, seed: int) -> dict[str, Any]:
    path = cell_dir(root, mode, seed) / "result_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def first_state_difference(a: np.ndarray, b: np.ndarray) -> tuple[int | None, int | None]:
    differences = np.argwhere(~np.isclose(a, b, equal_nan=True))
    if not len(differences):
        return None, None
    round_idx, agent_idx = differences[0]
    return int(round_idx), int(agent_idx)


def compare_cell(
    replay1: Path,
    replay2: Path,
    mode: str,
    seed: int,
    memory_window: int,
) -> dict[str, Any]:
    summary1 = load_summary(replay1, mode, seed)
    summary2 = load_summary(replay2, mode, seed)
    states1 = np.load(cell_dir(replay1, mode, seed) / "states.npy")
    states2 = np.load(cell_dir(replay2, mode, seed) / "states.npy")

    states_equal = bool(np.array_equal(states1, states2, equal_nan=True))
    first_diff_round, first_diff_agent = first_state_difference(states1, states2)
    parameters_equal = all(
        summary1[key] == summary2[key]
        for key in (
            "mode",
            "prompt_variant",
            "conformity_game",
            "seed",
            "agents",
            "rounds",
            "neighbors",
            "memory_window",
            "initial_majority_percent",
            "expected_token",
            "temperature",
            "request_seed",
            "max_output_tokens",
            "server_batch_invariant",
            "resolved_model",
        )
    ) and (
        summary1["memory_window"] == memory_window
        and summary1.get("request_timeout_s", 120)
        == summary2.get("request_timeout_s", 120)
    )
    classification_equal = classification(summary1) == classification(summary2)
    completed_rounds_equal = summary1["completed_rounds"] == summary2["completed_rounds"]
    consensus_token_equal = summary1["consensus_token"] == summary2["consensus_token"]
    png_equal = summary1["png_sha256"] == summary2["png_sha256"]
    normalized_log_equal = (
        summary1["normalized_prompt_log_sha256"]
        == summary2["normalized_prompt_log_sha256"]
    )
    exact_equal = parameters_equal and states_equal and png_equal and normalized_log_equal

    return {
        "memory_window": memory_window,
        "mode": mode,
        "seed": seed,
        "expected_token": summary1["expected_token"],
        "classification_replay1": classification(summary1),
        "classification_replay2": classification(summary2),
        "classification_equal": classification_equal,
        "consensus_token_replay1": summary1["consensus_token"],
        "consensus_token_replay2": summary2["consensus_token"],
        "consensus_token_equal": consensus_token_equal,
        "completed_rounds_replay1": summary1["completed_rounds"],
        "completed_rounds_replay2": summary2["completed_rounds"],
        "completed_rounds_equal": completed_rounds_equal,
        "parameters_equal": parameters_equal,
        "states_equal": states_equal,
        "png_equal": png_equal,
        "normalized_log_equal": normalized_log_equal,
        "request_failures_replay1": summary1.get("request_failure_count", 0),
        "request_failures_replay2": summary2.get("request_failure_count", 0),
        "exact_equal": exact_equal,
        "states_sha256_replay1": summary1["states_sha256"],
        "states_sha256_replay2": summary2["states_sha256"],
        "png_sha256_replay1": summary1["png_sha256"],
        "png_sha256_replay2": summary2["png_sha256"],
        "normalized_log_sha256_replay1": summary1["normalized_prompt_log_sha256"],
        "normalized_log_sha256_replay2": summary2["normalized_prompt_log_sha256"],
        "first_diff_round": first_diff_round,
        "first_diff_agent": first_diff_agent,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay1-root", type=Path, required=True)
    parser.add_argument("--replay2-root", type=Path, required=True)
    parser.add_argument("--memory-window", type=int, required=True, choices=range(0, 6))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = [
        compare_cell(args.replay1_root, args.replay2_root, mode, seed, args.memory_window)
        for mode in args.modes
        for seed in SEEDS
    ]
    write_csv(args.output_dir / "paired_determinism_comparison.csv", rows)

    with (args.output_dir / "replay02_hashes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "memory_window": row["memory_window"],
                        "mode": row["mode"],
                        "seed": row["seed"],
                        "states_sha256": row["states_sha256_replay2"],
                        "png_sha256": row["png_sha256_replay2"],
                        "normalized_prompt_log_sha256": row["normalized_log_sha256_replay2"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    mode_summary: dict[str, dict[str, int]] = {}
    for mode in args.modes:
        selected = [row for row in rows if row["mode"] == mode]
        mode_summary[mode] = {
            "cells": len(selected),
            "classification_equal": sum(bool(row["classification_equal"]) for row in selected),
            "completed_rounds_equal": sum(bool(row["completed_rounds_equal"]) for row in selected),
            "states_equal": sum(bool(row["states_equal"]) for row in selected),
            "png_equal": sum(bool(row["png_equal"]) for row in selected),
            "normalized_log_equal": sum(bool(row["normalized_log_equal"]) for row in selected),
            "exact_equal": sum(bool(row["exact_equal"]) for row in selected),
        }

    expected_cells = len(args.modes) * len(SEEDS)
    gate_pass = len(rows) == expected_cells and all(bool(row["exact_equal"]) for row in rows)
    summary = {
        "memory_window": args.memory_window,
        "replay1": str(args.replay1_root),
        "replay2": str(args.replay2_root),
        "cells": len(rows),
        "expected_cells": expected_cells,
        "modes": args.modes,
        "determinism_gate_pass": gate_pass,
        "classification_equal": sum(bool(row["classification_equal"]) for row in rows),
        "completed_rounds_equal": sum(bool(row["completed_rounds_equal"]) for row in rows),
        "states_equal": sum(bool(row["states_equal"]) for row in rows),
        "png_equal": sum(bool(row["png_equal"]) for row in rows),
        "normalized_log_equal": sum(bool(row["normalized_log_equal"]) for row in rows),
        "exact_equal": sum(bool(row["exact_equal"]) for row in rows),
        "mode_summary": mode_summary,
        "mismatches": [row for row in rows if not row["exact_equal"]],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# Gemma 3 4B Determinism Comparison - W={args.memory_window}",
        "",
        f"- Gate: `{'PASS' if gate_pass else 'FAIL'}`",
        f"- Exact cells: `{summary['exact_equal']}/{expected_cells}`",
        f"- States: `{summary['states_equal']}/{expected_cells}`",
        f"- PNG: `{summary['png_equal']}/{expected_cells}`",
        f"- Normalized logs: `{summary['normalized_log_equal']}/{expected_cells}`",
        "",
        "| Mode | Classification | States | PNG | Log | Exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, item in mode_summary.items():
        lines.append(
            f"| {mode} | {item['classification_equal']}/10 | {item['states_equal']}/10 | "
            f"{item['png_equal']}/10 | {item['normalized_log_equal']}/10 | {item['exact_equal']}/10 |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
