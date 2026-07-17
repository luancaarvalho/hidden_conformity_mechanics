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


def classification(summary: dict[str, Any]) -> str:
    if not summary["consensus"]:
        return "no_consensus"
    return "correct" if summary["correct_consensus"] else "wrong"


def first_state_difference(a: np.ndarray, b: np.ndarray) -> tuple[int | None, int | None]:
    differences = np.argwhere(~np.isclose(a, b, equal_nan=True))
    if not len(differences):
        return None, None
    round_idx, agent_idx = differences[0]
    return int(round_idx), int(agent_idx)


def load_summary(root: Path, mode: str, seed: int) -> dict[str, Any]:
    path = root / mode / "cells" / f"seed_{seed:02d}" / "result_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def compare_cell(replay1: Path, replay2: Path, mode: str, seed: int) -> dict[str, Any]:
    cell1 = replay1 / mode / "cells" / f"seed_{seed:02d}"
    cell2 = replay2 / mode / "cells" / f"seed_{seed:02d}"
    summary1 = load_summary(replay1, mode, seed)
    summary2 = load_summary(replay2, mode, seed)
    states1 = np.load(cell1 / "states.npy")
    states2 = np.load(cell2 / "states.npy")

    states_equal = bool(np.array_equal(states1, states2, equal_nan=True))
    first_diff_round, first_diff_agent = first_state_difference(states1, states2)
    classification1 = classification(summary1)
    classification2 = classification(summary2)
    classification_equal = classification1 == classification2
    consensus_token_equal = summary1["consensus_token"] == summary2["consensus_token"]
    completed_rounds_equal = summary1["completed_rounds"] == summary2["completed_rounds"]
    png_equal = summary1["png_sha256"] == summary2["png_sha256"]
    normalized_log_equal = (
        summary1["normalized_prompt_log_sha256"]
        == summary2["normalized_prompt_log_sha256"]
    )
    exact_equal = states_equal and png_equal and normalized_log_equal

    return {
        "mode": mode,
        "seed": seed,
        "expected_token": summary1["expected_token"],
        "classification_replay1": classification1,
        "classification_replay2": classification2,
        "classification_equal": classification_equal,
        "consensus_token_replay1": summary1["consensus_token"],
        "consensus_token_replay2": summary2["consensus_token"],
        "consensus_token_equal": consensus_token_equal,
        "completed_rounds_replay1": summary1["completed_rounds"],
        "completed_rounds_replay2": summary2["completed_rounds"],
        "completed_rounds_equal": completed_rounds_equal,
        "states_equal": states_equal,
        "png_equal": png_equal,
        "normalized_log_equal": normalized_log_equal,
        "exact_equal": exact_equal,
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
    parser.add_argument("--replay1", type=Path, required=True)
    parser.add_argument("--replay2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = [
        compare_cell(args.replay1, args.replay2, mode, seed)
        for mode in MODES
        for seed in range(1, 11)
    ]
    write_csv(args.output_dir / "paired_determinism_comparison.csv", rows)

    mode_summary: dict[str, dict[str, int]] = {}
    for mode in MODES:
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

    summary = {
        "replay1": str(args.replay1),
        "replay2": str(args.replay2),
        "cells": len(rows),
        "determinism_gate_pass": all(bool(row["exact_equal"]) for row in rows),
        "classification_equal": sum(bool(row["classification_equal"]) for row in rows),
        "completed_rounds_equal": sum(bool(row["completed_rounds_equal"]) for row in rows),
        "states_equal": sum(bool(row["states_equal"]) for row in rows),
        "png_equal": sum(bool(row["png_equal"]) for row in rows),
        "normalized_log_equal": sum(bool(row["normalized_log_equal"]) for row in rows),
        "exact_equal": sum(bool(row["exact_equal"]) for row in rows),
        "mode_summary": mode_summary,
        "classification_mismatches": [
            row for row in rows if not row["classification_equal"]
        ],
        "trajectory_mismatches": [
            row for row in rows if not row["states_equal"]
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Gemma 3 4B Determinism Comparison",
        "",
        f"- Gate: `{'PASS' if summary['determinism_gate_pass'] else 'FAIL'}`",
        f"- Exact cells: `{summary['exact_equal']}/{summary['cells']}`",
        f"- Classification-equal cells: `{summary['classification_equal']}/{summary['cells']}`",
        f"- State-equal cells: `{summary['states_equal']}/{summary['cells']}`",
        f"- PNG-equal cells: `{summary['png_equal']}/{summary['cells']}`",
        f"- Normalized-log-equal cells: `{summary['normalized_log_equal']}/{summary['cells']}`",
        "",
        "| Mode | Classification | States | PNG | Normalized log | Exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        item = mode_summary[mode]
        lines.append(
            f"| {mode} | {item['classification_equal']}/10 | {item['states_equal']}/10 | "
            f"{item['png_equal']}/10 | {item['normalized_log_equal']}/10 | {item['exact_equal']}/10 |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["determinism_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
