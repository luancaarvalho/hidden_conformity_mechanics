#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.parity_artifacts import (  # noqa: E402
    build_artifact_index,
    sha256_file,
    source_hashes,
    write_csv,
    write_json,
)
from utils.w0_parity_contract import (  # noqa: E402
    parse_final_choice,
    source_paths,
    token_pair_spec,
    variants_for_pair,
)


MODEL_FAMILY = "gemma-3-4b-it"
PAIR_ORDER = ("01", "kz", "triangle_circle")
STYLE_NAMES = ("v9_only_token", "v21_cot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True)
    parser.add_argument(
        "--run-01-w0",
        default="rules-ca-parity-w0-seed1-20_20260720T143020Z",
    )
    parser.add_argument(
        "--run-01-w1",
        default="memory-W1-seed1-20_20260720T174027Z",
    )
    parser.add_argument("--run-kz-w0", required=True)
    parser.add_argument("--run-kz-w1", required=True)
    parser.add_argument("--run-triangle-circle-w0", required=True)
    parser.add_argument("--run-triangle-circle-w1", required=True)
    parser.add_argument("--seeds", default="1-20")
    return parser.parse_args()


def parse_seed_range(value: str) -> list[int]:
    start, end = value.split("-", 1)
    return list(range(int(start), int(end) + 1))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def phase3_root(token_pair: str, memory_window: int, variant: str, run_id: str) -> Path:
    return (
        REPO_ROOT
        / "artifacts/phase3_memory/rtx5090/n=7"
        / MODEL_FAMILY
        / token_pair_spec(token_pair)["folder"]
        / f"W={memory_window}"
        / variant
        / run_id
    )


def cross_phase_root(token_pair: str, run_id: str) -> Path:
    return (
        REPO_ROOT
        / "artifacts/cross_phase_validation/rtx5090/n=7"
        / MODEL_FAMILY
        / token_pair_spec(token_pair)["folder"]
        / run_id
    )


def valid_states(states: np.ndarray) -> np.ndarray:
    if states.ndim != 2 or states.shape[1] != 30:
        raise ValueError(f"unexpected states shape: {states.shape}")
    valid_mask = ~np.isnan(states).any(axis=1)
    count = int(valid_mask.sum())
    if count == 0 or not valid_mask[:count].all() or valid_mask[count:].any():
        raise ValueError("valid state rows must be a non-empty contiguous prefix")
    return states[:count].astype(np.int8)


def classify(states: np.ndarray) -> dict[str, Any]:
    initial = states[0]
    count0 = int(np.sum(initial == 0))
    count1 = int(np.sum(initial == 1))
    if count0 == count1:
        raise ValueError("initial state has no strict majority")
    expected_state = int(count1 > count0)
    final = states[-1]
    consensus = bool(np.all(final == final[0]))
    consensus_state = int(final[0]) if consensus else None
    classification = (
        "no_consensus"
        if not consensus
        else "correct" if consensus_state == expected_state else "wrong"
    )
    return {
        "initial_count_0": count0,
        "initial_count_1": count1,
        "expected_state": expected_state,
        "consensus_state": consensus_state,
        "classification": classification,
        "completed_transitions": len(states) - 1,
        "stop_reason": "consensus" if consensus else "max_rounds",
    }


def proofread_cell(
    *,
    token_pair: str,
    memory_window: int,
    style: str,
    variant: str,
    run_id: str,
    seed: int,
) -> dict[str, Any]:
    root = phase3_root(token_pair, memory_window, variant, run_id)
    cell = root / "canonical/cells" / f"seed_{seed:02d}"
    states_path = cell / "states.npy"
    requests_path = cell / "requests.jsonl"
    summary_path = cell / "result_summary.json"
    if not all(path.is_file() for path in (states_path, requests_path, summary_path)):
        raise FileNotFoundError(f"incomplete canonical cell: {cell}")

    states = valid_states(np.load(states_path))
    derived = classify(states)
    summary = read_json(summary_path)
    requests = read_jsonl(requests_path)
    tokens = token_pair_spec(token_pair)["tokens"]
    expected_requests = derived["completed_transitions"] * len(states[0])
    request_failures = sum(int(row["request_failures"]) for row in requests)
    parse_failures = sum(int(row["parse_failures"]) for row in requests)
    contract_failures = 0
    for row in requests:
        choice_token = row.get("choice_token")
        choice = row.get("choice")
        if (
            choice_token not in tokens
            or choice not in (0, 1)
            or tokens[int(choice)] != choice_token
            or parse_final_choice(row.get("raw_response", ""), variant) != choice_token
        ):
            contract_failures += 1

    expected_token = tokens[derived["expected_state"]]
    consensus_token = (
        tokens[derived["consensus_state"]]
        if derived["consensus_state"] is not None
        else None
    )
    summary_matches = all(
        (
            summary["initial_count_0"] == derived["initial_count_0"],
            summary["initial_count_1"] == derived["initial_count_1"],
            summary["expected_token"] == expected_token,
            summary["consensus_token"] == consensus_token,
            summary["completed_transitions"] == derived["completed_transitions"],
            summary["stop_reason"] == derived["stop_reason"],
            summary["correct_consensus"] == (derived["classification"] == "correct"),
            summary["request_failures"] == request_failures,
            summary["parse_failures"] == parse_failures,
        )
    )
    return {
        "token_pair": token_pair,
        "token0": tokens[0],
        "token1": tokens[1],
        "memory_window": memory_window,
        "style": style,
        "variant": variant,
        "run_id": run_id,
        "seed": seed,
        "initial_count_0": derived["initial_count_0"],
        "initial_count_1": derived["initial_count_1"],
        "expected_token": expected_token,
        "consensus_token": consensus_token,
        "classification": derived["classification"],
        "completed_transitions": derived["completed_transitions"],
        "stop_reason": derived["stop_reason"],
        "request_records": len(requests),
        "expected_request_records": expected_requests,
        "request_failures": request_failures,
        "parse_failures": parse_failures,
        "strict_contract_failures": contract_failures,
        "summary_matches_raw": summary_matches,
        "states_sha256": sha256_file(states_path),
        "requests_sha256": sha256_file(requests_path),
    }


def grouped_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["token_pair"], row["memory_window"], row["style"])].append(row)
    summaries = []
    for (token_pair, memory_window, style), group in sorted(grouped.items()):
        expected_sides = {row["expected_token"] for row in group}
        mechanical_pass = len(group) == 20 and all(
            row["request_records"] == row["expected_request_records"]
            and row["request_failures"] == 0
            and row["parse_failures"] == 0
            and row["strict_contract_failures"] == 0
            and row["summary_matches_raw"]
            for row in group
        )
        summaries.append(
            {
                "token_pair": token_pair,
                "memory_window": memory_window,
                "style": style,
                "cells": len(group),
                "correct": sum(row["classification"] == "correct" for row in group),
                "wrong": sum(row["classification"] == "wrong" for row in group),
                "no_consensus": sum(
                    row["classification"] == "no_consensus" for row in group
                ),
                "both_majority_sides_present": len(expected_sides) == 2,
                "mean_completed_transitions": sum(
                    row["completed_transitions"] for row in group
                )
                / len(group),
                "mechanical_pass": mechanical_pass,
                "dct_bidirectional_pass": (
                    mechanical_pass
                    and len(expected_sides) == 2
                    and all(row["classification"] == "correct" for row in group)
                ),
            }
        )
    return summaries


def main() -> int:
    args = parse_args()
    seeds = parse_seed_range(args.seeds)
    if seeds != list(range(1, 21)):
        raise ValueError("cross-token comparison requires seeds 1-20")
    output_root = (
        REPO_ROOT
        / "artifacts/cross_token_validation/rtx5090/n=7"
        / MODEL_FAMILY
        / args.batch_id
    )
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    run_ids = {
        "01": {0: args.run_01_w0, 1: args.run_01_w1},
        "kz": {0: args.run_kz_w0, 1: args.run_kz_w1},
        "triangle_circle": {
            0: args.run_triangle_circle_w0,
            1: args.run_triangle_circle_w1,
        },
    }
    rows: list[dict[str, Any]] = []
    roots: list[Path] = [output_root]
    for token_pair in PAIR_ORDER:
        variants = variants_for_pair(token_pair)
        for memory_window in (0, 1):
            run_id = run_ids[token_pair][memory_window]
            roots.append(cross_phase_root(token_pair, run_id))
            for style, variant in zip(STYLE_NAMES, variants):
                roots.append(phase3_root(token_pair, memory_window, variant, run_id))
                for seed in seeds:
                    rows.append(
                        proofread_cell(
                            token_pair=token_pair,
                            memory_window=memory_window,
                            style=style,
                            variant=variant,
                            run_id=run_id,
                            seed=seed,
                        )
                    )

    summaries = grouped_summary(rows)
    mechanical_pass = len(rows) == 240 and len(summaries) == 12 and all(
        row["mechanical_pass"] for row in summaries
    )
    summary = {
        "status": "PASS" if mechanical_pass else "FAIL_MECHANICAL",
        "batch_id": args.batch_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_cells": 240,
        "proofread_cells": len(rows),
        "expected_groups": 12,
        "proofread_groups": len(summaries),
        "mechanical_pass_groups": sum(row["mechanical_pass"] for row in summaries),
        "dct_bidirectional_pass_groups": sum(
            row["dct_bidirectional_pass"] for row in summaries
        ),
        "run_ids": run_ids,
        "groups": summaries,
    }
    write_csv(output_root / "cell_scoreboard.csv", rows)
    write_csv(output_root / "pair_window_style_summary.csv", summaries)
    write_json(output_root / "summary.json", summary)
    lines = [
        "# Cross-token proofread",
        "",
        f"- Mechanical gate: `{summary['status']}`",
        f"- Raw cells recomputed: `{len(rows)}/240`",
        f"- DCT bidirectional passes: `{summary['dct_bidirectional_pass_groups']}/12`",
        "",
        "| Pair | W | Style | Correct | Wrong | No consensus | DCT pass |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['token_pair']} | {row['memory_window']} | {row['style']} | "
            f"{row['correct']} | {row['wrong']} | {row['no_consensus']} | "
            f"{row['dct_bidirectional_pass']} |"
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "batch_id": args.batch_id,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "model_family": MODEL_FAMILY,
        "sampling": {
            "temperature": 0,
            "seed": 42,
            "sampling_overrides": "omitted",
        },
        "run_ids": run_ids,
        "source_hashes": source_hashes(
            source_paths(REPO_ROOT) + [Path(__file__).resolve()]
        ),
    }
    write_json(output_root / "run_manifest.json", manifest)
    write_json(output_root / "source_hashes.json", manifest["source_hashes"])
    indexed = build_artifact_index(roots, output_root / "artifact_index.jsonl")
    summary["indexed_artifacts"] = indexed
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if mechanical_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
