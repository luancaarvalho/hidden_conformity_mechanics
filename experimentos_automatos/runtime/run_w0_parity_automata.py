#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.initial_distribution import generate_initial_distribution  # noqa: E402
from utils.parity_artifacts import (  # noqa: E402
    finalize_replays,
    first_array_difference,
    sha256_file,
    source_hashes,
    write_csv,
    write_json,
    write_jsonl,
)
from utils.w0_parity_contract import VARIANT_BASES, source_paths  # noqa: E402
from utils.parity_render import render_binary_trajectory_png  # noqa: E402


MODEL_FAMILY = "gemma-3-4b-it"
TOKEN_FOLDER = "tokens=0-1"
N_NEIGHBORS = 7


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def canonical_rule_root(run_id: str, variant: str) -> Path:
    return (
        REPO_ROOT
        / "artifacts/phase1_rule_extraction/rtx5090"
        / f"n={N_NEIGHBORS}"
        / MODEL_FAMILY
        / TOKEN_FOLDER
        / variant
        / run_id
        / "canonical"
    )


def load_rules(rule_csv: Path) -> dict[str, int]:
    with rule_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rules = {row["configuracao_binaria"]: int(row["escolha"]) for row in rows}
    if len(rules) != 2**N_NEIGHBORS or any(len(key) != N_NEIGHBORS for key in rules):
        raise ValueError(f"invalid n=7 rule table: {rule_csv}")
    return rules


def evolve(initial: np.ndarray, rules: dict[str, int], max_transitions: int) -> tuple[np.ndarray, int, str]:
    states = np.full((max_transitions + 1, len(initial)), np.nan)
    states[0] = initial
    half = N_NEIGHBORS // 2
    completed = 0
    stop_reason = "max_rounds"
    for transition in range(1, max_transitions + 1):
        previous = states[transition - 1].astype(np.int8)
        next_state = np.empty_like(previous)
        for agent in range(len(previous)):
            config = "".join(
                str(int(previous[(agent + offset) % len(previous)]))
                for offset in range(-half, half + 1)
            )
            next_state[agent] = rules[config]
        states[transition] = next_state
        completed = transition
        if np.all(next_state == next_state[0]):
            stop_reason = "consensus"
            break
    return states, completed, stop_reason


def run_cell(
    *,
    output_dir: Path,
    variant: str,
    seed: int,
    rules: dict[str, int],
    agents: int,
    majority_ratio: float,
    max_transitions: int,
) -> dict[str, Any]:
    cell_dir = output_dir / "cells" / f"seed_{seed:02d}"
    cell_dir.mkdir(parents=True, exist_ok=False)
    distribution = generate_initial_distribution(
        n_agents=agents,
        seed_distribution=seed,
        majority_ratio=majority_ratio,
        mode="ratio",
    )
    initial_path = output_dir / "initial_states" / f"seed_{seed:02d}.npy"
    initial_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(initial_path, distribution.opinions)

    states, completed_transitions, stop_reason = evolve(
        distribution.opinions, rules, max_transitions
    )
    states_path = cell_dir / "states.npy"
    np.save(states_path, states)
    valid_states = states[: completed_transitions + 1]
    png_path = cell_dir / "simulation.png"
    png_path.write_bytes(render_binary_trajectory_png(valid_states))
    log_rows = [
        {
            "transition": transition,
            "state": [int(value) for value in valid_states[transition]],
        }
        for transition in range(len(valid_states))
    ]
    log_path = cell_dir / "normalized_log.jsonl"
    write_jsonl(log_path, log_rows)

    final = valid_states[-1].astype(np.int8)
    consensus = bool(np.all(final == final[0]))
    consensus_token = str(int(final[0])) if consensus else None
    expected_token = str(distribution.initial_majority_opinion)
    summary = {
        "variant": variant,
        "seed": seed,
        "agents": agents,
        "n_neighbors_including_center": N_NEIGHBORS,
        "majority_ratio": majority_ratio,
        "initial_count_0": distribution.count_0,
        "initial_count_1": distribution.count_1,
        "expected_token": expected_token,
        "max_transitions": max_transitions,
        "completed_transitions": completed_transitions,
        "completed_rounds_including_initial": completed_transitions + 1,
        "stop_reason": stop_reason,
        "consensus": consensus,
        "consensus_token": consensus_token,
        "correct_consensus": consensus_token == expected_token,
        "states_sha256": sha256_file(states_path),
        "png_sha256": sha256_file(png_path),
        "normalized_log_sha256": sha256_file(log_path),
    }
    write_json(cell_dir / "result_summary.json", summary)
    return summary


def run_replay(
    *,
    output_dir: Path,
    run_id: str,
    variant: str,
    replay: int,
    agents: int,
    seeds: list[int],
    majority_ratio: float,
    max_transitions: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    rule_root = canonical_rule_root(run_id, variant)
    if not rule_root.is_dir():
        raise FileNotFoundError(f"canonical deterministic rule not found: {rule_root}")
    input_rule = output_dir / "input_rule"
    shutil.copytree(rule_root, input_rule)
    rules = load_rules(input_rule / "rule.csv")
    rows = [
        run_cell(
            output_dir=output_dir,
            variant=variant,
            seed=seed,
            rules=rules,
            agents=agents,
            majority_ratio=majority_ratio,
            max_transitions=max_transitions,
        )
        for seed in seeds
    ]
    write_csv(output_dir / "scoreboard.csv", rows)
    manifest = {
        "run_id": run_id,
        "replay": replay,
        "created_at_utc": utc_now(),
        "git_commit": git_commit(),
        "mechanism": "frozen_cellular_automaton",
        "variant": variant,
        "tokens": ["0", "1"],
        "n_neighbors_including_center": N_NEIGHBORS,
        "agents": agents,
        "seeds": seeds,
        "majority_ratio": majority_ratio,
        "max_transitions": max_transitions,
        "stop_conditions": ["consensus", "max_rounds"],
        "input_rule_sha256": sha256_file(input_rule / "rule.csv"),
        "source_hashes": source_hashes(source_paths(REPO_ROOT) + [Path(__file__).resolve()]),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(output_dir / "source_hashes.json", manifest["source_hashes"])
    write_json(
        output_dir / "proofread_summary.json",
        {
            "cells": len(rows),
            "expected_cells": len(seeds),
            "request_failures": 0,
            "missing_cells": len(seeds) - len(rows),
        },
    )
    (output_dir / "proofread_summary.md").write_text(
        f"# Phase 2 proofread\n\n- Cells: `{len(rows)}/{len(seeds)}`\n- Rule: `{manifest['input_rule_sha256']}`\n",
        encoding="utf-8",
    )


def load_summary(root: Path, seed: int) -> dict[str, Any]:
    return json.loads(
        (root / "cells" / f"seed_{seed:02d}" / "result_summary.json").read_text(
            encoding="utf-8"
        )
    )


def compare_replays(replay1: Path, replay2: Path, proof_dir: Path, seeds: list[int]) -> dict[str, Any]:
    proof_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        cell1 = replay1 / "cells" / f"seed_{seed:02d}"
        cell2 = replay2 / "cells" / f"seed_{seed:02d}"
        summary1 = load_summary(replay1, seed)
        summary2 = load_summary(replay2, seed)
        states1 = np.load(cell1 / "states.npy")
        states2 = np.load(cell2 / "states.npy")
        first_round, first_agent = first_array_difference(states1, states2)
        states_equal = bool(np.array_equal(states1, states2, equal_nan=True))
        summary_equal = all(
            summary1[key] == summary2[key]
            for key in (
                "stop_reason",
                "completed_transitions",
                "consensus_token",
                "correct_consensus",
            )
        )
        png_equal = summary1["png_sha256"] == summary2["png_sha256"]
        log_equal = summary1["normalized_log_sha256"] == summary2["normalized_log_sha256"]
        rows.append(
            {
                "seed": seed,
                "states_equal": states_equal,
                "summary_equal": summary_equal,
                "png_equal": png_equal,
                "normalized_log_equal": log_equal,
                "exact_equal": states_equal and summary_equal and png_equal and log_equal,
                "states_sha256_replay1": summary1["states_sha256"],
                "states_sha256_replay2": summary2["states_sha256"],
                "first_diff_round": first_round,
                "first_diff_agent": first_agent,
            }
        )
    write_csv(proof_dir / "paired_hashes.csv", rows)
    deterministic = len(rows) == len(seeds) and all(row["exact_equal"] for row in rows)
    mismatches = [row for row in rows if not row["exact_equal"]]
    summary = {
        "deterministic": deterministic,
        "expected_cells": len(seeds),
        "exact_cells": sum(row["exact_equal"] for row in rows),
        "first_divergence": mismatches[0] if mismatches else None,
    }
    write_json(proof_dir / "comparison_summary.json", summary)
    if mismatches:
        write_json(proof_dir / "first_divergence.json", mismatches[0])
    return summary


def parse_seed_range(value: str) -> list[int]:
    start, end = value.split("-", 1)
    return list(range(int(start), int(end) + 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANT_BASES), default=list(VARIANT_BASES))
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--seeds", default="1-20")
    parser.add_argument("--majority-ratio", type=float, default=0.51)
    parser.add_argument("--max-transitions", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_range(args.seeds)
    results: list[dict[str, Any]] = []
    for variant in args.variants:
        work_root = (
            REPO_ROOT
            / "artifacts/work/rtx5090/n=7"
            / args.run_id
            / "phase2"
            / variant
        )
        final_root = (
            REPO_ROOT
            / "artifacts/phase2_cellular_automata/rtx5090/n=7"
            / MODEL_FAMILY
            / TOKEN_FOLDER
            / variant
            / args.run_id
        )
        if work_root.exists() or final_root.exists():
            raise FileExistsError(work_root if work_root.exists() else final_root)
        replay1 = work_root / "replay_01"
        replay2 = work_root / "replay_02"
        for replay, output in ((1, replay1), (2, replay2)):
            run_replay(
                output_dir=output,
                run_id=args.run_id,
                variant=variant,
                replay=replay,
                agents=args.agents,
                seeds=seeds,
                majority_ratio=args.majority_ratio,
                max_transitions=args.max_transitions,
            )
        proof = compare_replays(replay1, replay2, work_root / "proof", seeds)
        finalize_replays(
            replay1=replay1,
            replay2=replay2,
            proof_dir=work_root / "proof",
            final_root=final_root,
            deterministic=proof["deterministic"],
            status_details={
                "run_id": args.run_id,
                "variant": variant,
                "n_neighbors": N_NEIGHBORS,
                "completed_at_utc": utc_now(),
            },
        )
        results.append({"variant": variant, **proof})
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if all(result["deterministic"] for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
