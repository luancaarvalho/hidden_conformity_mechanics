#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.parity_artifacts import (  # noqa: E402
    build_artifact_index,
    first_array_difference,
    source_hashes,
    write_csv,
    write_json,
)
from utils.w0_parity_contract import (  # noqa: E402
    TOKEN_PAIR_SPECS,
    resolve_variants,
    source_paths,
    token_pair_spec,
)


MODEL_FAMILY = "gemma-3-4b-it"


def phase3_root(
    run_id: str, variant: str, memory_window: int, token_pair: str
) -> Path:
    return (
        REPO_ROOT
        / "artifacts/phase3_memory/rtx5090/n=7"
        / MODEL_FAMILY
        / token_pair_spec(token_pair)["folder"]
        / f"W={memory_window}"
        / variant
        / run_id
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_seed_range(value: str) -> list[int]:
    start, end = value.split("-", 1)
    return list(range(int(start), int(end) + 1))


def classification(summary: dict[str, Any]) -> str:
    if not summary["consensus"]:
        return "no_consensus"
    return "correct" if summary["correct_consensus"] else "wrong"


def summarize(
    rows: list[dict[str, Any]], prefix: str, tokens: tuple[str, str]
) -> dict[str, Any]:
    return {
        "correct": sum(row[f"{prefix}_classification"] == "correct" for row in rows),
        "wrong": sum(row[f"{prefix}_classification"] == "wrong" for row in rows),
        "no_consensus": sum(
            row[f"{prefix}_classification"] == "no_consensus" for row in rows
        ),
        "consensus_token0": sum(
            row[f"{prefix}_consensus_token"] == tokens[0] for row in rows
        ),
        "consensus_token1": sum(
            row[f"{prefix}_consensus_token"] == tokens[1] for row in rows
        ),
        "mean_completed_transitions": (
            sum(row[f"{prefix}_completed_transitions"] for row in rows) / len(rows)
            if rows
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--memory-window", type=int, required=True)
    parser.add_argument("--token-pair", choices=list(TOKEN_PAIR_SPECS), default="01")
    parser.add_argument("--seeds", default="1-20")
    parser.add_argument(
        "--variants", nargs="+"
    )
    args = parser.parse_args()
    if args.memory_window < 1:
        raise ValueError("comparison memory window must be at least 1")

    seeds = parse_seed_range(args.seeds)
    variants = list(resolve_variants(args.token_pair, args.variants))
    spec = token_pair_spec(args.token_pair)
    output_root = (
        REPO_ROOT
        / "artifacts/cross_phase_validation/rtx5090/n=7"
        / MODEL_FAMILY
        / spec["folder"]
        / args.run_id
        / f"memory_impact_W{args.memory_window}_vs_W0"
    )
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    artifact_roots = [output_root]
    for variant in variants:
        baseline_root = phase3_root(args.baseline_run_id, variant, 0, args.token_pair)
        memory_root = phase3_root(
            args.run_id, variant, args.memory_window, args.token_pair
        )
        artifact_roots.extend((baseline_root, memory_root))
        if not (baseline_root / "canonical").is_dir():
            raise RuntimeError(f"canonical W=0 baseline missing: {baseline_root}")
        if not (memory_root / "canonical").is_dir():
            raise RuntimeError(f"canonical memory run missing: {memory_root}")

        variant_rows: list[dict[str, Any]] = []
        for seed in seeds:
            baseline_cell = baseline_root / "canonical/cells" / f"seed_{seed:02d}"
            memory_cell = memory_root / "canonical/cells" / f"seed_{seed:02d}"
            baseline_summary = read_json(baseline_cell / "result_summary.json")
            memory_summary = read_json(memory_cell / "result_summary.json")
            baseline_states = np.load(baseline_cell / "states.npy")
            memory_states = np.load(memory_cell / "states.npy")
            first_round, first_agent = first_array_difference(
                baseline_states, memory_states
            )
            row = {
                "variant": variant,
                "seed": seed,
                "expected_token": baseline_summary["expected_token"],
                "initial_state_equal": bool(
                    np.array_equal(baseline_states[0], memory_states[0])
                ),
                "trajectory_equal": bool(
                    np.array_equal(baseline_states, memory_states, equal_nan=True)
                ),
                "first_diff_round": first_round,
                "first_diff_agent": first_agent,
                "w0_classification": classification(baseline_summary),
                "w1_classification": classification(memory_summary),
                "w0_consensus_token": baseline_summary["consensus_token"],
                "w1_consensus_token": memory_summary["consensus_token"],
                "w0_completed_transitions": baseline_summary["completed_transitions"],
                "w1_completed_transitions": memory_summary["completed_transitions"],
                "transition_delta": (
                    memory_summary["completed_transitions"]
                    - baseline_summary["completed_transitions"]
                ),
                "classification_changed": (
                    classification(baseline_summary) != classification(memory_summary)
                ),
                "improved": (
                    classification(baseline_summary) != "correct"
                    and classification(memory_summary) == "correct"
                ),
                "regressed": (
                    classification(baseline_summary) == "correct"
                    and classification(memory_summary) != "correct"
                ),
                "w1_request_failures": memory_summary["request_failures"],
                "w1_parse_failures": memory_summary["parse_failures"],
            }
            variant_rows.append(row)
            all_rows.append(row)

        determinism = read_json(memory_root / "determinism/comparison_summary.json")
        summaries[variant] = {
            "w0": summarize(variant_rows, "w0", spec["tokens"]),
            "w1": summarize(variant_rows, "w1", spec["tokens"]),
            "initial_states_equal": sum(row["initial_state_equal"] for row in variant_rows),
            "trajectories_equal": sum(row["trajectory_equal"] for row in variant_rows),
            "classification_changes": sum(
                row["classification_changed"] for row in variant_rows
            ),
            "improved": sum(row["improved"] for row in variant_rows),
            "regressed": sum(row["regressed"] for row in variant_rows),
            "request_failures": sum(row["w1_request_failures"] for row in variant_rows),
            "parse_failures": sum(row["w1_parse_failures"] for row in variant_rows),
            "determinism": determinism,
        }

    write_csv(output_root / "w1_vs_w0_scoreboard.csv", all_rows)
    summary = {
        "status": "PASS",
        "baseline_run_id": args.baseline_run_id,
        "memory_run_id": args.run_id,
        "memory_window": args.memory_window,
        "token_pair": args.token_pair,
        "tokens": list(spec["tokens"]),
        "expected_cases": len(variants) * len(seeds),
        "compared_cases": len(all_rows),
        "memory_semantics": (
            "W=1 repeats the immediately preceding local snapshot, which is also the "
            "current neighborhood used for the next synchronous decision."
        ),
        "variants": summaries,
    }
    if len(all_rows) != summary["expected_cases"] or not all(
        row["initial_state_equal"] for row in all_rows
    ):
        summary["status"] = "FAIL_COMPARISON"
    write_json(output_root / "summary.json", summary)
    lines = [
        "# W=1 versus W=0",
        "",
        f"- Gate: `{summary['status']}`",
        f"- Compared cases: `{len(all_rows)}/{summary['expected_cases']}`",
        "- W=1 historical semantics: repeats the immediately preceding local snapshot.",
    ]
    for variant, values in summaries.items():
        lines.extend(
            [
                "",
                f"## {variant}",
                f"- W=0: `{values['w0']}`",
                f"- W=1: `{values['w1']}`",
                f"- Classification changes: `{values['classification_changes']}`",
                f"- Improved/regressed: `{values['improved']}/{values['regressed']}`",
                f"- Deterministic replay cells: `{values['determinism']['exact_cells']}/{values['determinism']['expected_cells']}`",
            ]
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "baseline_run_id": args.baseline_run_id,
        "memory_run_id": args.run_id,
        "memory_window": args.memory_window,
        "token_pair": args.token_pair,
        "tokens": list(spec["tokens"]),
        "variants": variants,
        "seeds": seeds,
        "source_hashes": source_hashes(
            source_paths(REPO_ROOT) + [Path(__file__).resolve()]
        ),
    }
    write_json(output_root / "run_manifest.json", manifest)
    write_json(output_root / "source_hashes.json", manifest["source_hashes"])
    count = build_artifact_index(
        artifact_roots, output_root / "artifact_index.jsonl"
    )
    summary["indexed_artifacts"] = count
    write_json(output_root / "summary.json", summary)
    build_artifact_index(artifact_roots, output_root / "artifact_index.jsonl")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
