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
from utils.w0_parity_contract import VARIANT_BASES, source_paths  # noqa: E402


MODEL_FAMILY = "gemma-3-4b-it"
TOKEN_FOLDER = "tokens=0-1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def phase_root(phase: int, run_id: str, variant: str, n: int = 7) -> Path:
    if phase == 1:
        return (
            REPO_ROOT
            / "artifacts/phase1_rule_extraction/rtx5090"
            / f"n={n}"
            / MODEL_FAMILY
            / TOKEN_FOLDER
            / variant
            / run_id
        )
    if phase == 2:
        return (
            REPO_ROOT
            / "artifacts/phase2_cellular_automata/rtx5090/n=7"
            / MODEL_FAMILY
            / TOKEN_FOLDER
            / variant
            / run_id
        )
    if phase == 3:
        return (
            REPO_ROOT
            / "artifacts/phase3_memory/rtx5090/n=7"
            / MODEL_FAMILY
            / TOKEN_FOLDER
            / "W=0"
            / variant
            / run_id
        )
    raise ValueError(phase)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_seed_range(value: str) -> list[int]:
    start, end = value.split("-", 1)
    return list(range(int(start), int(end) + 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANT_BASES), default=list(VARIANT_BASES))
    parser.add_argument("--seeds", default="1-20")
    parser.add_argument("--refresh-index-only", action="store_true")
    return parser.parse_args()


def artifact_roots(run_id: str, variants: list[str], output_root: Path) -> list[Path]:
    roots = [output_root]
    for variant in variants:
        roots.extend(phase_root(1, run_id, variant, n) for n in (3, 5, 7))
        roots.extend((phase_root(2, run_id, variant), phase_root(3, run_id, variant)))
    return roots


def main() -> int:
    args = parse_args()
    seeds = parse_seed_range(args.seeds)
    output_root = (
        REPO_ROOT
        / "artifacts/cross_phase_validation/rtx5090/n=7"
        / MODEL_FAMILY
        / TOKEN_FOLDER
        / args.run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)
    roots = artifact_roots(args.run_id, args.variants, output_root)
    if args.refresh_index_only:
        count = build_artifact_index(roots, output_root / "artifact_index.jsonl")
        print(json.dumps({"indexed_artifacts": count, "run_id": args.run_id}))
        return 0
    rows: list[dict[str, Any]] = []
    first_divergences: list[dict[str, Any]] = []

    for variant in args.variants:
        phase2 = phase_root(2, args.run_id, variant)
        phase3 = phase_root(3, args.run_id, variant)
        if not (phase2 / "canonical").is_dir() or not (phase3 / "canonical").is_dir():
            raise RuntimeError(f"canonical deterministic phase output missing for {variant}")
        for seed in seeds:
            cell2 = phase2 / "canonical/cells" / f"seed_{seed:02d}"
            cell3 = phase3 / "canonical/cells" / f"seed_{seed:02d}"
            states2 = np.load(cell2 / "states.npy")
            states3 = np.load(cell3 / "states.npy")
            summary2 = read_json(cell2 / "result_summary.json")
            summary3 = read_json(cell3 / "result_summary.json")
            first_round, first_agent = first_array_difference(states2, states3)
            states_equal = bool(np.array_equal(states2, states3, equal_nan=True))
            stop_equal = (
                summary2["stop_reason"] == summary3["stop_reason"]
                and summary2["completed_transitions"] == summary3["completed_transitions"]
            )
            result_equal = (
                summary2["consensus_token"] == summary3["consensus_token"]
                and summary2["correct_consensus"] == summary3["correct_consensus"]
            )
            png_equal = summary2["png_sha256"] == summary3["png_sha256"]
            row = {
                "variant": variant,
                "seed": seed,
                "expected_token": summary2["expected_token"],
                "states_equal": states_equal,
                "stop_equal": stop_equal,
                "result_equal": result_equal,
                "png_equal": png_equal,
                "exact_parity": states_equal and stop_equal and result_equal,
                "first_diff_round": first_round,
                "first_diff_agent": first_agent,
                "automaton_stop_reason": summary2["stop_reason"],
                "online_stop_reason": summary3["stop_reason"],
                "automaton_consensus_token": summary2["consensus_token"],
                "online_consensus_token": summary3["consensus_token"],
            }
            rows.append(row)
            case_dir = output_root / "cases" / variant / f"seed_{seed:02d}"
            write_json(case_dir / "state_comparison.json", row)
            if not row["exact_parity"]:
                first_divergences.append(row)

    write_csv(output_root / "parity_scoreboard.csv", rows)
    write_csv(
        output_root / "first_divergences.csv",
        first_divergences,
        list(rows[0]) if rows else [],
    )
    gate = len(rows) == len(args.variants) * len(seeds) and all(
        row["exact_parity"] for row in rows
    )
    proofread = {
        "status": "PASS" if gate else "FAIL_PARITY",
        "expected_cases": len(args.variants) * len(seeds),
        "compared_cases": len(rows),
        "exact_parity_cases": sum(row["exact_parity"] for row in rows),
        "png_equal_cases": sum(row["png_equal"] for row in rows),
        "first_divergence": first_divergences[0] if first_divergences else None,
    }
    write_json(output_root / "proofread_summary.json", proofread)
    (output_root / "proofread_summary.md").write_text(
        "# Cross-phase W=0 proofread\n\n"
        f"- Gate: `{proofread['status']}`\n"
        f"- Exact state trajectories: `{proofread['exact_parity_cases']}/{proofread['expected_cases']}`\n"
        f"- Identical PNGs: `{proofread['png_equal_cases']}/{proofread['expected_cases']}`\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": args.run_id,
        "created_at_utc": utc_now(),
        "git_commit": git_commit(),
        "machine": "rtx5090",
        "model_family": MODEL_FAMILY,
        "tokens": ["0", "1"],
        "n_neighbors_including_center": 7,
        "variants": args.variants,
        "seeds": seeds,
        "phase_roots": {
            variant: {
                "phase1_n3": str(phase_root(1, args.run_id, variant, 3)),
                "phase1_n5": str(phase_root(1, args.run_id, variant, 5)),
                "phase1_n7": str(phase_root(1, args.run_id, variant, 7)),
                "phase2_n7": str(phase_root(2, args.run_id, variant)),
                "phase3_n7_w0": str(phase_root(3, args.run_id, variant)),
            }
            for variant in args.variants
        },
        "source_hashes": source_hashes(source_paths(REPO_ROOT) + [Path(__file__).resolve()]),
    }
    write_json(output_root / "run_manifest.json", manifest)
    write_json(output_root / "source_hashes.json", manifest["source_hashes"])
    readme_lines = [
        "# Artifact index",
        "",
        f"Run ID: `{args.run_id}`",
        "",
        "- Phase 1: deterministic frozen rules for n=3, n=5 and n=7.",
        "- Phase 2: n=7 frozen cellular automata.",
        "- Phase 3: n=7 online Gemma simulations with W=0.",
        "- `canonical/` exists only after an exact two-replay determinism gate.",
    ]
    (output_root / "README_ARTIFACTS.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    count = build_artifact_index(roots, output_root / "artifact_index.jsonl")
    proofread["indexed_artifacts"] = count
    write_json(output_root / "proofread_summary.json", proofread)
    build_artifact_index(roots, output_root / "artifact_index.jsonl")
    print(json.dumps(proofread, indent=2, ensure_ascii=False))
    return 0 if gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
