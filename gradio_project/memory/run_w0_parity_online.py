#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.initial_distribution import generate_initial_distribution  # noqa: E402
from utils.parity_artifacts import (  # noqa: E402
    atomic_append_jsonl,
    finalize_replays,
    first_array_difference,
    sha256_file,
    source_hashes,
    write_csv,
    write_json,
    write_jsonl,
)
from utils.w0_parity_contract import (  # noqa: E402
    MAX_OUTPUT_TOKENS,
    TOKEN_PAIR_SPECS,
    VARIANT_BASES,
    build_responses_payload,
    payload_hash,
    prompt_hash,
    query_responses_with_retries,
    render_memory_prompt,
    resolve_variants,
    source_paths,
    token_pair_spec,
)
from utils.parity_render import render_binary_trajectory_png  # noqa: E402


MODEL_FAMILY = "gemma-3-4b-it"
N_NEIGHBORS = 7


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def cross_root(run_id: str, token_pair: str) -> Path:
    return (
        REPO_ROOT
        / "artifacts/cross_phase_validation/rtx5090/n=7"
        / MODEL_FAMILY
        / token_pair_spec(token_pair)["folder"]
        / run_id
    )


def query_agent(
    *,
    round_idx: int,
    agent_idx: int,
    neighborhood: list[int],
    memory_snapshots: list[tuple[int, list[int]]],
    variant: str,
    base_url: str,
    model: str,
    timeout_s: int,
    max_attempts: int,
) -> dict[str, Any]:
    system_prompt, user_prompt = render_memory_prompt(
        variant,
        neighborhood,
        memory_snapshots=memory_snapshots,
    )
    payload = build_responses_payload(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        variant=variant,
    )
    query = query_responses_with_retries(
        base_url=base_url,
        payload=payload,
        variant=variant,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
    )
    return {
        "round": round_idx,
        "agent": agent_idx,
        "neighborhood": neighborhood,
        "memory_snapshots": [
            {"round": round_idx, "neighborhood": snapshot}
            for round_idx, snapshot in memory_snapshots
        ],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt_sha256": prompt_hash(system_prompt, user_prompt),
        "payload": payload,
        "payload_sha256": payload_hash(payload),
        **query,
    }


def run_cell(
    *,
    output_dir: Path,
    progress_path: Path,
    variant: str,
    token_pair: str,
    seed: int,
    base_url: str,
    model: str,
    agents: int,
    majority_ratio: float,
    memory_window: int,
    max_transitions: int,
    request_workers: int,
    timeout_s: int,
    max_attempts: int,
    replay: int,
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

    states = np.full((max_transitions + 1, agents), np.nan)
    states[0] = distribution.opinions
    requests_path = cell_dir / "requests.jsonl"
    normalized_rows: list[dict[str, Any]] = []
    request_failures = 0
    parse_failures = 0
    completed_transitions = 0
    stop_reason = "max_rounds"
    half = N_NEIGHBORS // 2

    with ThreadPoolExecutor(max_workers=min(request_workers, agents)) as pool:
        for transition in range(1, max_transitions + 1):
            previous = states[transition - 1].astype(np.int8)
            future_to_agent = {}
            for agent in range(agents):
                neighborhood = [
                    int(previous[(agent + offset) % agents])
                    for offset in range(-half, half + 1)
                ]
                memory_snapshots = []
                for memory_round in range(max(0, transition - memory_window), transition):
                    memory_state = states[memory_round].astype(np.int8)
                    memory_snapshots.append(
                        (
                            memory_round,
                            [
                                int(memory_state[(agent + offset) % agents])
                                for offset in range(-half, half + 1)
                            ],
                        )
                    )
                future = pool.submit(
                    query_agent,
                    round_idx=transition,
                    agent_idx=agent,
                    neighborhood=neighborhood,
                    memory_snapshots=memory_snapshots,
                    variant=variant,
                    base_url=base_url,
                    model=model,
                    timeout_s=timeout_s,
                    max_attempts=max_attempts,
                )
                future_to_agent[future] = agent

            round_records = [future.result() for future in as_completed(future_to_agent)]
            round_records.sort(key=lambda item: item["agent"])
            with requests_path.open("a", encoding="utf-8") as handle:
                for record in round_records:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            request_failures += sum(record["request_failures"] for record in round_records)
            parse_failures += sum(record["parse_failures"] for record in round_records)
            invalid = [record for record in round_records if record["choice"] is None]
            if invalid:
                raise RuntimeError(
                    f"terminal parse/request failure at transition={transition}, agents="
                    f"{[record['agent'] for record in invalid]}"
                )
            states[transition] = [record["choice"] for record in round_records]
            completed_transitions = transition
            normalized_rows.extend(
                {
                    "round": record["round"],
                    "agent": record["agent"],
                    "neighborhood": record["neighborhood"],
                    "memory_snapshots": record["memory_snapshots"],
                    "prompt_sha256": record["prompt_sha256"],
                    "payload_sha256": record["payload_sha256"],
                    "raw_response_sha256": record["raw_response_sha256"],
                    "choice": record["choice"],
                }
                for record in round_records
            )
            atomic_append_jsonl(
                progress_path,
                {
                    "event": "round_completed",
                    "phase": 3,
                    "replay": replay,
                    "variant": variant,
                    "seed": seed,
                    "transition": transition,
                },
            )
            if np.all(states[transition] == states[transition, 0]):
                stop_reason = "consensus"
                break

    normalized_path = cell_dir / "normalized_log.jsonl"
    write_jsonl(normalized_path, normalized_rows)
    states_path = cell_dir / "states.npy"
    np.save(states_path, states)
    valid_states = states[: completed_transitions + 1]
    png_path = cell_dir / "simulation.png"
    png_path.write_bytes(render_binary_trajectory_png(valid_states))
    final = valid_states[-1].astype(np.int8)
    consensus = bool(np.all(final == final[0]))
    tokens = token_pair_spec(token_pair)["tokens"]
    consensus_state = int(final[0]) if consensus else None
    consensus_token = tokens[consensus_state] if consensus_state is not None else None
    expected_state = int(distribution.initial_majority_opinion)
    expected_token = tokens[expected_state]
    summary = {
        "variant": variant,
        "token_pair": token_pair,
        "tokens": list(tokens),
        "seed": seed,
        "model_family": MODEL_FAMILY,
        "served_model": model,
        "agents": agents,
        "n_neighbors_including_center": N_NEIGHBORS,
        "memory_window": memory_window,
        "majority_ratio": majority_ratio,
        "initial_count_0": distribution.count_0,
        "initial_count_1": distribution.count_1,
        "expected_token": expected_token,
        "expected_state": expected_state,
        "max_transitions": max_transitions,
        "completed_transitions": completed_transitions,
        "completed_rounds_including_initial": completed_transitions + 1,
        "stop_reason": stop_reason,
        "consensus": consensus,
        "consensus_token": consensus_token,
        "consensus_state": consensus_state,
        "correct_consensus": consensus_state == expected_state,
        "temperature": 0.0,
        "request_seed": 42,
        "max_output_tokens": MAX_OUTPUT_TOKENS[variant],
        "request_failures": request_failures,
        "parse_failures": parse_failures,
        "states_sha256": sha256_file(states_path),
        "png_sha256": sha256_file(png_path),
        "normalized_log_sha256": sha256_file(normalized_path),
        "requests_sha256": sha256_file(requests_path),
    }
    write_json(cell_dir / "result_summary.json", summary)
    return summary


def run_replay(
    *,
    output_dir: Path,
    progress_path: Path,
    run_id: str,
    variant: str,
    token_pair: str,
    replay: int,
    seeds: list[int],
    base_url: str,
    model: str,
    agents: int,
    majority_ratio: float,
    memory_window: int,
    max_transitions: int,
    request_workers: int,
    timeout_s: int,
    max_attempts: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for seed in seeds:
        try:
            row = run_cell(
                output_dir=output_dir,
                progress_path=progress_path,
                variant=variant,
                token_pair=token_pair,
                seed=seed,
                base_url=base_url,
                model=model,
                agents=agents,
                majority_ratio=majority_ratio,
                memory_window=memory_window,
                max_transitions=max_transitions,
                request_workers=request_workers,
                timeout_s=timeout_s,
                max_attempts=max_attempts,
                replay=replay,
            )
            rows.append(row)
            event = {"event": "cell_completed", "phase": 3, "replay": replay, **row}
        except Exception as exc:
            error = {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}
            errors.append(error)
            cell_dir = output_dir / "cells" / f"seed_{seed:02d}"
            write_json(cell_dir / "error.json", error)
            event = {"event": "cell_error", "phase": 3, "replay": replay, "variant": variant, **error}
        atomic_append_jsonl(progress_path, event)
        print(json.dumps(event, ensure_ascii=False), flush=True)

    write_csv(output_dir / "scoreboard.csv", rows)
    manifest = {
        "run_id": run_id,
        "replay": replay,
        "created_at_utc": utc_now(),
        "git_commit": git_commit(),
        "mechanism": "online_llm_memory",
        "model_family": MODEL_FAMILY,
        "served_model": model,
        "base_url": base_url,
        "variant": variant,
        "token_pair": token_pair,
        "historical_base_variant": VARIANT_BASES[variant],
        "tokens": list(token_pair_spec(token_pair)["tokens"]),
        "n_neighbors_including_center": N_NEIGHBORS,
        "memory_window": memory_window,
        "agents": agents,
        "seeds": seeds,
        "majority_ratio": majority_ratio,
        "max_transitions": max_transitions,
        "stop_conditions": ["consensus", "max_rounds"],
        "sampling": {
            "temperature": 0.0,
            "seed": 42,
            "max_output_tokens": MAX_OUTPUT_TOKENS[variant],
            "sampling_overrides": "omitted",
        },
        "request_workers": request_workers,
        "source_hashes": source_hashes(source_paths(REPO_ROOT) + [Path(__file__).resolve()]),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(output_dir / "source_hashes.json", manifest["source_hashes"])
    proofread = {
        "completed_cells": len(rows),
        "expected_cells": len(seeds),
        "error_cells": len(errors),
        "request_failures": sum(row["request_failures"] for row in rows),
        "parse_failures": sum(row["parse_failures"] for row in rows),
        "errors": errors,
    }
    write_json(output_dir / "proofread_summary.json", proofread)
    (output_dir / "proofread_summary.md").write_text(
        f"# Phase 3 proofread\n\n- Cells: `{len(rows)}/{len(seeds)}`\n"
        f"- Errors: `{len(errors)}`\n- Request failures: `{proofread['request_failures']}`\n"
        f"- Parse failures: `{proofread['parse_failures']}`\n",
        encoding="utf-8",
    )


def load_summary(root: Path, seed: int) -> dict[str, Any] | None:
    path = root / "cells" / f"seed_{seed:02d}" / "result_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def compare_replays(replay1: Path, replay2: Path, proof_dir: Path, seeds: list[int]) -> dict[str, Any]:
    proof_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        summary1 = load_summary(replay1, seed)
        summary2 = load_summary(replay2, seed)
        if summary1 is None or summary2 is None:
            rows.append(
                {
                    "seed": seed,
                    "states_equal": False,
                    "summary_equal": False,
                    "png_equal": False,
                    "normalized_log_equal": False,
                    "responses_letter_exact": False,
                    "exact_equal": False,
                    "first_diff_round": None,
                    "first_diff_agent": None,
                    "missing_replay1": summary1 is None,
                    "missing_replay2": summary2 is None,
                }
            )
            continue
        cell1 = replay1 / "cells" / f"seed_{seed:02d}"
        cell2 = replay2 / "cells" / f"seed_{seed:02d}"
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
                "request_failures",
                "parse_failures",
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
                "responses_letter_exact": log_equal,
                "exact_equal": states_equal and summary_equal and png_equal and log_equal,
                "first_diff_round": first_round,
                "first_diff_agent": first_agent,
                "missing_replay1": False,
                "missing_replay2": False,
            }
        )
    write_csv(proof_dir / "paired_hashes.csv", rows)
    deterministic = len(rows) == len(seeds) and all(row["exact_equal"] for row in rows)
    mismatches = [row for row in rows if not row["exact_equal"]]
    summary = {
        "deterministic": deterministic,
        "expected_cells": len(seeds),
        "exact_cells": sum(row["exact_equal"] for row in rows),
        "letter_exact_cells": sum(row["responses_letter_exact"] for row in rows),
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
    parser.add_argument("--base-url", default="http://127.0.0.1:8127/v1")
    parser.add_argument("--model", default="gemma3-4b-temp0")
    parser.add_argument("--token-pair", choices=list(TOKEN_PAIR_SPECS), default="01")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--seeds", default="1-20")
    parser.add_argument("--majority-ratio", type=float, default=0.51)
    parser.add_argument("--memory-window", type=int, default=0)
    parser.add_argument("--max-transitions", type=int, default=60)
    parser.add_argument("--request-workers", type=int, default=30)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.memory_window < 0:
        raise ValueError("--memory-window must be non-negative")
    seeds = parse_seed_range(args.seeds)
    variants = resolve_variants(args.token_pair, args.variants)
    spec = token_pair_spec(args.token_pair)
    progress_path = cross_root(args.run_id, args.token_pair) / "progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for variant in variants:
        work_root = (
            REPO_ROOT
            / "artifacts/work/rtx5090/n=7"
            / args.run_id
            / "phase3"
            / f"W={args.memory_window}"
            / args.token_pair
            / variant
        )
        final_root = (
            REPO_ROOT
            / "artifacts/phase3_memory/rtx5090/n=7"
            / MODEL_FAMILY
            / spec["folder"]
            / f"W={args.memory_window}"
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
                progress_path=progress_path,
                run_id=args.run_id,
                variant=variant,
                token_pair=args.token_pair,
                replay=replay,
                seeds=seeds,
                base_url=args.base_url,
                model=args.model,
                agents=args.agents,
                majority_ratio=args.majority_ratio,
                memory_window=args.memory_window,
                max_transitions=args.max_transitions,
                request_workers=args.request_workers,
                timeout_s=args.timeout_s,
                max_attempts=args.max_attempts,
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
                "token_pair": args.token_pair,
                "n_neighbors": N_NEIGHBORS,
                "memory_window": args.memory_window,
                "completed_at_utc": utc_now(),
            },
        )
        results.append({"variant": variant, **proof})
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if all(result["deterministic"] for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
