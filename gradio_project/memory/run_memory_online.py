#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests


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
from utils.parity_render import render_binary_trajectory_png  # noqa: E402
from utils.w0_parity_contract import (  # noqa: E402
    MAX_OUTPUT_TOKENS,
    VARIANT_BASES,
    build_responses_payload,
    payload_hash,
    prompt_hash,
    query_responses_with_retries,
    render_memory_prompt,
    source_paths,
    token_pair_spec,
)


MODEL_FAMILY = "gemma-3-4b-it"
N_NEIGHBORS = 7


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    run_id: str
    token_pair: str
    variant: str
    memory_window: int
    seed: int

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "CellSpec":
        spec = cls(
            cell_id=str(row["cell_id"]),
            run_id=str(row["run_id"]),
            token_pair=str(row["token_pair"]),
            variant=str(row["variant"]),
            memory_window=int(row["memory_window"]),
            seed=int(row["seed"]),
        )
        if spec.memory_window < 0 or spec.seed < 1:
            raise ValueError(f"invalid cell spec: {row}")
        expected = token_pair_spec(spec.token_pair)["variants"]
        if spec.variant not in expected:
            raise ValueError(f"variant {spec.variant} does not belong to {spec.token_pair}")
        return spec


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def read_manifest(path: Path) -> list[CellSpec]:
    rows = [
        CellSpec.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({row.cell_id for row in rows}) != len(rows):
        raise ValueError("manifest must contain unique cells")
    return rows


def group_key(spec: CellSpec) -> tuple[int, str, str, str]:
    return spec.memory_window, spec.token_pair, spec.variant, spec.run_id


def group_root(work_root: Path, spec: CellSpec) -> Path:
    return (
        work_root
        / f"W={spec.memory_window}"
        / spec.token_pair
        / spec.variant
    )


def final_root(spec: CellSpec) -> Path:
    return (
        REPO_ROOT
        / "artifacts/phase3_memory/rtx5090/n=7"
        / MODEL_FAMILY
        / token_pair_spec(spec.token_pair)["folder"]
        / f"W={spec.memory_window}"
        / spec.variant
        / spec.run_id
    )


def replay_root(work_root: Path, spec: CellSpec, replay: int) -> Path:
    return group_root(work_root, spec) / f"replay_{replay:02d}"


def cell_root(work_root: Path, spec: CellSpec, replay: int) -> Path:
    return replay_root(work_root, spec, replay) / "cells" / f"seed_{spec.seed:02d}"


def wait_for_endpoint(
    base_url: str, fatal_endpoint: threading.Event, timeout_s: int = 1800
) -> None:
    if fatal_endpoint.is_set():
        raise RuntimeError("FAIL_ENDPOINT: endpoint already marked unavailable")
    deadline = time.monotonic() + timeout_s
    delay = 5
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url.rstrip('/')}/models", timeout=10)
            response.raise_for_status()
            return
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 60)
    fatal_endpoint.set()
    raise RuntimeError("FAIL_ENDPOINT: endpoint unavailable for 30 minutes")


def query_agent(
    *,
    transition: int,
    agent: int,
    neighborhood: list[int],
    memory_snapshots: list[tuple[int, list[int]]],
    spec: CellSpec,
    base_url: str,
    model: str,
    timeout_s: int,
    max_attempts: int,
) -> dict[str, Any]:
    system_prompt, user_prompt = render_memory_prompt(
        spec.variant, neighborhood, memory_snapshots=memory_snapshots
    )
    payload = build_responses_payload(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        variant=spec.variant,
    )
    result = query_responses_with_retries(
        base_url=base_url,
        payload=payload,
        variant=spec.variant,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
    )
    return {
        "round": transition,
        "agent": agent,
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
        **result,
    }


def completed_cell_is_valid(path: Path, spec: CellSpec, replay: int) -> bool:
    required = (
        path / "states.npy",
        path / "requests.jsonl",
        path / "normalized_log.jsonl",
        path / "simulation.png",
        path / "result_summary.json",
    )
    if not all(item.is_file() for item in required):
        return False
    try:
        summary = json.loads(required[-1].read_text(encoding="utf-8"))
        return all(
            (
                summary["seed"] == spec.seed,
                summary["cell_id"] == spec.cell_id,
                summary["run_id"] == spec.run_id,
                summary["token_pair"] == spec.token_pair,
                summary["variant"] == spec.variant,
                summary["memory_window"] == spec.memory_window,
                summary["replay"] == replay,
                summary["agents"] == 30,
                summary["n_neighbors_including_center"] == N_NEIGHBORS,
                summary["max_transitions"] == 60,
                summary["states_sha256"] == sha256_file(required[0]),
                summary["requests_sha256"] == sha256_file(required[1]),
                summary["normalized_log_sha256"] == sha256_file(required[2]),
                summary["png_sha256"] == sha256_file(required[3]),
            )
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def isolate_partial(path: Path, work_root: Path, spec: CellSpec, replay: int) -> None:
    if not path.exists():
        return
    quarantine = (
        work_root
        / "quarantine"
        / f"replay_{replay:02d}"
        / f"{spec.cell_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(quarantine))


def run_cell(
    *,
    spec: CellSpec,
    replay: int,
    output_dir: Path,
    progress_path: Path,
    request_pool: ThreadPoolExecutor,
    fatal_endpoint: threading.Event,
    base_url: str,
    model: str,
    agents: int,
    majority_ratio: float,
    max_transitions: int,
    timeout_s: int,
    max_attempts: int,
) -> dict[str, Any]:
    wait_for_endpoint(base_url, fatal_endpoint)
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_append_jsonl(
        progress_path,
        {
            "utc": utc_now(),
            "event": "cell_running",
            "status": "running",
            "replay": replay,
            **asdict(spec),
        },
    )
    distribution = generate_initial_distribution(
        n_agents=agents,
        seed_distribution=spec.seed,
        majority_ratio=majority_ratio,
        mode="ratio",
    )
    states = np.full((max_transitions + 1, agents), np.nan)
    states[0] = distribution.opinions
    requests_path = output_dir / "requests.jsonl"
    normalized_rows: list[dict[str, Any]] = []
    request_failures = 0
    parse_failures = 0
    completed_transitions = 0
    stop_reason = "max_rounds"
    half = N_NEIGHBORS // 2

    for transition in range(1, max_transitions + 1):
        wait_for_endpoint(base_url, fatal_endpoint)
        previous = states[transition - 1].astype(np.int8)
        futures: list[Future[dict[str, Any]]] = []
        for agent in range(agents):
            neighborhood = [
                int(previous[(agent + offset) % agents])
                for offset in range(-half, half + 1)
            ]
            memory_snapshots = []
            for memory_round in range(
                max(0, transition - spec.memory_window), transition
            ):
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
            futures.append(
                request_pool.submit(
                    query_agent,
                    transition=transition,
                    agent=agent,
                    neighborhood=neighborhood,
                    memory_snapshots=memory_snapshots,
                    spec=spec,
                    base_url=base_url,
                    model=model,
                    timeout_s=timeout_s,
                    max_attempts=max_attempts,
                )
            )

        records = sorted((future.result() for future in futures), key=lambda row: row["agent"])
        with requests_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        request_failures += sum(record["request_failures"] for record in records)
        parse_failures += sum(record["parse_failures"] for record in records)
        invalid = [record["agent"] for record in records if record["choice"] is None]
        if invalid:
            raise RuntimeError(
                f"terminal request/parse failure transition={transition} agents={invalid}"
            )
        states[transition] = [record["choice"] for record in records]
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
            for record in records
        )
        atomic_append_jsonl(
            progress_path,
            {
                "utc": utc_now(),
                "event": "round_completed",
                "replay": replay,
                "cell_id": spec.cell_id,
                "transition": transition,
            },
        )
        if np.all(states[transition] == states[transition, 0]):
            stop_reason = "consensus"
            break

    normalized_path = output_dir / "normalized_log.jsonl"
    write_jsonl(normalized_path, normalized_rows)
    states_path = output_dir / "states.npy"
    np.save(states_path, states)
    valid_states = states[: completed_transitions + 1]
    png_path = output_dir / "simulation.png"
    png_path.write_bytes(render_binary_trajectory_png(valid_states))
    final = valid_states[-1].astype(np.int8)
    consensus = bool(np.all(final == final[0]))
    tokens = token_pair_spec(spec.token_pair)["tokens"]
    consensus_state = int(final[0]) if consensus else None
    expected_state = int(distribution.initial_majority_opinion)
    summary = {
        **asdict(spec),
        "replay": replay,
        "model_family": MODEL_FAMILY,
        "served_model": model,
        "agents": agents,
        "n_neighbors_including_center": N_NEIGHBORS,
        "majority_ratio": majority_ratio,
        "initial_count_0": distribution.count_0,
        "initial_count_1": distribution.count_1,
        "expected_token": tokens[expected_state],
        "expected_state": expected_state,
        "max_transitions": max_transitions,
        "completed_transitions": completed_transitions,
        "completed_rounds_including_initial": completed_transitions + 1,
        "stop_reason": stop_reason,
        "consensus": consensus,
        "consensus_token": tokens[consensus_state] if consensus else None,
        "consensus_state": consensus_state,
        "correct_consensus": consensus_state == expected_state,
        "temperature": 0.0,
        "request_seed": 42,
        "max_output_tokens": MAX_OUTPUT_TOKENS[spec.variant],
        "request_failures": request_failures,
        "parse_failures": parse_failures,
        "states_sha256": sha256_file(states_path),
        "png_sha256": sha256_file(png_path),
        "normalized_log_sha256": sha256_file(normalized_path),
        "requests_sha256": sha256_file(requests_path),
    }
    write_json(output_dir / "result_summary.json", summary)
    return summary


def grouped_specs(specs: Iterable[CellSpec]) -> dict[tuple[int, str, str, str], list[CellSpec]]:
    groups: dict[tuple[int, str, str, str], list[CellSpec]] = {}
    for spec in specs:
        groups.setdefault(group_key(spec), []).append(spec)
    return groups


def write_replay_group_metadata(
    *,
    specs: list[CellSpec],
    work_root: Path,
    replay: int,
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    base_url: str,
    model: str,
    agents: int,
    majority_ratio: float,
    max_transitions: int,
    request_workers: int,
    cell_workers: int,
) -> None:
    root = replay_root(work_root, specs[0], replay)
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "scoreboard.csv", sorted(rows, key=lambda row: row["seed"]))
    manifest = {
        "run_id": specs[0].run_id,
        "replay": replay,
        "created_at_utc": utc_now(),
        "git_commit": git_commit(),
        "mechanism": "online_llm_memory",
        "model_family": MODEL_FAMILY,
        "served_model": model,
        "base_url": base_url,
        "variant": specs[0].variant,
        "historical_base_variant": VARIANT_BASES[specs[0].variant],
        "token_pair": specs[0].token_pair,
        "tokens": list(token_pair_spec(specs[0].token_pair)["tokens"]),
        "n_neighbors_including_center": N_NEIGHBORS,
        "memory_window": specs[0].memory_window,
        "agents": agents,
        "seeds": sorted(spec.seed for spec in specs),
        "majority_ratio": majority_ratio,
        "max_transitions": max_transitions,
        "stop_conditions": ["consensus", "max_rounds"],
        "sampling": {
            "temperature": 0.0,
            "seed": 42,
            "max_output_tokens": MAX_OUTPUT_TOKENS[specs[0].variant],
            "sampling_overrides": "omitted",
        },
        "cell_workers": cell_workers,
        "request_workers": request_workers,
        "source_hashes": source_hashes(source_paths(REPO_ROOT) + [Path(__file__).resolve()]),
    }
    write_json(root / "run_manifest.json", manifest)
    write_json(root / "source_hashes.json", manifest["source_hashes"])
    proofread = {
        "completed_cells": len(rows),
        "expected_cells": len(specs),
        "error_cells": len(errors),
        "request_failures": sum(row["request_failures"] for row in rows),
        "parse_failures": sum(row["parse_failures"] for row in rows),
        "errors": errors,
    }
    write_json(root / "proofread_summary.json", proofread)
    (root / "proofread_summary.md").write_text(
        "# Phase 3 replay proofread\n\n"
        f"- Cells: `{len(rows)}/{len(specs)}`\n"
        f"- Errors: `{len(errors)}`\n"
        f"- Request failures: `{proofread['request_failures']}`\n"
        f"- Parse failures: `{proofread['parse_failures']}`\n",
        encoding="utf-8",
    )


def run_manifest_replay(
    *,
    specs: list[CellSpec],
    replay: int,
    work_root: Path,
    progress_path: Path,
    base_url: str,
    model: str,
    stage_status_path: Path | None = None,
    agents: int = 30,
    majority_ratio: float = 0.51,
    max_transitions: int = 60,
    cell_workers: int = 2,
    request_workers: int = 32,
    timeout_s: int = 300,
    max_attempts: int = 5,
) -> dict[str, Any]:
    if replay not in (1, 2) or cell_workers < 1 or request_workers < agents:
        raise ValueError("replay must be 1/2 and request_workers must cover one full round")
    pending: list[CellSpec] = []
    completed: list[dict[str, Any]] = []
    for spec in specs:
        path = cell_root(work_root, spec, replay)
        if completed_cell_is_valid(path, spec, replay):
            completed.append(json.loads((path / "result_summary.json").read_text(encoding="utf-8")))
            continue
        isolate_partial(path, work_root, spec, replay)
        pending.append(spec)
        atomic_append_jsonl(
            progress_path,
            {
                "utc": utc_now(),
                "event": "cell_pending",
                "status": "pending",
                "replay": replay,
                **asdict(spec),
            },
        )

    errors: list[dict[str, Any]] = []
    fatal_endpoint = threading.Event()
    with ThreadPoolExecutor(max_workers=request_workers) as request_pool:
        with ThreadPoolExecutor(max_workers=cell_workers) as cell_pool:
            futures = {
                cell_pool.submit(
                    run_cell,
                    spec=spec,
                    replay=replay,
                    output_dir=cell_root(work_root, spec, replay),
                    progress_path=progress_path,
                    request_pool=request_pool,
                    fatal_endpoint=fatal_endpoint,
                    base_url=base_url,
                    model=model,
                    agents=agents,
                    majority_ratio=majority_ratio,
                    max_transitions=max_transitions,
                    timeout_s=timeout_s,
                    max_attempts=max_attempts,
                ): spec
                for spec in pending
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    row = future.result()
                    completed.append(row)
                    event = {
                        "utc": utc_now(),
                        "event": "cell_complete",
                        "status": "complete",
                        "replay": replay,
                        **asdict(spec),
                    }
                except Exception as exc:
                    error = {**asdict(spec), "error": f"{type(exc).__name__}: {exc}"}
                    errors.append(error)
                    path = cell_root(work_root, spec, replay)
                    path.mkdir(parents=True, exist_ok=True)
                    write_json(path / "error.json", error)
                    event = {
                        "utc": utc_now(),
                        "event": "cell_error",
                        "status": "error",
                        "replay": replay,
                        **error,
                    }
                atomic_append_jsonl(progress_path, event)
                if stage_status_path is not None:
                    atomic_append_jsonl(stage_status_path, event)
                print(json.dumps(event, ensure_ascii=False), flush=True)

    by_group = grouped_specs(specs)
    for key, group in by_group.items():
        group_rows = [row for row in completed if group_key(CellSpec.from_dict(row)) == key]
        group_errors = [row for row in errors if group_key(CellSpec.from_dict(row)) == key]
        write_replay_group_metadata(
            specs=group,
            work_root=work_root,
            replay=replay,
            rows=group_rows,
            errors=group_errors,
            base_url=base_url,
            model=model,
            agents=agents,
            majority_ratio=majority_ratio,
            max_transitions=max_transitions,
            request_workers=request_workers,
            cell_workers=cell_workers,
        )
    if fatal_endpoint.is_set():
        raise RuntimeError("FAIL_ENDPOINT: replay stopped after 30-minute backoff")
    return {
        "replay": replay,
        "expected_cells": len(specs),
        "completed_cells": len(completed),
        "error_cells": len(errors),
    }


def load_summary(root: Path, seed: int) -> dict[str, Any] | None:
    path = root / "cells" / f"seed_{seed:02d}" / "result_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def compare_group_replays(
    replay1: Path, replay2: Path, proof_dir: Path, seeds: list[int]
) -> dict[str, Any]:
    proof_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        summary1, summary2 = load_summary(replay1, seed), load_summary(replay2, seed)
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
        states1, states2 = np.load(cell1 / "states.npy"), np.load(cell2 / "states.npy")
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
    mismatches = [row for row in rows if not row["exact_equal"]]
    summary = {
        "deterministic": len(rows) == len(seeds) and not mismatches,
        "expected_cells": len(seeds),
        "exact_cells": sum(row["exact_equal"] for row in rows),
        "letter_exact_cells": sum(row["responses_letter_exact"] for row in rows),
        "first_divergence": mismatches[0] if mismatches else None,
    }
    write_json(proof_dir / "comparison_summary.json", summary)
    if mismatches:
        write_json(proof_dir / "first_divergence.json", mismatches[0])
    return summary


def finalize_manifest_groups(specs: list[CellSpec], work_root: Path) -> list[dict[str, Any]]:
    results = []
    for group in grouped_specs(specs).values():
        representative = group[0]
        destination = final_root(representative)
        if destination.exists():
            status = json.loads((destination / "status.json").read_text(encoding="utf-8"))
            results.append(status)
            continue
        root = group_root(work_root, representative)
        proof = compare_group_replays(
            root / "replay_01",
            root / "replay_02",
            root / "proof",
            sorted(spec.seed for spec in group),
        )
        finalize_replays(
            replay1=root / "replay_01",
            replay2=root / "replay_02",
            proof_dir=root / "proof",
            final_root=destination,
            deterministic=proof["deterministic"],
            status_details={
                "run_id": representative.run_id,
                "variant": representative.variant,
                "token_pair": representative.token_pair,
                "n_neighbors": N_NEIGHBORS,
                "memory_window": representative.memory_window,
                "completed_at_utc": utc_now(),
                **proof,
            },
        )
        results.append(json.loads((destination / "status.json").read_text(encoding="utf-8")))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replay", type=int, choices=(1, 2), required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8127/v1")
    parser.add_argument("--model", default="gemma3-4b-temp0")
    parser.add_argument("--cell-workers", type=int, default=2)
    parser.add_argument("--request-workers", type=int, default=32)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_manifest_replay(
        specs=read_manifest(args.manifest),
        replay=args.replay,
        work_root=args.work_root,
        progress_path=args.progress,
        stage_status_path=None,
        base_url=args.base_url,
        model=args.model,
        cell_workers=args.cell_workers,
        request_workers=args.request_workers,
        timeout_s=args.timeout_s,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["error_cells"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
