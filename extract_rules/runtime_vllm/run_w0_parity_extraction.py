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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.parity_artifacts import (  # noqa: E402
    finalize_replays,
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
    render_w0_prompt,
    resolve_variants,
    source_paths,
    token_pair_spec,
    visible_tokens,
)


MODEL_FAMILY = "gemma-3-4b-it"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def query_configuration(
    *,
    binary_config: str,
    variant: str,
    base_url: str,
    model: str,
    timeout_s: int,
    max_attempts: int,
) -> dict[str, Any]:
    system_prompt, user_prompt = render_w0_prompt(variant, binary_config)
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
        "configuration": binary_config,
        "configuration_letters": "".join(visible_tokens(binary_config, variant)),
        "num_ones": binary_config.count("1"),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt_sha256": prompt_hash(system_prompt, user_prompt),
        "payload": payload,
        "payload_sha256": payload_hash(payload),
        **query,
    }


def run_replay(
    *,
    output_dir: Path,
    variant: str,
    token_pair: str,
    n_neighbors: int,
    base_url: str,
    model: str,
    workers: int,
    timeout_s: int,
    max_attempts: int,
    run_id: str,
    replay: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    configs = [format(index, f"0{n_neighbors}b") for index in range(2**n_neighbors)]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(configs))) as pool:
        futures = {
            pool.submit(
                query_configuration,
                binary_config=config,
                variant=variant,
                base_url=base_url,
                model=model,
                timeout_s=timeout_s,
                max_attempts=max_attempts,
            ): config
            for config in configs
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                json.dumps(
                    {
                        "phase": 1,
                        "replay": replay,
                        "variant": variant,
                        "n": n_neighbors,
                        "configuration": record["configuration"],
                        "choice": record["choice"],
                    }
                ),
                flush=True,
            )
    records.sort(key=lambda item: item["configuration"])
    write_jsonl(output_dir / "requests.jsonl", records)

    rule_rows = [
        {
            "variante": variant,
            "temperatura": 0.0,
            "configuracao_binaria": record["configuration"],
            "configuracao_letras": record["configuration_letters"],
            "num_ones": record["num_ones"],
            "escolha": record["choice"],
            "logprobs_json": "",
        }
        for record in records
        if record["choice"] is not None
    ]
    write_csv(
        output_dir / "rule.csv",
        rule_rows,
        [
            "variante",
            "temperatura",
            "configuracao_binaria",
            "configuracao_letras",
            "num_ones",
            "escolha",
            "logprobs_json",
        ],
    )
    extended_rows = [
        {
            "variante": variant,
            "temperatura": 0.0,
            "configuracao_binaria": record["configuration"],
            "configuracao_letras": record["configuration_letters"],
            "num_ones": record["num_ones"],
            "escolha": record["choice"],
            "prompt_input": f"SYSTEM:\n{record['system_prompt']}\n\nUSER:\n{record['user_prompt']}",
            "llm_response": record["raw_response"],
        }
        for record in records
    ]
    write_csv(output_dir / "extended.csv", extended_rows)

    complete = len(rule_rows) == len(configs)
    rules = {row["configuracao_binaria"]: row["escolha"] for row in rule_rows}
    write_json(
        output_dir / "rule.json",
        {
            "source_csv": "rule.csv",
            "n_agents": n_neighbors,
            "n_neighbors": n_neighbors,
            "total_rules": len(rules),
            "rules": rules,
        },
    )
    manifest = {
        "run_id": run_id,
        "replay": replay,
        "created_at_utc": utc_now(),
        "git_commit": git_commit(),
        "model_family": MODEL_FAMILY,
        "served_model": model,
        "base_url": base_url,
        "variant": variant,
        "historical_base_variant": VARIANT_BASES[variant],
        "token_pair": token_pair,
        "tokens": list(token_pair_spec(token_pair)["tokens"]),
        "binary_mapping": {
            "0": token_pair_spec(token_pair)["tokens"][0],
            "1": token_pair_spec(token_pair)["tokens"][1],
        },
        "n_neighbors_including_center": n_neighbors,
        "expected_configurations": len(configs),
        "sampling": {
            "temperature": 0.0,
            "seed": 42,
            "max_output_tokens": MAX_OUTPUT_TOKENS[variant],
            "sampling_overrides": "omitted",
        },
        "workers": workers,
        "source_hashes": source_hashes(source_paths(REPO_ROOT) + [Path(__file__).resolve()]),
    }
    write_json(output_dir / "rule_manifest.json", manifest)
    summary = {
        "complete": complete,
        "expected_configurations": len(configs),
        "valid_configurations": len(rule_rows),
        "request_failures": sum(record["request_failures"] for record in records),
        "parse_failures": sum(record["parse_failures"] for record in records),
        "terminal_failures": sum(record["choice"] is None for record in records),
        "rule_csv_sha256": sha256_file(output_dir / "rule.csv"),
        "requests_sha256": sha256_file(output_dir / "requests.jsonl"),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "source_hashes.json", manifest["source_hashes"])
    return summary


def load_requests(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def compare_replays(replay1: Path, replay2: Path, proof_dir: Path) -> dict[str, Any]:
    proof_dir.mkdir(parents=True, exist_ok=False)
    rows1 = load_requests(replay1 / "requests.jsonl")
    rows2 = load_requests(replay2 / "requests.jsonl")
    paired: list[dict[str, Any]] = []
    for left, right in zip(rows1, rows2):
        equal = all(
            left[key] == right[key]
            for key in (
                "configuration",
                "prompt_sha256",
                "payload_sha256",
                "raw_response",
                "choice",
            )
        )
        paired.append(
            {
                "configuration": left["configuration"],
                "prompt_sha256_replay1": left["prompt_sha256"],
                "prompt_sha256_replay2": right["prompt_sha256"],
                "response_sha256_replay1": left["raw_response_sha256"],
                "response_sha256_replay2": right["raw_response_sha256"],
                "choice_replay1": left["choice"],
                "choice_replay2": right["choice"],
                "exact_equal": equal,
            }
        )
    summary1 = json.loads((replay1 / "summary.json").read_text(encoding="utf-8"))
    summary2 = json.loads((replay2 / "summary.json").read_text(encoding="utf-8"))
    expected = summary1["expected_configurations"]
    gate = (
        summary1["complete"]
        and summary2["complete"]
        and len(rows1) == len(rows2) == expected
        and len(paired) == expected
        and all(row["exact_equal"] for row in paired)
    )
    write_csv(proof_dir / "paired_hashes.csv", paired)
    mismatches = [row for row in paired if not row["exact_equal"]]
    summary = {
        "deterministic": gate,
        "expected_configurations": expected,
        "compared_configurations": len(paired),
        "exact_configurations": sum(row["exact_equal"] for row in paired),
        "replay1_rule_sha256": summary1["rule_csv_sha256"],
        "replay2_rule_sha256": summary2["rule_csv_sha256"],
        "first_divergence": mismatches[0] if mismatches else None,
    }
    write_json(proof_dir / "comparison_summary.json", summary)
    if mismatches:
        write_json(proof_dir / "first_divergence.json", mismatches[0])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8127/v1")
    parser.add_argument("--model", default="gemma3-4b-temp0")
    parser.add_argument("--token-pair", choices=list(TOKEN_PAIR_SPECS), default="01")
    parser.add_argument("--neighbors", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(n < 3 or n % 2 == 0 for n in args.neighbors):
        raise ValueError("all neighborhood sizes must be odd and >= 3")
    variants = resolve_variants(args.token_pair, args.variants)
    spec = token_pair_spec(args.token_pair)
    results: list[dict[str, Any]] = []
    for n_neighbors in args.neighbors:
        for variant in variants:
            work_root = (
                REPO_ROOT
                / "artifacts/work/rtx5090"
                / f"n={n_neighbors}"
                / args.run_id
                / "phase1"
                / args.token_pair
                / variant
            )
            final_root = (
                REPO_ROOT
                / "artifacts/phase1_rule_extraction/rtx5090"
                / f"n={n_neighbors}"
                / MODEL_FAMILY
                / spec["folder"]
                / variant
                / args.run_id
            )
            if work_root.exists() or final_root.exists():
                raise FileExistsError(work_root if work_root.exists() else final_root)
            replay1 = work_root / "replay_01"
            replay2 = work_root / "replay_02"
            run_replay(
                output_dir=replay1,
                variant=variant,
                token_pair=args.token_pair,
                n_neighbors=n_neighbors,
                base_url=args.base_url,
                model=args.model,
                workers=args.workers,
                timeout_s=args.timeout_s,
                max_attempts=args.max_attempts,
                run_id=args.run_id,
                replay=1,
            )
            run_replay(
                output_dir=replay2,
                variant=variant,
                token_pair=args.token_pair,
                n_neighbors=n_neighbors,
                base_url=args.base_url,
                model=args.model,
                workers=args.workers,
                timeout_s=args.timeout_s,
                max_attempts=args.max_attempts,
                run_id=args.run_id,
                replay=2,
            )
            proof = compare_replays(replay1, replay2, work_root / "proof")
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
                    "n_neighbors": n_neighbors,
                    "completed_at_utc": utc_now(),
                },
            )
            results.append({"variant": variant, "n": n_neighbors, **proof})
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if all(result["deterministic"] for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
