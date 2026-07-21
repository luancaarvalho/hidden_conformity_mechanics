#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experimentos_automatos.runtime.metrics_utils import (  # noqa: E402
    calcular_ici_token_bias_from_csv,
)
from utils.parity_artifacts import (  # noqa: E402
    first_array_difference,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)
from utils.w0_parity_contract import parse_final_choice, token_pair_spec  # noqa: E402


MODEL_FAMILY = "gemma-3-4b-it"
PAIRS = ("01", "kz", "triangle_circle")
STYLES = {0: "v9_only_token", 1: "v21_cot"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def valid_states(path: Path) -> np.ndarray:
    states = np.load(path)
    if states.ndim != 2 or states.shape[1] != 30:
        raise ValueError(f"unexpected states shape: {states.shape} at {path}")
    valid_mask = np.isfinite(states).all(axis=1)
    count = int(valid_mask.sum())
    if count == 0 or not valid_mask[:count].all() or valid_mask[count:].any():
        raise ValueError(f"states are not a contiguous valid prefix: {path}")
    return states[:count].astype(np.int8)


def classify(states: np.ndarray) -> dict[str, Any]:
    count0, count1 = int(np.sum(states[0] == 0)), int(np.sum(states[0] == 1))
    if count0 == count1:
        raise ValueError("initial state has no strict majority")
    expected = int(count1 > count0)
    final = states[-1]
    consensus = bool(np.all(final == final[0]))
    consensus_state = int(final[0]) if consensus else None
    classification = (
        "no_consensus"
        if not consensus
        else "correct" if consensus_state == expected else "wrong"
    )
    return {
        "initial_count_0": count0,
        "initial_count_1": count1,
        "expected_state": expected,
        "consensus_state": consensus_state,
        "classification": classification,
        "completed_transitions": len(states) - 1,
        "stop_reason": "consensus" if consensus else "max_rounds",
        "final_token1_density": float(states[-1].mean()),
    }


class CampaignIndex:
    def __init__(self, campaign_id: str) -> None:
        self.campaign_id = campaign_id
        self.root = REPO_ROOT / "artifacts/orchestration/rtx5090" / campaign_id
        self.manifest = read_json(self.root / "campaign_manifest.json")
        self.timestamp = campaign_id.rsplit("_", 1)[-1]
        self.new_runs = {int(key): value for key, value in self.manifest["run_ids"].items()}
        self.old_runs = {
            pair: {int(key): value for key, value in runs.items()}
            for pair, runs in self.manifest["old_memory_run_ids"].items()
        }
        self.rule_runs = self.manifest["rule_run_ids"]

    def variant(self, pair: str, style: str) -> str:
        variants = token_pair_spec(pair)["variants"]
        return variants[0] if style == "v9_only_token" else variants[1]

    def run_id(self, pair: str, window: int, seed: int) -> str:
        if window in (0, 1) and seed <= 20:
            return self.old_runs[pair][window]
        return self.new_runs[window]

    def phase3_root(self, pair: str, window: int, variant: str, seed: int) -> Path:
        return (
            REPO_ROOT
            / "artifacts/phase3_memory/rtx5090/n=7"
            / MODEL_FAMILY
            / token_pair_spec(pair)["folder"]
            / f"W={window}"
            / variant
            / self.run_id(pair, window, seed)
        )

    def phase2_root(self, pair: str, variant: str, seed: int) -> Path:
        run_id = self.old_runs[pair][0] if seed <= 20 else self.new_runs[0]
        return (
            REPO_ROOT
            / "artifacts/phase2_cellular_automata/rtx5090/n=7"
            / MODEL_FAMILY
            / token_pair_spec(pair)["folder"]
            / variant
            / run_id
        )

    def rule_csv(self, pair: str, variant: str) -> Path:
        return (
            REPO_ROOT
            / "artifacts/phase1_rule_extraction/rtx5090/n=7"
            / MODEL_FAMILY
            / token_pair_spec(pair)["folder"]
            / variant
            / self.rule_runs[pair]
            / "canonical/rule.csv"
        )

    def output_root(self) -> Path:
        return (
            REPO_ROOT
            / "artifacts/cross_window_validation/rtx5090/n=7"
            / MODEL_FAMILY
            / f"W0-5_seed1-50_{self.timestamp}"
        )


def proofread_online_cell(
    index: CampaignIndex, pair: str, style: str, window: int, seed: int
) -> dict[str, Any]:
    variant = index.variant(pair, style)
    root = index.phase3_root(pair, window, variant, seed)
    cell = root / "canonical/cells" / f"seed_{seed:02d}"
    base = {
        "token_pair": pair,
        "token0": token_pair_spec(pair)["tokens"][0],
        "token1": token_pair_spec(pair)["tokens"][1],
        "style": style,
        "variant": variant,
        "memory_window": window,
        "seed": seed,
        "run_id": index.run_id(pair, window, seed),
        "cell_path": str(cell),
    }
    required = {
        "states": cell / "states.npy",
        "requests": cell / "requests.jsonl",
        "summary": cell / "result_summary.json",
        "png": cell / "simulation.png",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {
            **base,
            "mechanical_pass": False,
            "error": f"missing:{','.join(missing)}",
        }
    try:
        states = valid_states(required["states"])
        derived = classify(states)
        summary = read_json(required["summary"])
        requests = read_jsonl(required["requests"])
        tokens = token_pair_spec(pair)["tokens"]
        expected_requests = derived["completed_transitions"] * 30
        request_failures = sum(int(row["request_failures"]) for row in requests)
        parse_failures = sum(int(row["parse_failures"]) for row in requests)
        strict_failures = 0
        choices = Counter()
        for row in requests:
            choice, choice_token = row.get("choice"), row.get("choice_token")
            if choice in (0, 1):
                choices[int(choice)] += 1
            if (
                choice not in (0, 1)
                or choice_token not in tokens
                or tokens[int(choice)] != choice_token
                or parse_final_choice(row.get("raw_response", ""), variant) != choice_token
            ):
                strict_failures += 1
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
                summary["request_failures"] == request_failures,
                summary["parse_failures"] == parse_failures,
                summary["states_sha256"] == sha256_file(required["states"]),
                summary["requests_sha256"] == sha256_file(required["requests"]),
                summary["png_sha256"] == sha256_file(required["png"]),
            )
        )
        mechanical = all(
            (
                len(requests) == expected_requests,
                request_failures == 0,
                parse_failures == 0,
                strict_failures == 0,
                summary_matches,
            )
        )
        valid_choices = choices[0] + choices[1]
        return {
            **base,
            **derived,
            "expected_token": expected_token,
            "consensus_token": consensus_token,
            "request_records": len(requests),
            "expected_request_records": expected_requests,
            "request_failures": request_failures,
            "parse_failures": parse_failures,
            "strict_contract_failures": strict_failures,
            "choice_0_count": choices[0],
            "choice_1_count": choices[1],
            "empirical_choice_bias": (
                (choices[1] - choices[0]) / valid_choices if valid_choices else None
            ),
            "summary_matches_raw": summary_matches,
            "mechanical_pass": mechanical,
            "states_sha256": sha256_file(required["states"]),
            "requests_sha256": sha256_file(required["requests"]),
            "error": "",
        }
    except Exception as exc:
        return {
            **base,
            "mechanical_pass": False,
            "error": f"{type(exc).__name__}:{exc}",
        }


def rule_metrics(index: CampaignIndex) -> list[dict[str, Any]]:
    rows = []
    for pair in PAIRS:
        for style in STYLES.values():
            variant = index.variant(pair, style)
            path = index.rule_csv(pair, variant)
            ici, bias = calcular_ici_token_bias_from_csv(path)
            with path.open(encoding="utf-8", newline="") as handle:
                rules = list(csv.DictReader(handle))
            count1 = sum(int(row["escolha"]) for row in rules)
            rows.append(
                {
                    "token_pair": pair,
                    "style": style,
                    "variant": variant,
                    "ici": ici,
                    "token_bias": bias,
                    "output_0_count": len(rules) - count1,
                    "output_1_count": count1,
                    "rows": len(rules),
                    "rule_sha256": sha256_file(path),
                    "rule_path": str(path),
                }
            )
    return rows


def group_summary(cells: list[dict[str, Any]], rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        grouped[(row["token_pair"], row["style"], row["memory_window"])].append(row)
    rule_index = {(row["token_pair"], row["style"]): row for row in rules}
    output = []
    for (pair, style, window), rows in sorted(grouped.items()):
        valid = [row for row in rows if row.get("mechanical_pass")]
        classifications = Counter(row.get("classification") for row in valid)
        choice0 = sum(int(row.get("choice_0_count", 0)) for row in valid)
        choice1 = sum(int(row.get("choice_1_count", 0)) for row in valid)
        expected_sides = {row.get("expected_state") for row in valid}
        majority_0 = sum(row.get("expected_state") == 0 for row in valid)
        majority_1 = sum(row.get("expected_state") == 1 for row in valid)
        baseline = rule_index[(pair, style)]
        output.append(
            {
                "token_pair": pair,
                "style": style,
                "memory_window": window,
                "cells": len(rows),
                "mechanical_cells": len(valid),
                "correct": classifications["correct"],
                "wrong": classifications["wrong"],
                "no_consensus": classifications["no_consensus"],
                "both_majority_sides_present": expected_sides == {0, 1},
                "majority_0_seeds": majority_0,
                "majority_1_seeds": majority_1,
                "mean_completed_transitions": (
                    sum(row["completed_transitions"] for row in valid) / len(valid)
                    if valid
                    else None
                ),
                "mean_final_token1_density": (
                    sum(row["final_token1_density"] for row in valid) / len(valid)
                    if valid
                    else None
                ),
                "empirical_choice_bias": (
                    (choice1 - choice0) / (choice1 + choice0)
                    if choice1 + choice0
                    else None
                ),
                "static_rule_ici": baseline["ici"],
                "static_rule_token_bias": baseline["token_bias"],
                "static_metrics_role": (
                    "frozen_rule_metrics"
                    if window == 0
                    else "W0_baseline_covariates_only"
                ),
                "mechanical_pass": (
                    len(rows) == 50
                    and len(valid) == 50
                    and majority_0 == 25
                    and majority_1 == 25
                ),
                "dct_bidirectional_pass": (
                    len(rows) == 50
                    and len(valid) == 50
                    and majority_0 == 25
                    and majority_1 == 25
                    and classifications["correct"] == 50
                ),
            }
        )
    return output


def compare_windows(cells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["token_pair"], row["style"], row["memory_window"], row["seed"]): row
        for row in cells
        if row.get("mechanical_pass")
    }
    rows, divergences = [], []
    for pair in PAIRS:
        for style in STYLES.values():
            for window in range(1, 6):
                for seed in range(1, 51):
                    baseline = lookup.get((pair, style, 0, seed))
                    memory = lookup.get((pair, style, window, seed))
                    row = {
                        "token_pair": pair,
                        "style": style,
                        "memory_window": window,
                        "seed": seed,
                        "comparable": baseline is not None and memory is not None,
                    }
                    if baseline is not None and memory is not None:
                        states0 = valid_states(Path(baseline["cell_path"]) / "states.npy")
                        statesw = valid_states(Path(memory["cell_path"]) / "states.npy")
                        first_round, first_agent = first_array_difference(states0, statesw)
                        row.update(
                            {
                                "trajectory_equal": bool(
                                    np.array_equal(states0, statesw, equal_nan=True)
                                ),
                                "first_diff_round": first_round,
                                "first_diff_agent": first_agent,
                                "w0_classification": baseline["classification"],
                                "memory_classification": memory["classification"],
                                "classification_changed": (
                                    baseline["classification"] != memory["classification"]
                                ),
                                "improved": baseline["classification"] != "correct"
                                and memory["classification"] == "correct",
                                "regressed": baseline["classification"] == "correct"
                                and memory["classification"] != "correct",
                                "transition_delta": memory["completed_transitions"]
                                - baseline["completed_transitions"],
                            }
                        )
                        if not row["trajectory_equal"]:
                            divergences.append(row.copy())
                    rows.append(row)
    return rows, divergences


def compare_w0_automata(
    index: CampaignIndex, seeds: range
) -> list[dict[str, Any]]:
    rows = []
    for pair in PAIRS:
        for style in STYLES.values():
            variant = index.variant(pair, style)
            for seed in seeds:
                online = index.phase3_root(pair, 0, variant, seed) / "canonical/cells" / f"seed_{seed:02d}"
                automaton = index.phase2_root(pair, variant, seed) / "canonical/cells" / f"seed_{seed:02d}"
                base = {
                    "token_pair": pair,
                    "style": style,
                    "variant": variant,
                    "seed": seed,
                }
                try:
                    online_states = np.load(online / "states.npy")
                    automaton_states = np.load(automaton / "states.npy")
                    first_round, first_agent = first_array_difference(
                        automaton_states, online_states
                    )
                    rows.append(
                        {
                            **base,
                            "states_equal": bool(
                                np.array_equal(
                                    automaton_states, online_states, equal_nan=True
                                )
                            ),
                            "png_equal": sha256_file(automaton / "simulation.png")
                            == sha256_file(online / "simulation.png"),
                            "first_diff_round": first_round,
                            "first_diff_agent": first_agent,
                            "error": "",
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            **base,
                            "states_equal": False,
                            "png_equal": False,
                            "first_diff_round": None,
                            "first_diff_agent": None,
                            "error": f"{type(exc).__name__}:{exc}",
                        }
                    )
    return rows


def source_runs(index: CampaignIndex) -> list[dict[str, Any]]:
    rows = []
    for pair in PAIRS:
        for window in range(6):
            if window in (0, 1):
                rows.append(
                    {
                        "token_pair": pair,
                        "memory_window": window,
                        "seeds": "1-20",
                        "run_id": index.old_runs[pair][window],
                        "provenance": "existing_immutable",
                    }
                )
            rows.append(
                {
                    "token_pair": pair,
                    "memory_window": window,
                    "seeds": "21-50" if window in (0, 1) else "1-50",
                    "run_id": index.new_runs[window],
                    "provenance": "campaign",
                }
            )
    return rows


def proofread_new_determinism(index: CampaignIndex) -> list[dict[str, Any]]:
    rows = []
    for window in range(6):
        expected = 30 if window in (0, 1) else 50
        seed = 21 if window in (0, 1) else 1
        for pair in PAIRS:
            for style in STYLES.values():
                variant = index.variant(pair, style)
                root = index.phase3_root(pair, window, variant, seed)
                status_path = root / "status.json"
                proof_path = root / "determinism/comparison_summary.json"
                try:
                    status = read_json(status_path)
                    proof = read_json(proof_path)
                    passed = (
                        status["status"] == "PASS"
                        and proof["deterministic"] is True
                        and proof["expected_cells"] == expected
                        and proof["exact_cells"] == expected
                        and proof["letter_exact_cells"] == expected
                    )
                    error = ""
                except Exception as exc:
                    proof = {}
                    passed = False
                    error = f"{type(exc).__name__}:{exc}"
                rows.append(
                    {
                        "token_pair": pair,
                        "style": style,
                        "variant": variant,
                        "memory_window": window,
                        "expected_cells": expected,
                        "exact_cells": proof.get("exact_cells", 0),
                        "letter_exact_cells": proof.get("letter_exact_cells", 0),
                        "deterministic_pass": passed,
                        "error": error,
                    }
                )
    return rows


def run_w0_extension(index: CampaignIndex) -> int:
    output = index.root / "w0_extension_proofread"
    output.mkdir(parents=True, exist_ok=True)
    parity = compare_w0_automata(index, range(21, 51))
    write_csv(output / "parity_scoreboard.csv", parity)
    exact = sum(row["states_equal"] and row["png_equal"] for row in parity)
    summary = {
        "status": "PASS" if len(parity) == 180 and exact == 180 else "FAIL_PARITY",
        "expected_cases": 180,
        "compared_cases": len(parity),
        "exact_cases": exact,
    }
    write_json(output / "summary.json", summary)
    return 0 if summary["status"] == "PASS" else 3


def run_final(index: CampaignIndex) -> int:
    output = index.output_root()
    output.mkdir(parents=True, exist_ok=True)
    rules = rule_metrics(index)
    cells = [
        proofread_online_cell(index, pair, style, window, seed)
        for pair in PAIRS
        for style in STYLES.values()
        for window in range(6)
        for seed in range(1, 51)
    ]
    groups = group_summary(cells, rules)
    impacts, divergences = compare_windows(cells)
    parity = compare_w0_automata(index, range(1, 51))
    determinism = proofread_new_determinism(index)
    write_csv(output / "cell_scoreboard.csv", cells)
    write_csv(output / "group_summary.csv", groups)
    write_csv(output / "window_impact_vs_w0.csv", impacts)
    write_csv(output / "first_divergences_vs_w0.csv", divergences, list(impacts[0]))
    write_csv(output / "w0_automaton_parity.csv", parity)
    write_csv(output / "new_replay_determinism.csv", determinism)
    write_csv(output / "n7_rule_metrics.csv", rules)
    write_csv(output / "ici_token_bias_window_linkage.csv", groups)
    write_jsonl(output / "source_runs.jsonl", source_runs(index))
    mechanical_cells = sum(row.get("mechanical_pass") is True for row in cells)
    mechanical_groups = sum(row["mechanical_pass"] for row in groups)
    parity_exact = sum(row["states_equal"] and row["png_equal"] for row in parity)
    dct_passes = sum(row["dct_bidirectional_pass"] for row in groups)
    deterministic_groups = sum(row["deterministic_pass"] for row in determinism)
    deterministic_cells = sum(
        row["exact_cells"] for row in determinism if row["deterministic_pass"]
    )
    gate = all(
        (
            len(cells) == 1800,
            mechanical_cells == 1800,
            len(groups) == 36,
            mechanical_groups == 36,
            len(parity) == 300,
            parity_exact == 300,
            len(determinism) == 36,
            deterministic_groups == 36,
            deterministic_cells == 1560,
        )
    )
    summary = {
        "status": "PASS" if gate else "FAIL_PROOFREAD",
        "campaign_id": index.campaign_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "expected_cells": 1800,
        "proofread_cells": len(cells),
        "mechanical_cells": mechanical_cells,
        "expected_groups": 36,
        "proofread_groups": len(groups),
        "mechanical_groups": mechanical_groups,
        "w0_parity_exact": parity_exact,
        "w0_parity_expected": 300,
        "deterministic_new_cells": deterministic_cells,
        "deterministic_new_cells_expected": 1560,
        "deterministic_new_groups": deterministic_groups,
        "deterministic_new_groups_expected": 36,
        "dct_bidirectional_pass_groups": dct_passes,
        "static_metric_warning": (
            "ICI and token_bias describe the complete W=0 frozen rule only; for W>0 "
            "they are baseline covariates, not metrics of the history-conditioned policy."
        ),
    }
    write_json(output / "proofread_summary.json", summary)
    (output / "proofread_summary.md").write_text(
        "# Gemma 3 4B n=7 W0-W5 proofread\n\n"
        f"- Gate: `{summary['status']}`\n"
        f"- Cells: `{mechanical_cells}/1800`\n"
        f"- Groups: `{mechanical_groups}/36`\n"
        f"- W0 automaton parity: `{parity_exact}/300`\n"
        f"- New deterministic cells: `{deterministic_cells}/1560`\n"
        f"- Bidirectional DCT passes: `{dct_passes}/36`\n\n"
        "ICI and token bias are static W=0 rule metrics. W1-W5 report them only as baseline covariates.\n",
        encoding="utf-8",
    )
    return 0 if gate else 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--stage", choices=("w0-extension", "final"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = CampaignIndex(args.campaign_id)
    return run_w0_extension(index) if args.stage == "w0-extension" else run_final(index)


if __name__ == "__main__":
    raise SystemExit(main())
