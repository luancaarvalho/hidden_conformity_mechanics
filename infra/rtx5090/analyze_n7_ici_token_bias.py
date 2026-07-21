#!/usr/bin/env python3
"""Relate n=7 rule-table ICI/token bias to W=0 and W=1 simulations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np


PAIR_META = {
    "01": {"path": "tokens=0-1", "token0": "0", "token1": "1"},
    "kz": {"path": "tokens=k-z", "token0": "k", "token1": "z"},
    "triangle_circle": {
        "path": "tokens=triangle-circle",
        "token0": "△",
        "token1": "○",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def infer_pair(path: Path) -> str:
    path_text = str(path)
    matches = [pair for pair, meta in PAIR_META.items() if meta["path"] in path_text]
    if len(matches) != 1:
        raise ValueError(f"Cannot infer token pair from {path}")
    return matches[0]


def infer_style(path: Path) -> str:
    if "v21_" in str(path):
        return "v21_cot"
    if "v9_" in str(path):
        return "v9_only_token"
    raise ValueError(f"Cannot infer style from {path}")


def load_rule(path: Path) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for row in read_csv(path):
        config = str(row["configuracao_binaria"]).zfill(7)
        choice = int(row["escolha"])
        if len(config) != 7 or set(config) - {"0", "1"} or choice not in (0, 1):
            raise ValueError(f"Invalid rule row in {path}: {row}")
        if config in lookup:
            raise ValueError(f"Duplicate configuration {config} in {path}")
        lookup[config] = choice
    expected = {format(value, "07b") for value in range(128)}
    if set(lookup) != expected:
        raise ValueError(f"Incomplete n=7 rule in {path}: {len(lookup)}/128")
    return lookup


def rule_metrics(path: Path) -> tuple[dict, list[dict]]:
    lookup = load_rule(path)
    pair = infer_pair(path)
    style = infer_style(path)
    processed: set[str] = set()
    pair_types: Counter[str] = Counter()
    for config, choice in lookup.items():
        if config in processed:
            continue
        complement = "".join("1" if bit == "0" else "0" for bit in config)
        pair_types[f"{choice}{lookup[complement]}"] += 1
        processed.update((config, complement))

    count_1 = sum(lookup.values())
    count_0 = len(lookup) - count_1
    complementary_pairs = pair_types["01"] + pair_types["10"]
    ici = complementary_pairs / 64
    token_bias = (count_1 - count_0) / 128
    majority_agreement = mean(
        choice == int(config.count("1") >= 4) for config, choice in lookup.items()
    )

    density_rows = []
    for ones in range(8):
        choices = [choice for config, choice in lookup.items() if config.count("1") == ones]
        density_rows.append(
            {
                "token_pair": pair,
                "style": style,
                "ones": ones,
                "local_token1_density": ones / 7,
                "configurations": len(choices),
                "choice_1_count": sum(choices),
                "p_choose_token1": mean(choices),
            }
        )

    metrics = {
        "token_pair": pair,
        "token0": PAIR_META[pair]["token0"],
        "token1": PAIR_META[pair]["token1"],
        "style": style,
        "ici": ici,
        "token_bias": token_bias,
        "abs_token_bias": abs(token_bias),
        "output_0_count": count_0,
        "output_1_count": count_1,
        "inversion_pairs": 64,
        "complementary_output_pairs": complementary_pairs,
        "same_00_pairs": pair_types["00"],
        "same_11_pairs": pair_types["11"],
        "majority_rule_agreement": majority_agreement,
        "f_all_0": lookup["0000000"],
        "f_all_1": lookup["1111111"],
        "preserves_both_homogeneous_states": (
            lookup["0000000"] == 0 and lookup["1111111"] == 1
        ),
        "rule_sha256": sha256(path),
        "rule_path": str(path),
    }
    if not math.isclose(abs(token_bias), abs((pair_types["11"] - pair_types["00"]) / 64)):
        raise AssertionError(f"Token-bias identity failed for {path}")
    return metrics, density_rows


def find_scoreboard(results_root: Path) -> Path:
    matches = sorted(
        (results_root / "cross_token" / "gemma-3-4b-it").glob(
            "token-pair-*/cell_scoreboard.csv"
        )
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one cross-token scoreboard, found {len(matches)}")
    return matches[0]


def final_state_density(results_root: Path, row: dict[str, str]) -> tuple[float, float, str]:
    pair = row["token_pair"]
    variant = row["variant"]
    memory_window = int(row["memory_window"])
    seed = int(row["seed"])
    pattern = (
        results_root
        / "phase3"
        / "gemma-3-4b-it"
        / PAIR_META[pair]["path"]
        / f"W={memory_window}"
        / variant
    )
    matches = list(pattern.glob(f"*/canonical/cells/seed_{seed:02d}/states.npy"))
    if len(matches) != 1:
        raise ValueError(f"Expected one states.npy for {pair}/{variant}/W={memory_window}/seed={seed}")
    states = np.load(matches[0])
    valid = states[np.isfinite(states).all(axis=1)]
    if len(valid) != int(row["completed_transitions"]) + 1:
        raise ValueError(f"Trajectory length mismatch in {matches[0]}")
    return float(valid[0].mean()), float(valid[-1].mean()), str(matches[0])


def simulation_metrics(
    results_root: Path, scoreboard: list[dict[str, str]], rules: list[dict]
) -> list[dict]:
    rule_index = {(row["token_pair"], row["style"]): row for row in rules}
    output = []
    for pair in PAIR_META:
        for style in ("v9_only_token", "v21_cot"):
            baseline = rule_index[(pair, style)]
            for memory_window in (0, 1):
                cells = [
                    row
                    for row in scoreboard
                    if row["token_pair"] == pair
                    and row["style"] == style
                    and int(row["memory_window"]) == memory_window
                ]
                if len(cells) != 20:
                    raise ValueError(f"Expected 20 cells for {pair}/{style}/W={memory_window}")
                initial_densities = []
                final_densities = []
                state_paths = []
                for cell in cells:
                    initial_density, final_density, state_path = final_state_density(
                        results_root, cell
                    )
                    initial_densities.append(initial_density)
                    final_densities.append(final_density)
                    state_paths.append(state_path)

                classifications = Counter(row["classification"] for row in cells)
                consensus_sides = Counter()
                for row in cells:
                    token = row["consensus_token"]
                    if token == PAIR_META[pair]["token0"]:
                        consensus_sides[0] += 1
                    elif token == PAIR_META[pair]["token1"]:
                        consensus_sides[1] += 1
                    elif token:
                        raise ValueError(f"Unexpected consensus token: {token}")
                if consensus_sides[0] == consensus_sides[1]:
                    dominant_side = ""
                else:
                    dominant_side = str(int(consensus_sides[1] > consensus_sides[0]))
                bias_side = "1" if baseline["token_bias"] > 0 else "0" if baseline["token_bias"] < 0 else ""
                output.append(
                    {
                        "token_pair": pair,
                        "style": style,
                        "memory_window": memory_window,
                        "static_rule_ici": baseline["ici"],
                        "static_rule_token_bias": baseline["token_bias"],
                        "static_bias_favored_side": bias_side,
                        "cells": len(cells),
                        "correct": classifications["correct"],
                        "wrong": classifications["wrong"],
                        "no_consensus": classifications["no_consensus"],
                        "consensus_rate": (consensus_sides[0] + consensus_sides[1]) / len(cells),
                        "token0_consensus": consensus_sides[0],
                        "token1_consensus": consensus_sides[1],
                        "dominant_consensus_side": dominant_side,
                        "bias_matches_dominant_consensus": (
                            bias_side == dominant_side if dominant_side else ""
                        ),
                        "mean_initial_token1_density": mean(initial_densities),
                        "mean_final_token1_density": mean(final_densities),
                        "mean_completed_transitions": mean(
                            int(row["completed_transitions"]) for row in cells
                        ),
                        "dct_bidirectional_pass": classifications["correct"] == 20,
                        "states_count": len(state_paths),
                    }
                )
    return output


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for index, _ in indexed[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    if left_ss == 0 or right_ss == 0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def correlation(rows: list[dict], scope: str, x: str, y: str) -> dict:
    left = [float(row[x]) for row in rows]
    right = [float(row[y]) for row in rows]
    return {
        "scope": scope,
        "x": x,
        "y": y,
        "n": len(rows),
        "pearson": pearson(left, right),
        "spearman": pearson(average_ranks(left), average_ranks(right)),
        "interpretation_guard": "descriptive_only_small_non_independent_sample",
    }


def render_report(rules: list[dict], simulations: list[dict], summary: dict) -> str:
    w0 = [row for row in simulations if row["memory_window"] == 0]
    lines = [
        "# ICI and Token Bias Audit - Gemma 3 4B, n=7",
        "",
        "ICI is the fraction of complement pairs satisfying `f(inv(x)) = 1 - f(x)`. ",
        "Token bias is `(outputs_1 - outputs_0) / 128`; positive values favor token 1.",
        "",
        "| Pair | Prompt | ICI | Bias | Outputs 0/1 | Majority agreement | W=0 result | Mean final density(1) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sim_index = {(row["token_pair"], row["style"]): row for row in w0}
    for rule in sorted(rules, key=lambda row: (row["token_pair"], row["style"])):
        sim = sim_index[(rule["token_pair"], rule["style"])]
        result = f"{sim['correct']}C/{sim['wrong']}W/{sim['no_consensus']}NC"
        lines.append(
            f"| {rule['token_pair']} | {rule['style']} | {rule['ici']:.6f} | "
            f"{rule['token_bias']:+.6f} | {rule['output_0_count']}/{rule['output_1_count']} | "
            f"{rule['majority_rule_agreement']:.6f} | {result} | "
            f"{sim['mean_final_token1_density']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- All {summary['rule_tables_complete']}/6 rule tables were complete and mechanically valid.",
            f"- At W=0, the sign of token bias matched the dominant consensus side in "
            f"{summary['w0_bias_attractor_matches']}/{summary['w0_groups_with_consensus']} groups. "
            "The k/z CoT match has only one consensus and is weak evidence.",
            f"- No static rule solved the bidirectional DCT: {summary['w0_dct_pass_groups']}/6 W=0 groups passed.",
            "- The highest ICI was k/z CoT, but it produced 19/20 no-consensus runs. "
            "ICI therefore measures inversion symmetry, not density-classification competence.",
            "- W=1 changed the policy. In k/z CoT, the static rule bias favored token 0, "
            "but the observed W=1 consensuses favored token 1. Static ICI/bias cannot fully explain a memory-bearing policy.",
            "- The correlations in analysis_summary.json are descriptive only (six related rule tables, no inferential claim).",
            "",
            "## Interpretation",
            "",
            "Token bias is useful as a directional diagnostic: it identifies the label toward which a frozen rule tends to drive the system. "
            "ICI is useful as a symmetry diagnostic: it identifies whether swapping all labels swaps the local decision. "
            "Neither index is sufficient for DCT success. Global behavior also depends on the spatial arrangement of rule outputs, preservation of homogeneous states, and temporal dynamics.",
            "",
            "For W=1, the reported ICI and token bias are explicitly the W=0 static-rule baseline, not metrics of the full memory-conditioned transition function.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rule_paths = sorted(
        (results_root / "phase1" / "gemma-3-4b-it").glob(
            "tokens=*/v*/rules*/canonical/rule.csv"
        )
    )
    if len(rule_paths) != 6:
        raise ValueError(f"Expected six n=7 rule tables, found {len(rule_paths)}")

    rules: list[dict] = []
    density_rows: list[dict] = []
    for path in rule_paths:
        metrics, density = rule_metrics(path)
        metrics["rule_path"] = str(path.relative_to(results_root))
        rules.append(metrics)
        density_rows.extend(density)

    scoreboard_path = find_scoreboard(results_root)
    scoreboard = read_csv(scoreboard_path)
    if len(scoreboard) != 240:
        raise ValueError(f"Expected 240 simulation cells, found {len(scoreboard)}")
    simulations = simulation_metrics(results_root, scoreboard, rules)
    w0 = [row for row in simulations if row["memory_window"] == 0]
    w1 = [row for row in simulations if row["memory_window"] == 1]

    correlations = [
        correlation(w0, "W=0", "static_rule_token_bias", "mean_final_token1_density"),
        correlation(w0, "W=0", "static_rule_ici", "consensus_rate"),
        correlation(w0, "W=0", "static_rule_ici", "mean_completed_transitions"),
        correlation(
            w1,
            "W=1_using_static_W0_metrics_as_baseline",
            "static_rule_token_bias",
            "mean_final_token1_density",
        ),
    ]
    summary = {
        "scope": {"model": "google/gemma-3-4b-it", "n": 7, "pairs": list(PAIR_META)},
        "formulas": {
            "ici": "complementary_output_pairs / 64",
            "token_bias": "(output_1_count - output_0_count) / 128",
        },
        "rule_tables_complete": len(rules),
        "rule_rows": sum(row["output_0_count"] + row["output_1_count"] for row in rules),
        "simulation_cells": len(scoreboard),
        "w0_groups_with_consensus": sum(row["consensus_rate"] > 0 for row in w0),
        "w0_bias_attractor_matches": sum(row["bias_matches_dominant_consensus"] is True for row in w0),
        "w1_groups_with_consensus": sum(row["consensus_rate"] > 0 for row in w1),
        "w1_bias_attractor_matches": sum(row["bias_matches_dominant_consensus"] is True for row in w1),
        "w0_dct_pass_groups": sum(row["dct_bidirectional_pass"] for row in w0),
        "w1_dct_pass_groups": sum(row["dct_bidirectional_pass"] for row in w1),
        "correlations": correlations,
        "scoreboard_sha256": sha256(scoreboard_path),
        "scoreboard_path": str(scoreboard_path.relative_to(results_root)),
        "limitations": [
            "Only six related rule tables; correlations are descriptive, not inferential.",
            "ICI and token bias characterize a static complete W=0 rule table.",
            "W=1 is history-conditioned, so its static-rule values are baseline covariates only.",
            "Twenty balanced seeds support attractor diagnosis but not a universal DCT claim.",
        ],
    }

    write_csv(output_dir / "n7_rule_metrics.csv", rules)
    write_csv(output_dir / "n7_density_response.csv", density_rows)
    write_csv(output_dir / "n7_simulation_linkage.csv", simulations)
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "README_N7_ICI_TOKEN_BIAS.md").write_text(
        render_report(rules, simulations, summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
