#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


MODES = (
    "standard_only_token",
    "standard_cot",
    "conformity_only_token",
    "conformity_cot",
)
MODE_TITLES = {
    "standard_only_token": "Gemma 3 4B | Standard | Only-token",
    "standard_cot": "Gemma 3 4B | Standard | CoT",
    "conformity_only_token": "Gemma 3 4B | Conformidade | Only-token",
    "conformity_cot": "Gemma 3 4B | Conformidade | CoT",
}
WINDOWS = range(0, 6)
SEEDS = range(1, 11)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_memory_log(path: Path, memory_window: int) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "=== MEMORY (Previous Rounds) ==="
    if memory_window == 0:
        if marker in text:
            raise RuntimeError(f"W=0 contains a MEMORY block: {path}")
        return
    if marker not in text:
        raise RuntimeError(f"W={memory_window} has no MEMORY block: {path}")
    for block in text.split(marker)[1:]:
        memory_text = block.split("\n\n", 1)[0]
        count = len(re.findall(r"^Round \d+:$", memory_text, flags=re.MULTILINE))
        if not 1 <= count <= memory_window:
            raise RuntimeError(
                f"Invalid memory depth {count} for W={memory_window}: {path}"
            )


def generate_slide(
    batch_root: Path,
    mode: str,
    memory_window: int,
    rows: list[dict[str, str]],
    expected_rounds: int,
) -> dict[str, Any]:
    status_colors = {
        "CORRETO": "#006d2c",
        "INCORRETO": "#a50f15",
        "SEM CONSENSO": "#4d4d4d",
    }
    state_cmap = ListedColormap(["#ffffff", "#111111"])
    state_cmap.set_bad("#d9d9d9")
    state_norm = BoundaryNorm([-0.5, 0.5, 1.5], state_cmap.N)
    index = {int(row["seed"]): row for row in rows}
    correct = sum(row["correct_consensus"] == "True" for row in rows)
    wrong = sum(
        row["consensus"] == "True" and row["correct_consensus"] != "True"
        for row in rows
    )
    no_consensus = len(rows) - correct - wrong

    fig, axes = plt.subplots(2, 5, figsize=(16, 9), dpi=240)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=0.045,
        right=0.985,
        bottom=0.075,
        top=0.735,
        wspace=0.20,
        hspace=0.43,
    )
    fig.text(
        0.045,
        0.952,
        MODE_TITLES[mode],
        ha="left",
        va="top",
        fontsize=23,
        fontweight="bold",
        color="#111111",
    )
    fig.text(
        0.045,
        0.902,
        f"N=30 | vizinhança=7 | memória W={memory_window} | tokens [0]/[1] | "
        f"seeds 1–10 | máximo={expected_rounds} rodadas | replay determinístico",
        ha="left",
        va="top",
        fontsize=14.5,
        color="#333333",
    )
    fig.text(
        0.955,
        0.945,
        f"{correct} corretos | {wrong} incorretos | {no_consensus} sem consenso",
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "#f2f2f2",
            "edgecolor": "#cccccc",
            "linewidth": 0.8,
        },
    )
    fig.legend(
        handles=[
            Patch(facecolor="#ffffff", edgecolor="#777777", label="Estado [0]"),
            Patch(facecolor="#111111", edgecolor="#111111", label="Estado [1]"),
            Patch(facecolor="#d9d9d9", edgecolor="#aaaaaa", label="Rodada não executada"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.042, 0.842),
        ncol=3,
        frameon=False,
        fontsize=12.5,
        handlelength=1.6,
        columnspacing=1.8,
    )
    fig.text(
        0.985,
        0.821,
        "E = esperado | F = consenso final | ímpar: maioria [1] | par: maioria [0]",
        ha="right",
        va="center",
        fontsize=11.5,
        color="#444444",
    )

    state_entries: list[dict[str, Any]] = []
    for seed, ax in zip(SEEDS, axes.flat):
        row = index[seed]
        state_path = (
            batch_root
            / "strategies"
            / mode
            / f"W{memory_window}"
            / "cells"
            / f"seed_{seed:02d}"
            / "states.npy"
        )
        states = np.load(state_path)
        if states.shape != (expected_rounds, 30):
            raise RuntimeError(f"Unexpected state shape {states.shape}: {state_path}")
        ax.imshow(
            states,
            cmap=state_cmap,
            norm=state_norm,
            origin="upper",
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xlim(-0.5, 29.5)
        ax.set_ylim(expected_rounds - 0.5, -0.5)
        ax.set_xticks([0, 10, 20, 29])
        ax.set_yticks(
            np.unique(
                np.linspace(0, expected_rounds - 1, min(10, expected_rounds), dtype=int)
            )
        )
        ax.tick_params(axis="both", labelsize=10, length=3, width=0.7, colors="#333333")
        ax.set_xlabel("Agentes", fontsize=11, labelpad=3)
        ax.set_ylabel("Rodadas" if seed in (1, 6) else "", fontsize=11, labelpad=3)
        ax.set_xticks(np.arange(-0.5, 30, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, expected_rounds, 1), minor=True)
        ax.grid(which="minor", color="#bdbdbd", linewidth=0.22, alpha=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.8)

        if row["correct_consensus"] == "True":
            status = "CORRETO"
        elif row["consensus"] == "True":
            status = "INCORRETO"
        else:
            status = "SEM CONSENSO"
        final_token = row["consensus_token"] or "–"
        ax.set_title(
            f"S{seed:02d} | E[{row['expected_token']}] | F[{final_token}]\n{status}",
            fontsize=12.2,
            fontweight="bold",
            color=status_colors[status],
            pad=7,
        )
        state_entries.append(
            {
                "seed": seed,
                "path": str(state_path),
                "sha256": sha256_file(state_path),
                "expected_token": row["expected_token"],
                "final_token": row["consensus_token"] or None,
                "status": status,
            }
        )

    output_dir = batch_root / "slides_4k" / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"W{memory_window}.png"
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)
    return {
        "mode": mode,
        "memory_window": memory_window,
        "path": str(output),
        "pixels": [3840, 2160],
        "sha256": sha256_file(output),
        "states": state_entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--windows", default="0,1,2,3,4,5")
    parser.add_argument("--expected-rounds", type=int, default=10)
    return parser.parse_args()


def parse_windows(value: str) -> list[int]:
    windows = [int(item) for item in value.split(",")]
    if not windows or any(window not in WINDOWS for window in windows):
        raise ValueError("--windows must contain values from 0 through 5")
    if len(windows) != len(set(windows)):
        raise ValueError("--windows cannot contain duplicates")
    return windows


def main() -> int:
    args = parse_args()
    if args.expected_rounds < 1:
        raise ValueError("--expected-rounds must be a positive integer")
    windows = parse_windows(args.windows)
    expected_per_window = len(args.modes) * len(SEEDS)
    expected_total = len(windows) * expected_per_window
    summary_rows: list[dict[str, Any]] = []
    slide_manifest: list[dict[str, Any]] = []
    paired_rows: list[dict[str, str]] = []

    for memory_window in windows:
        proof_dir = args.batch_root / "determinism_proof" / f"W{memory_window}"
        proof = json.loads((proof_dir / "summary.json").read_text(encoding="utf-8"))
        if (
            not proof["determinism_gate_pass"]
            or proof["exact_equal"] != expected_per_window
        ):
            raise RuntimeError(f"Determinism gate not met for W={memory_window}")
        paired_rows.extend(read_csv(proof_dir / "paired_determinism_comparison.csv"))

        for mode in args.modes:
            run_dir = args.batch_root / "strategies" / mode / f"W{memory_window}"
            rows = read_csv(run_dir / "scoreboard.csv")
            if len(rows) != 10 or {int(row["seed"]) for row in rows} != set(SEEDS):
                raise RuntimeError(f"Incomplete seeds: {run_dir}")
            if any(int(row["memory_window"]) != memory_window for row in rows):
                raise RuntimeError(f"Wrong memory_window: {run_dir}")
            if any(int(row["parse_failure_count"]) != 0 for row in rows):
                raise RuntimeError(f"Parse failure found: {run_dir}")
            if any(int(row.get("request_failure_count", 0)) != 0 for row in rows):
                raise RuntimeError(f"Request failure found: {run_dir}")
            if any(str(row["status"]).startswith("Erro") for row in rows):
                raise RuntimeError(f"Simulation error found: {run_dir}")
            if any(int(row["rounds"]) != args.expected_rounds for row in rows):
                raise RuntimeError(f"Wrong round limit: {run_dir}")
            for seed in SEEDS:
                validate_memory_log(
                    run_dir / "cells" / f"seed_{seed:02d}" / "prompt_log_normalized.txt",
                    memory_window,
                )

            correct = sum(row["correct_consensus"] == "True" for row in rows)
            wrong = sum(
                row["consensus"] == "True" and row["correct_consensus"] != "True"
                for row in rows
            )
            no_consensus = len(rows) - correct - wrong
            summary_rows.append(
                {
                    "mode": mode,
                    "memory_window": memory_window,
                    "completed": len(rows),
                    "correct_consensus": correct,
                    "wrong_consensus": wrong,
                    "no_consensus": no_consensus,
                    "parse_failures": sum(int(row["parse_failure_count"]) for row in rows),
                    "determinism_exact": proof["mode_summary"][mode]["exact_equal"],
                }
            )
            slide_manifest.append(
                generate_slide(
                    args.batch_root,
                    mode,
                    memory_window,
                    rows,
                    args.expected_rounds,
                )
            )

    if len(paired_rows) != expected_total or not all(
        row["exact_equal"] == "True" for row in paired_rows
    ):
        raise RuntimeError(
            f"Global {expected_total}/{expected_total} determinism gate not met"
        )
    write_csv(args.batch_root / "batch_summary.csv", summary_rows)
    write_csv(args.batch_root / "paired_determinism_comparison.csv", paired_rows)
    (args.batch_root / "slides_4k" / "manifest.json").write_text(
        json.dumps(slide_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "determinism_gate_pass": True,
        "determinism_exact": expected_total,
        "expected_cells": expected_total,
        "retained_state_files": len(list(args.batch_root.glob("strategies/*/W*/cells/seed_*/states.npy"))),
        "retained_png_files": len(list(args.batch_root.glob("strategies/*/W*/seed_*.png"))),
        "slides_4k": len(list(args.batch_root.glob("slides_4k/*/W*.png"))),
        "strategy_window_summary": summary_rows,
        "modes": args.modes,
        "windows": windows,
        "max_rounds": args.expected_rounds,
    }
    if (
        summary["retained_state_files"] != expected_total
        or summary["retained_png_files"] != expected_total
        or summary["slides_4k"] != len(windows) * len(args.modes)
    ):
        raise RuntimeError(f"Final artifact count mismatch: {summary}")
    (args.batch_root / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Gemma 3 4B - Memory Batch Summary",
        "",
        "- Determinism gate: `PASS`",
        f"- Modes: `{', '.join(args.modes)}`",
        f"- Windows: `{','.join(str(window) for window in windows)}`",
        f"- Maximum rounds: `{args.expected_rounds}`",
        f"- Exact replay cells: `{expected_total}/{expected_total}`",
        f"- Retained trajectories: `{expected_total}`",
        f"- Retained individual PNGs: `{expected_total}`",
        f"- Google Slides 4K: `{len(windows) * len(args.modes)}`",
        "",
        "| Mode | W | Correct | Wrong | No consensus | Exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['mode']} | {row['memory_window']} | {row['correct_consensus']} | "
            f"{row['wrong_consensus']} | {row['no_consensus']} | {row['determinism_exact']}/10 |"
        )
    (args.batch_root / "batch_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
