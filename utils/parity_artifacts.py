#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def finalize_replays(
    *,
    replay1: Path,
    replay2: Path,
    proof_dir: Path,
    final_root: Path,
    deterministic: bool,
    status_details: dict[str, Any],
) -> None:
    if final_root.exists():
        raise FileExistsError(final_root)
    final_root.mkdir(parents=True)
    shutil.move(str(proof_dir), str(final_root / "determinism"))

    if deterministic:
        shutil.move(str(replay1), str(final_root / "canonical"))
        shutil.rmtree(replay2)
        status = "PASS"
    else:
        shutil.move(str(replay1), str(final_root / "replay_01"))
        shutil.move(str(replay2), str(final_root / "replay_02"))
        status = "FAIL_NONDETERMINISTIC"

    write_json(final_root / "status.json", {"status": status, **status_details})
    for parent in {replay1.parent, replay2.parent}:
        try:
            parent.rmdir()
        except OSError:
            pass


def first_array_difference(a: Any, b: Any) -> tuple[int | None, int | None]:
    import numpy as np

    if a.shape != b.shape:
        return -1, -1
    differences = np.argwhere(~np.isclose(a, b, equal_nan=True))
    if not len(differences):
        return None, None
    row, column = differences[0]
    return int(row), int(column)


def build_artifact_index(paths: Iterable[Path], output_path: Path) -> int:
    rows: list[dict[str, Any]] = []
    for root in paths:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path == output_path:
                continue
            rows.append(
                {
                    "path": str(path),
                    "phase": next((part for part in path.parts if part.startswith("phase")), "cross_phase"),
                    "artifact_type": path.suffix.lstrip(".") or "file",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_jsonl(output_path, rows)
    return len(rows)


def atomic_append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)
