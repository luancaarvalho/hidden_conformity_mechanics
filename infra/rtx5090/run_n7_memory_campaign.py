#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gradio_project.memory.run_memory_online import (  # noqa: E402
    CellSpec,
    finalize_manifest_groups,
    read_manifest,
    run_manifest_replay,
)
from utils.parity_artifacts import atomic_append_jsonl, write_json, write_jsonl  # noqa: E402
from utils.w0_parity_contract import (  # noqa: E402
    build_responses_payload,
    query_responses_with_retries,
    render_memory_prompt,
    token_pair_spec,
)


MODEL = "gemma3-4b-temp0"
MODEL_FAMILY = "gemma-3-4b-it"
BASE_URL = "http://127.0.0.1:8127/v1"
PAIRS = ("01", "kz", "triangle_circle")
RULE_RUN_IDS = {
    "01": "rules-ca-parity-w0-seed1-20_20260720T143020Z",
    "kz": "rules-ca-parity-w0-kz-seed1-20_20260720T200526Z",
    "triangle_circle": "rules-ca-parity-w0-triangle_circle-seed1-20_20260720T200526Z",
}
OLD_MEMORY_RUN_IDS = {
    "01": {
        0: "rules-ca-parity-w0-seed1-20_20260720T143020Z",
        1: "memory-W1-seed1-20_20260720T174027Z",
    },
    "kz": {
        0: "rules-ca-parity-w0-kz-seed1-20_20260720T200526Z",
        1: "memory-W1-kz-seed1-20_20260720T200526Z",
    },
    "triangle_circle": {
        0: "rules-ca-parity-w0-triangle_circle-seed1-20_20260720T200526Z",
        1: "memory-W1-triangle_circle-seed1-20_20260720T200526Z",
    },
}
MIN_FREE_BYTES = 20 * 1024**3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Campaign:
    def __init__(self, campaign_id: str) -> None:
        self.campaign_id = campaign_id
        self.timestamp = campaign_id.rsplit("_", 1)[-1]
        self.root = REPO_ROOT / "artifacts/orchestration/rtx5090" / campaign_id
        self.manifest_dir = self.root / "manifests"
        self.work_root = (
            REPO_ROOT / "artifacts/work/rtx5090/n=7" / campaign_id / "phase3"
        )
        self.stage_path = self.root / "stage_status.jsonl"
        self.progress_path = self.root / "cell_status.jsonl"
        self.heartbeat_path = self.root / "heartbeat.json"
        self.terminal_path = self.root / "terminal_status.json"
        self.stop_heartbeat = threading.Event()
        self.run_ids = {
            window: (
                f"memory-W{window}-seed21-50_{self.timestamp}"
                if window in (0, 1)
                else f"memory-W{window}-seed1-50_{self.timestamp}"
            )
            for window in range(6)
        }

    def stage(self, name: str, status: str, **details: Any) -> None:
        row = {"utc": utc_now(), "stage": name, "status": status, **details}
        atomic_append_jsonl(self.stage_path, row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def heartbeat(self) -> None:
        while not self.stop_heartbeat.wait(30):
            self.write_heartbeat(alive=True)

    def write_heartbeat(self, *, alive: bool) -> None:
        write_json(
            self.heartbeat_path,
            {
                "utc": utc_now(),
                "campaign_id": self.campaign_id,
                "pid": os.getpid(),
                "alive": alive,
            },
        )

    def start_heartbeat(self) -> threading.Thread:
        self.write_heartbeat(alive=True)
        thread = threading.Thread(target=self.heartbeat, daemon=True)
        thread.start()
        return thread

    def terminal(self, status: str, **details: Any) -> None:
        write_json(
            self.terminal_path,
            {
                "utc": utc_now(),
                "campaign_id": self.campaign_id,
                "status": status,
                "complete": True,
                **details,
            },
        )
        self.stage("campaign", status, **details)

    def manifest_path(self, window: int) -> Path:
        return self.manifest_dir / f"W{window}.jsonl"

    def specs_for_window(self, window: int) -> list[CellSpec]:
        seeds = range(21, 51) if window in (0, 1) else range(1, 51)
        return [
            CellSpec(
                cell_id=f"W{window}_{pair}_{variant}_seed{seed:02d}",
                run_id=self.run_ids[window],
                token_pair=pair,
                variant=variant,
                memory_window=window,
                seed=seed,
            )
            for pair in PAIRS
            for variant in token_pair_spec(pair)["variants"]
            for seed in seeds
        ]

    def create_manifests(self) -> None:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        all_rows = []
        for window in range(6):
            rows = [spec.__dict__ for spec in self.specs_for_window(window)]
            path = self.manifest_path(window)
            if path.exists():
                existing = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if existing != rows:
                    raise RuntimeError(f"manifest changed across resume: {path}")
            else:
                write_jsonl(path, rows)
            all_rows.extend(rows)
        if len(all_rows) != 1560:
            raise AssertionError(f"unexpected new cell count: {len(all_rows)}")
        campaign_manifest = self.root / "campaign_manifest.json"
        previous = (
            json.loads(campaign_manifest.read_text(encoding="utf-8"))
            if campaign_manifest.exists()
            else None
        )
        manifest = {
            "campaign_id": self.campaign_id,
            "created_at_utc": previous["created_at_utc"] if previous else utc_now(),
            "git_commit": git_output("rev-parse", "HEAD").strip(),
            "model": MODEL,
            "model_family": MODEL_FAMILY,
            "base_url": BASE_URL,
            "agents": 30,
            "neighbors_including_center": 7,
            "majority_ratio": 0.51,
            "max_transitions": 60,
            "stop_conditions": ["consensus", "max_rounds"],
            "cell_workers": 2,
            "request_workers": 32,
            "new_unique_cells": len(all_rows),
            "new_replay_trajectories": len(all_rows) * 2,
            "run_ids": self.run_ids,
            "rule_run_ids": RULE_RUN_IDS,
            "old_memory_run_ids": OLD_MEMORY_RUN_IDS,
        }
        manifest = json.loads(json.dumps(manifest, ensure_ascii=False))
        if previous is not None:
            if previous != manifest:
                raise RuntimeError("campaign manifest changed across resume")
        else:
            write_json(campaign_manifest, manifest)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def capture(command: list[str], path: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")
    return result


def vllm_process() -> tuple[int, str]:
    sockets = subprocess.check_output(["ss", "-ltnp"], text=True)
    listeners = [line for line in sockets.splitlines() if re.search(r":8127\s", line)]
    pids = {
        int(match.group(1))
        for line in listeners
        if (match := re.search(r"pid=(\d+)", line)) is not None
    }
    if len(pids) != 1:
        raise RuntimeError(f"expected one listener on 8127, found PIDs {sorted(pids)}")
    pid = pids.pop()
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    if "vllm" not in command or "serve" not in command:
        raise RuntimeError(f"port 8127 is not served by vLLM: {pid} {command}")
    return pid, command


def endpoint_ready() -> bool:
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=10)
        response.raise_for_status()
        models = {item["id"] for item in response.json()["data"]}
        return MODEL in models
    except Exception:
        return False


def wait_for_endpoint(timeout_s: int = 1800) -> None:
    deadline = time.monotonic() + timeout_s
    delay = 5
    while time.monotonic() < deadline:
        if endpoint_ready():
            return
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise RuntimeError("FAIL_ENDPOINT: endpoint unavailable for 30 minutes")


def verify_disk() -> None:
    free = shutil.disk_usage(REPO_ROOT).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"FAIL_DISK: only {free / 1024**3:.1f} GiB free")


def preflight(campaign: Campaign) -> None:
    root = campaign.root / "preflight"
    root.mkdir(parents=True, exist_ok=True)
    if git_output("status", "--porcelain").strip():
        raise RuntimeError("preflight requires a clean Git worktree")
    (root / "git_commit.txt").write_text(git_output("rev-parse", "HEAD"), encoding="utf-8")
    (root / "git_status.txt").write_text(git_output("status", "--short", "--branch"), encoding="utf-8")
    capture(["nvidia-smi"], root / "nvidia_smi.txt")
    capture(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.free,memory.total",
            "--format=csv,noheader",
        ],
        root / "gpu_summary.csv",
    )
    capture(["ps", "-eo", "pid,ppid,etime,%cpu,%mem,args"], root / "processes.txt")
    capture(["ss", "-ltnp"], root / "ports.txt")
    capture(["tmux", "ls"], root / "tmux.txt", check=False)
    wait_for_endpoint(60)
    response = requests.get(f"{BASE_URL}/models", timeout=10)
    write_json(root / "vllm_model.json", response.json())
    pid, command = vllm_process()
    environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    env = {item.decode() for item in environment if item.startswith(b"VLLM_")}
    required_env = {"VLLM_BATCH_INVARIANT=1", "VLLM_USE_FLASHINFER_SAMPLER=0"}
    if not required_env.issubset(env):
        raise RuntimeError(f"missing deterministic vLLM environment: {required_env - env}")
    for flag in ("--generation-config vllm", "--seed 42", "--max-num-seqs 32"):
        if flag not in command:
            raise RuntimeError(f"vLLM missing {flag}")
    (root / "vllm_process.txt").write_text(f"{pid} {command}\n", encoding="utf-8")
    (root / "vllm_environment.txt").write_text("\n".join(sorted(env)) + "\n", encoding="utf-8")
    verify_disk()
    python = str(REPO_ROOT / "artifacts/conda/runtime/bin/python")
    sources = [
        "utils/w0_parity_contract.py",
        "gradio_project/memory/run_memory_online.py",
        "experimentos_automatos/runtime/run_w0_parity_automata.py",
        "infra/rtx5090/run_n7_memory_campaign.py",
        "infra/rtx5090/proofread_n7_memory_campaign.py",
    ]
    capture([python, "-m", "py_compile", *sources], root / "py_compile.txt")
    capture(
        [python, "-m", "unittest", "discover", "-s", "extract_rules/runtime_vllm/tests", "-p", "test_*.py"],
        root / "test_extract_rules.txt",
    )
    capture(
        [python, "-m", "unittest", "discover", "-s", "experimentos_automatos/runtime/tests", "-p", "test_*.py"],
        root / "test_automata.txt",
    )
    capture(["bash", "gradio_project/tests/run_tests.sh"], root / "test_gradio.txt")


def run_canaries(campaign: Campaign) -> None:
    neighborhood = [0, 1, 0, 0, 1, 1, 0]
    cases = []
    for pair in PAIRS:
        for variant in token_pair_spec(pair)["variants"]:
            for window in (0, 5):
                memory = [] if window == 0 else [(index, neighborhood) for index in range(5)]
                system, user = render_memory_prompt(
                    variant, neighborhood, memory_snapshots=memory
                )
                payload = build_responses_payload(
                    model=MODEL,
                    system_prompt=system,
                    user_prompt=user,
                    variant=variant,
                )
                cases.append((pair, variant, window, system, user, payload))

    def query(case: tuple[Any, ...]) -> dict[str, Any]:
        pair, variant, window, system, user, payload = case
        result = query_responses_with_retries(
            base_url=BASE_URL,
            payload=payload,
            variant=variant,
            timeout_s=300,
            max_attempts=2,
        )
        return {
            "token_pair": pair,
            "variant": variant,
            "memory_window": window,
            "system_prompt": system,
            "user_prompt": user,
            "payload": payload,
            "result": result,
        }

    isolated = [[query(case), query(case)] for case in cases]
    with ThreadPoolExecutor(max_workers=32) as pool:
        concurrent = list(pool.map(query, cases + cases))
    for index, pair in enumerate(isolated):
        expected = pair[0]["result"]
        candidates = [pair[1]["result"], concurrent[index]["result"], concurrent[index + len(cases)]["result"]]
        if expected["choice"] not in (0, 1) or any(
            candidate["choice"] != expected["choice"]
            or candidate["raw_response"] != expected["raw_response"]
            for candidate in candidates
        ):
            raise RuntimeError(f"determinism canary failed: {cases[index][:3]}")
    write_json(
        campaign.root / "preflight/canaries.json",
        {"isolated": isolated, "concurrent": concurrent},
    )


def automaton_final_root(pair: str, variant: str, run_id: str) -> Path:
    return (
        REPO_ROOT
        / "artifacts/phase2_cellular_automata/rtx5090/n=7"
        / MODEL_FAMILY
        / token_pair_spec(pair)["folder"]
        / variant
        / run_id
    )


def isolate_automaton_work(campaign: Campaign, pair: str, variant: str) -> None:
    work = (
        REPO_ROOT
        / "artifacts/work/rtx5090/n=7"
        / campaign.run_ids[0]
        / "phase2"
        / pair
        / variant
    )
    if not work.exists():
        return
    quarantine = campaign.root / "quarantine" / "phase2" / pair / variant / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(work), str(quarantine))


def run_w0_automata(campaign: Campaign) -> None:
    python = str(REPO_ROOT / "artifacts/conda/runtime/bin/python")
    for pair in PAIRS:
        for variant in token_pair_spec(pair)["variants"]:
            destination = automaton_final_root(pair, variant, campaign.run_ids[0])
            if destination.is_dir() and (destination / "status.json").is_file():
                continue
            isolate_automaton_work(campaign, pair, variant)
            command = [
                python,
                "experimentos_automatos/runtime/run_w0_parity_automata.py",
                "--run-id",
                campaign.run_ids[0],
                "--rule-run-id",
                RULE_RUN_IDS[pair],
                "--token-pair",
                pair,
                "--variants",
                variant,
                "--agents",
                "30",
                "--seeds",
                "21-50",
                "--majority-ratio",
                "0.51",
                "--max-transitions",
                "60",
            ]
            result = subprocess.run(command, cwd=REPO_ROOT)
            if result.returncode not in (0, 2):
                raise RuntimeError(f"automaton failed for {pair}/{variant}: {result.returncode}")


def proofread(campaign: Campaign, stage: str) -> int:
    python = str(REPO_ROOT / "artifacts/conda/runtime/bin/python")
    result = subprocess.run(
        [
            python,
            "infra/rtx5090/proofread_n7_memory_campaign.py",
            "--campaign-id",
            campaign.campaign_id,
            "--stage",
            stage,
        ],
        cwd=REPO_ROOT,
    )
    return result.returncode


def execute(campaign: Campaign) -> None:
    campaign.root.mkdir(parents=True, exist_ok=True)
    heartbeat_thread = campaign.start_heartbeat()
    try:
        campaign.create_manifests()
        campaign.stage("preflight", "STARTED")
        preflight(campaign)
        run_canaries(campaign)
        campaign.stage("preflight", "PASS")

        for window in range(6):
            verify_disk()
            wait_for_endpoint()
            campaign.stage(f"W{window}", "STARTED")
            if window == 0:
                campaign.stage("W0_automata", "STARTED")
                run_w0_automata(campaign)
                campaign.stage("W0_automata", "COMPLETE")
            specs = read_manifest(campaign.manifest_path(window))
            replay_results = []
            for replay in (1, 2):
                campaign.stage(f"W{window}_replay_{replay}", "STARTED")
                result = run_manifest_replay(
                    specs=specs,
                    replay=replay,
                    work_root=campaign.work_root,
                    progress_path=campaign.progress_path,
                    stage_status_path=campaign.stage_path,
                    base_url=BASE_URL,
                    model=MODEL,
                    cell_workers=2,
                    request_workers=32,
                    timeout_s=300,
                    max_attempts=5,
                )
                replay_results.append(result)
                campaign.stage(f"W{window}_replay_{replay}", "COMPLETE", **result)
            groups = finalize_manifest_groups(specs, campaign.work_root)
            deterministic = sum(row.get("status") == "PASS" for row in groups)
            campaign.stage(
                f"W{window}_determinism",
                "PASS" if deterministic == 6 else "FAIL_NONDETERMINISTIC",
                deterministic_groups=deterministic,
                expected_groups=6,
                replay_results=replay_results,
            )
            if window == 0:
                status = proofread(campaign, "w0-extension")
                campaign.stage("W0_parity", "PASS" if status == 0 else "FAIL_PARITY")
            campaign.stage(f"W{window}", "COMPLETE")

        final_status = proofread(campaign, "final")
        campaign.terminal("PASS" if final_status == 0 else "FAIL_PROOFREAD", proofread_exit=final_status)
    except Exception as exc:
        campaign.terminal("INCOMPLETE", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        campaign.stop_heartbeat.set()
        heartbeat_thread.join(timeout=2)
        campaign.write_heartbeat(alive=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign = Campaign(args.campaign_id)
    execute(campaign)
    status = json.loads(campaign.terminal_path.read_text(encoding="utf-8"))["status"]
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
