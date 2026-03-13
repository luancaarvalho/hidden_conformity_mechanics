#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = _THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompt_strategies import get_prompt_strategy
from projeto_final.streamlit_test.llm_sim_runner import (
    format_memory_block_timeline,
    get_neighbor_indices,
    state_to_token,
    token_pair_for_prompt_variant,
)
from projeto_final.utils.utils import call_llm_responses, generate_initial_distribution_shared, parse_opinion_token


def _default_opinion_like_interface(prompt_variant: str) -> str:
    v = (prompt_variant or "").strip()
    if "_ab" in v:
        return "a"
    if "_01" in v:
        return "0"
    if "_αβ" in v:
        return "α"
    if "_△○" in v:
        return "△"
    if "_⊕⊖" in v:
        return "⊕"
    if "_pq" in v:
        return "p"
    if "_łþ" in v:
        return "ł"
    if "_yesno" in v:
        return "no"
    return "k"


def _format_memory_block_like_interface(
    states: np.ndarray,
    *,
    n_agents: int,
    n_neighbors: int,
    agent_idx: int,
    current_round: int,
    memory_window: int,
    token0: str,
    token1: str,
) -> str:
    if memory_window == 0 or current_round == 0:
        return ""

    start_round = max(0, current_round - memory_window)
    end_round = current_round
    if start_round >= end_round:
        return ""

    left_indices, right_indices = get_neighbor_indices(n_agents, n_neighbors, agent_idx)
    lines: List[str] = []
    lines.append("=== MEMORY (Previous Rounds) ===")
    for r in range(start_round, end_round):
        self_token = state_to_token(states[r, agent_idx], token0, token1)
        left_tokens = [state_to_token(states[r, idx], token0, token1) for idx in left_indices]
        right_tokens = [state_to_token(states[r, idx], token0, token1) for idx in right_indices]
        lines.append(f"Round {r}:")
        lines.append(f"  You: [{self_token}]")
        lines.append(f"  Left: {left_tokens}")
        lines.append(f"  Right: {right_tokens}")
    return "\n".join(lines)


def _build_round1_prompt(
    *,
    prompt_variant: str,
    n_agents: int,
    n_neighbors: int,
    seed_distribution: int,
    initial_majority_ratio: float,
    initial_distribution_mode: str,
    memory_window: int,
    agent_index: int,
    current_opinion_mode: str,
) -> Dict[str, Any]:
    token0, token1 = token_pair_for_prompt_variant(prompt_variant)
    yaml_path = REPO_ROOT / "prompt_templates.yaml"
    strategy = get_prompt_strategy(prompt_variant, str(yaml_path))

    dist = generate_initial_distribution_shared(
        n_agents=n_agents,
        seed_distribution=seed_distribution,
        majority_ratio=initial_majority_ratio,
        requested_mode=initial_distribution_mode,
    )

    states = np.full((2, n_agents), np.nan, dtype=np.float32)
    states[0, :] = dist.opinions.astype(np.float32)

    left_indices, right_indices = get_neighbor_indices(n_agents, n_neighbors, agent_index)
    left = [state_to_token(states[0, idx], token0, token1) for idx in left_indices if not np.isnan(states[0, idx])]
    right = [state_to_token(states[0, idx], token0, token1) for idx in right_indices if not np.isnan(states[0, idx])]

    current_val = states[0, agent_index]
    if not np.isnan(current_val):
        current_opinion = state_to_token(current_val, token0, token1)
    elif current_opinion_mode == "batch":
        current_opinion = token0
    else:
        current_opinion = _default_opinion_like_interface(prompt_variant)

    system_prompt, user_prompt = strategy.build_prompt(
        left=left,
        right=right,
        current_opinion=current_opinion,
    )

    if current_opinion_mode == "batch":
        mem_block = format_memory_block_timeline(
            states,
            n_agents=n_agents,
            n_neighbors=n_neighbors,
            agent_idx=agent_index,
            current_round=1,
            memory_window=memory_window,
            token0=token0,
            token1=token1,
        )
    else:
        mem_block = _format_memory_block_like_interface(
            states,
            n_agents=n_agents,
            n_neighbors=n_neighbors,
            agent_idx=agent_index,
            current_round=1,
            memory_window=memory_window,
            token0=token0,
            token1=token1,
        )

    if mem_block:
        user_prompt = mem_block + "\n\n" + user_prompt

    return {
        "token_pair": [token0, token1],
        "distribution": {
            "seed_distribution": seed_distribution,
            "mode_resolved": dist.mode,
            "initial_majority_opinion": int(dist.initial_majority_opinion),
            "count_0": int(dist.count_0),
            "count_1": int(dist.count_1),
            "opinions_round0": dist.opinions.astype(int).tolist(),
        },
        "agent": {
            "agent_number_1based": int(agent_index + 1),
            "agent_index_0based": int(agent_index),
            "left_neighbors": left,
            "right_neighbors": right,
            "current_opinion": current_opinion,
        },
        "prompt": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "memory_block": mem_block,
        },
    }


def _run_llm_call(
    *,
    base_url: str,
    model: str,
    temperature: float,
    request_seed: Optional[int],
    max_tokens: int,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    result = call_llm_responses(
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        seed=request_seed,
        max_output_tokens=max_tokens,
        timeout_s=120,
    )
    parsed = parse_opinion_token(
        result.get("raw_response", ""),
        allowed_tokens=("k", "z", "a", "b", "0", "1", "p", "q", "α", "β", "△", "○", "⊕", "⊖", "ł", "þ", "no", "yes"),
        prefer_last=True,
    )
    return {
        "http_status": result.get("http_status"),
        "request_elapsed_s": result.get("request_elapsed_s"),
        "response_model": result.get("response_model"),
        "usage": result.get("usage"),
        "payload": result.get("payload"),
        "raw_response": result.get("raw_response"),
        "parsed_token": parsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe round-1 prompt/call for batch vs interface-v3 semantics.")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="google/gemma-3-4b")
    parser.add_argument("--prompt-variant", default="v9_lista_completa_meio_kz")
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--neighbors", type=int, default=7)
    parser.add_argument("--initial-majority-ratio", type=float, default=0.51)
    parser.add_argument("--initial-distribution-mode", choices=["auto", "ratio", "half_split"], default="ratio")
    parser.add_argument("--seed-distribution", type=int, default=1)
    parser.add_argument("--memory-windows", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--agent-number", type=int, default=1, help="1-based agent number")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--no-send", action="store_true", help="Only build prompt/payload preview, do not call LLM")
    parser.add_argument(
        "--output-dir",
        default=str(_THIS_FILE.parent / "batch_outputs" / "prompt_probe"),
    )
    args = parser.parse_args()

    if args.agent_number < 1 or args.agent_number > args.agents:
        raise ValueError(f"agent_number must be in [1, {args.agents}]")

    agent_index = args.agent_number - 1
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "timestamp": ts,
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "prompt_variant": args.prompt_variant,
            "agents": args.agents,
            "neighbors": args.neighbors,
            "initial_majority_ratio": args.initial_majority_ratio,
            "initial_distribution_mode": args.initial_distribution_mode,
            "seed_distribution": args.seed_distribution,
            "memory_windows": args.memory_windows,
            "agent_number_1based": args.agent_number,
            "agent_index_0based": agent_index,
            "temperature": args.temperature,
            "request_seed": args.request_seed,
            "max_tokens": args.max_tokens,
            "no_send": bool(args.no_send),
        },
        "windows": {},
    }

    for w in args.memory_windows:
        batch_built = _build_round1_prompt(
            prompt_variant=args.prompt_variant,
            n_agents=args.agents,
            n_neighbors=args.neighbors,
            seed_distribution=args.seed_distribution,
            initial_majority_ratio=args.initial_majority_ratio,
            initial_distribution_mode=args.initial_distribution_mode,
            memory_window=w,
            agent_index=agent_index,
            current_opinion_mode="batch",
        )
        iface_built = _build_round1_prompt(
            prompt_variant=args.prompt_variant,
            n_agents=args.agents,
            n_neighbors=args.neighbors,
            seed_distribution=args.seed_distribution,
            initial_majority_ratio=args.initial_majority_ratio,
            initial_distribution_mode=args.initial_distribution_mode,
            memory_window=w,
            agent_index=agent_index,
            current_opinion_mode="interface",
        )

        sys_equal = batch_built["prompt"]["system_prompt"] == iface_built["prompt"]["system_prompt"]
        usr_equal = batch_built["prompt"]["user_prompt"] == iface_built["prompt"]["user_prompt"]

        diff_text = ""
        if not (sys_equal and usr_equal):
            batch_full = "[SYSTEM]\n" + batch_built["prompt"]["system_prompt"] + "\n\n[USER]\n" + batch_built["prompt"]["user_prompt"]
            iface_full = "[SYSTEM]\n" + iface_built["prompt"]["system_prompt"] + "\n\n[USER]\n" + iface_built["prompt"]["user_prompt"]
            diff_text = "\n".join(
                difflib.unified_diff(
                    batch_full.splitlines(),
                    iface_full.splitlines(),
                    fromfile="batch",
                    tofile="interface_v3",
                    lineterm="",
                )
            )

        item: Dict[str, Any] = {
            "memory_window": w,
            "prompt_equal": {"system": sys_equal, "user": usr_equal},
            "batch": batch_built,
            "interface_v3": iface_built,
            "prompt_diff": diff_text,
        }

        if not args.no_send:
            item["batch"]["call"] = _run_llm_call(
                base_url=args.base_url,
                model=args.model,
                temperature=args.temperature,
                request_seed=args.request_seed,
                max_tokens=args.max_tokens,
                system_prompt=batch_built["prompt"]["system_prompt"],
                user_prompt=batch_built["prompt"]["user_prompt"],
            )
            item["interface_v3"]["call"] = _run_llm_call(
                base_url=args.base_url,
                model=args.model,
                temperature=args.temperature,
                request_seed=42,
                max_tokens=args.max_tokens,
                system_prompt=iface_built["prompt"]["system_prompt"],
                user_prompt=iface_built["prompt"]["user_prompt"],
            )

        report["windows"][str(w)] = item

    out_json = out_dir / f"probe_round1_agent{args.agent_number}_seed{args.seed_distribution}_{ts}.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {out_json}")
    for w in args.memory_windows:
        p = report["windows"][str(w)]["prompt_equal"]
        print(f"W={w} prompt_equal system={p['system']} user={p['user']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
