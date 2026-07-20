#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import requests

from gradio_project.prompts.prompt_strategies import get_prompt_strategy
from utils.utils import extract_output_text_from_responses, parse_opinion_token


TOKEN_PAIR_SPECS = {
    "01": {
        "tokens": ("0", "1"),
        "folder": "tokens=0-1",
        "variants": (
            "v9_lista_completa_meio_parity_01",
            "v21_zero_shot_cot_parity_01",
        ),
        "bases": (
            "v9_lista_completa_meio_01",
            "v21_zero_shot_cot_01",
        ),
    },
    "kz": {
        "tokens": ("k", "z"),
        "folder": "tokens=k-z",
        "variants": (
            "v9_lista_completa_meio_parity_kz",
            "v21_zero_shot_cot_parity_kz",
        ),
        "bases": (
            "v9_lista_completa_meio_kz",
            "v21_zero_shot_cot",
        ),
    },
    "triangle_circle": {
        "tokens": ("△", "○"),
        "folder": "tokens=triangle-circle",
        "variants": (
            "v9_lista_completa_meio_parity_triangle_circle",
            "v21_zero_shot_cot_parity_triangle_circle",
        ),
        "bases": (
            "v9_lista_completa_meio_△○",
            "v21_zero_shot_cot_△○",
        ),
    },
}
VARIANT_BASES = {
    variant: base
    for spec in TOKEN_PAIR_SPECS.values()
    for variant, base in zip(spec["variants"], spec["bases"])
}
VARIANT_TOKEN_PAIRS = {
    variant: pair_key
    for pair_key, spec in TOKEN_PAIR_SPECS.items()
    for variant in spec["variants"]
}
MAX_OUTPUT_TOKENS = {
    variant: (8 if variant.startswith("v9_") else 768)
    for variant in VARIANT_BASES
}
TOKENS = TOKEN_PAIR_SPECS["01"]["tokens"]
_V21_MEMORY_PREFIX = "Use the MEMORY section above as prior-round context. "


def token_pair_spec(token_pair: str) -> dict[str, Any]:
    try:
        return TOKEN_PAIR_SPECS[token_pair]
    except KeyError as exc:
        raise ValueError(f"unsupported token pair: {token_pair}") from exc


def variants_for_pair(token_pair: str) -> tuple[str, str]:
    return token_pair_spec(token_pair)["variants"]


def resolve_variants(token_pair: str, variants: Sequence[str] | None) -> tuple[str, ...]:
    expected = variants_for_pair(token_pair)
    selected = tuple(variants) if variants else expected
    invalid = [variant for variant in selected if variant not in expected]
    if invalid:
        raise ValueError(f"variants {invalid} do not belong to token pair {token_pair}")
    return selected


def token_pair_for_variant(variant: str) -> str:
    try:
        return VARIANT_TOKEN_PAIRS[variant]
    except KeyError as exc:
        raise ValueError(f"unsupported parity variant: {variant}") from exc


def tokens_for_variant(variant: str) -> tuple[str, str]:
    return token_pair_spec(token_pair_for_variant(variant))["tokens"]


def visible_tokens(binary_values: Sequence[int | str], variant: str) -> list[str]:
    tokens = tokens_for_variant(variant)
    visible: list[str] = []
    for value in binary_values:
        text = str(value)
        if text in tokens:
            visible.append(text)
        elif text in {"0", "1"}:
            visible.append(tokens[int(text)])
        else:
            raise ValueError(f"value {value!r} is neither binary nor in {tokens}")
    return visible


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render_w0_prompt(
    variant: str,
    neighborhood: Sequence[int | str],
) -> tuple[str, str]:
    return render_memory_prompt(variant, neighborhood, memory_snapshots=())


def render_memory_prompt(
    variant: str,
    neighborhood: Sequence[int | str],
    *,
    memory_snapshots: Sequence[tuple[int, Sequence[int | str]]],
) -> tuple[str, str]:
    if variant not in VARIANT_BASES:
        raise ValueError(f"unsupported parity variant: {variant}")
    if len(neighborhood) < 3 or len(neighborhood) % 2 == 0:
        raise ValueError("neighborhood must have an odd length >= 3")

    opinions = visible_tokens(neighborhood, variant)
    center = len(opinions) // 2
    strategy = get_prompt_strategy(VARIANT_BASES[variant])
    system_prompt, user_prompt = strategy.build_prompt(
        left=opinions[:center],
        right=opinions[center + 1 :],
        current_opinion=opinions[center],
    )

    if variant.startswith("v21_") and not memory_snapshots:
        if user_prompt.count(_V21_MEMORY_PREFIX) != 1:
            raise RuntimeError("historical v21 MEMORY prefix changed unexpectedly")
        user_prompt = user_prompt.replace(_V21_MEMORY_PREFIX, "", 1)

    if memory_snapshots:
        memory_lines = ["=== MEMORY (Previous Rounds) ==="]
        for round_idx, snapshot in memory_snapshots:
            snapshot_tokens = visible_tokens(snapshot, variant)
            if len(snapshot_tokens) != len(opinions):
                raise ValueError("memory snapshot length must match the current neighborhood")
            memory_lines.extend(
                [
                    f"Round {round_idx}:",
                    f"  You: [{snapshot_tokens[center]}]",
                    f"  Left: {snapshot_tokens[:center]!r}",
                    f"  Right: {snapshot_tokens[center + 1:]!r}",
                ]
            )
        user_prompt = "\n".join(memory_lines) + "\n\n" + user_prompt
    elif "MEMORY" in user_prompt or "MEMORY" in system_prompt:
        raise RuntimeError("W=0 parity prompt must not mention MEMORY")
    return system_prompt, user_prompt


def build_responses_payload(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    variant: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "seed": 42,
        "max_output_tokens": MAX_OUTPUT_TOKENS[variant],
        "stream": False,
    }


def prompt_hash(system_prompt: str, user_prompt: str) -> str:
    return sha256_text(stable_json({"system": system_prompt, "user": user_prompt}))


def payload_hash(payload: dict[str, Any]) -> str:
    return sha256_text(stable_json(payload))


def parse_final_choice(raw_response: str, variant: str) -> str | None:
    return parse_opinion_token(
        raw_response,
        allowed_tokens=tokens_for_variant(variant),
        prefer_last=True,
    )


def responses_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return f"{root}/v1/responses"


def query_responses_with_retries(
    *,
    base_url: str,
    payload: dict[str, Any],
    variant: str,
    timeout_s: int,
    max_attempts: int,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    final_response = ""
    final_choice: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                responses_url(base_url),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout_s,
            )
            response.raise_for_status()
            data = response.json()
            raw_response = extract_output_text_from_responses(data)
            choice = parse_final_choice(raw_response, variant)
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": int(response.status_code),
                    "raw_response": raw_response,
                    "choice": choice,
                    "error": None,
                }
            )
            if choice is not None:
                final_response = raw_response
                final_choice = choice
                break
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": None,
                    "raw_response": "",
                    "choice": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            time.sleep(0.5)
    return {
        "raw_response": final_response,
        "raw_response_sha256": sha256_text(final_response),
        "choice_token": final_choice,
        "choice": (
            tokens_for_variant(variant).index(final_choice)
            if final_choice is not None
            else None
        ),
        "attempt_count": len(attempts),
        "request_failures": sum(item["http_status"] is None for item in attempts),
        "parse_failures": sum(
            item["http_status"] is not None and item["choice"] is None for item in attempts
        ),
        "attempts": attempts,
    }


def source_paths(repo_root: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        repo_root / "gradio_project/prompts/prompt_strategies.py",
        repo_root / "gradio_project/prompts/prompt_templates.yaml",
        repo_root / "utils/utils.py",
        repo_root / "utils/initial_distribution.py",
        repo_root / "utils/parity_render.py",
    ]
