from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gradio_project.prompts.prompt_strategies import get_prompt_strategy
from utils.parity_artifacts import finalize_replays, write_json
from utils.w0_parity_contract import (
    build_responses_payload,
    parse_final_choice,
    render_memory_prompt,
    render_w0_prompt,
    resolve_variants,
    token_pair_spec,
    variants_for_pair,
    visible_tokens,
)


class W0ParityContractTest(unittest.TestCase):
    def test_v9_is_textually_equal_to_historical_prompt(self) -> None:
        neighborhood = [0, 1, 0, 1, 0, 1, 0]
        expected = get_prompt_strategy("v9_lista_completa_meio_01").build_prompt(
            left=["0", "1", "0"], right=["0", "1", "0"], current_opinion="1"
        )
        self.assertEqual(
            render_w0_prompt("v9_lista_completa_meio_parity_01", neighborhood),
            expected,
        )

    def test_all_pairs_match_their_frozen_historical_templates(self) -> None:
        neighborhood = [0, 1, 0, 1, 0, 1, 0]
        cases = {
            "01": ("v9_lista_completa_meio_01", "v21_zero_shot_cot_01"),
            "kz": ("v9_lista_completa_meio_kz", "v21_zero_shot_cot"),
            "triangle_circle": (
                "v9_lista_completa_meio_△○",
                "v21_zero_shot_cot_△○",
            ),
        }
        memory_prefix = "Use the MEMORY section above as prior-round context. "
        for token_pair, bases in cases.items():
            tokens = token_pair_spec(token_pair)["tokens"]
            left = [tokens[0], tokens[1], tokens[0]]
            right = [tokens[0], tokens[1], tokens[0]]
            for variant, base in zip(variants_for_pair(token_pair), bases):
                expected_system, expected_user = get_prompt_strategy(base).build_prompt(
                    left=left,
                    right=right,
                    current_opinion=tokens[1],
                )
                if variant.startswith("v21_"):
                    expected_user = expected_user.replace(memory_prefix, "", 1)
                self.assertEqual(
                    render_w0_prompt(variant, neighborhood),
                    (expected_system, expected_user),
                    msg=f"historical prompt changed for {variant}",
                )
    def test_v21_only_removes_dangling_memory_reference(self) -> None:
        neighborhood = [0, 1, 0, 1, 0, 1, 0]
        historical_system, historical_user = get_prompt_strategy(
            "v21_zero_shot_cot_01"
        ).build_prompt(
            left=["0", "1", "0"], right=["0", "1", "0"], current_opinion="1"
        )
        system, user = render_w0_prompt("v21_zero_shot_cot_parity_01", neighborhood)
        self.assertEqual(system, historical_system)
        self.assertEqual(
            user,
            historical_user.replace(
                "Use the MEMORY section above as prior-round context. ", "", 1
            ),
        )
        self.assertNotIn("MEMORY", system + user)

    def test_w1_uses_historical_timeline_without_changing_current_input(self) -> None:
        neighborhood = [0, 1, 0, 1, 0, 1, 0]
        system, user = render_memory_prompt(
            "v21_zero_shot_cot_parity_01",
            neighborhood,
            memory_snapshots=[(0, neighborhood)],
        )
        self.assertIn("=== MEMORY (Previous Rounds) ===", user)
        self.assertIn("Round 0:\n  You: [1]", user)
        self.assertIn("Left: ['0', '1', '0']", user)
        self.assertIn("Right: ['0', '1', '0']", user)
        self.assertIn("Use the MEMORY section above as prior-round context.", user)
        self.assertIn("Complete Opinion List: ['0', '1', '0', '1', '0', '1', '0']", user)
        self.assertNotIn("MEMORY", system)

    def test_w1_maps_binary_memory_to_each_visible_token_pair(self) -> None:
        neighborhood = [0, 1, 0, 1, 0, 1, 0]
        for token_pair in ("kz", "triangle_circle"):
            token0, token1 = token_pair_spec(token_pair)["tokens"]
            for variant in variants_for_pair(token_pair):
                _, user = render_memory_prompt(
                    variant,
                    neighborhood,
                    memory_snapshots=[(0, neighborhood)],
                )
                self.assertIn(f"Round 0:\n  You: [{token1}]", user)
                self.assertIn(
                    f"Left: {[token0, token1, token0]!r}",
                    user,
                )
                self.assertIn(
                    repr([token0, token1, token0, token1, token0, token1, token0]),
                    user,
                )

    def test_w0_wrapper_matches_generic_renderer_without_memory(self) -> None:
        neighborhood = [1, 0, 1, 0, 1, 0, 1]
        for variant in (
            "v9_lista_completa_meio_parity_01",
            "v21_zero_shot_cot_parity_01",
        ):
            self.assertEqual(
                render_w0_prompt(variant, neighborhood),
                render_memory_prompt(variant, neighborhood, memory_snapshots=[]),
            )

    def test_sampling_payload_is_minimal_and_parser_prefers_last(self) -> None:
        payload = build_responses_payload(
            model="gemma3-4b-temp0",
            system_prompt="system",
            user_prompt="user",
            variant="v21_zero_shot_cot_parity_01",
        )
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["max_output_tokens"], 768)
        for key in ("top_k", "top_p", "min_p", "repeat_penalty"):
            self.assertNotIn(key, payload)
        self.assertEqual(
            parse_final_choice(
                "First [0].\nFinal answer:\n[1]",
                "v21_zero_shot_cot_parity_01",
            ),
            "1",
        )

    def test_parser_maps_only_the_active_pair(self) -> None:
        cases = {
            "01": ("[0]", "0", "[k]"),
            "kz": ("[z]", "z", "[0]"),
            "triangle_circle": ("[△]", "△", "[z]"),
        }
        for token_pair, (valid, expected, foreign) in cases.items():
            variant = variants_for_pair(token_pair)[1]
            self.assertEqual(parse_final_choice(valid, variant), expected)
            self.assertIsNone(parse_final_choice(foreign, variant))

    def test_binary_visible_mapping_is_bidirectional_and_pair_scoped(self) -> None:
        binary = [0, 1, 1, 0]
        self.assertEqual(
            visible_tokens(binary, "v9_lista_completa_meio_parity_kz"),
            ["k", "z", "z", "k"],
        )
        self.assertEqual(
            visible_tokens(binary, "v21_zero_shot_cot_parity_triangle_circle"),
            ["△", "○", "○", "△"],
        )
        with self.assertRaises(ValueError):
            resolve_variants("kz", ["v21_zero_shot_cot_parity_01"])

    def test_retention_keeps_one_copy_on_pass_and_two_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for deterministic in (True, False):
                work = root / f"work_{deterministic}"
                replay1 = work / "replay_01"
                replay2 = work / "replay_02"
                proof = work / "proof"
                for path in (replay1, replay2, proof):
                    path.mkdir(parents=True)
                    write_json(path / "value.json", {"value": 1})
                final = root / f"final_{deterministic}"
                finalize_replays(
                    replay1=replay1,
                    replay2=replay2,
                    proof_dir=proof,
                    final_root=final,
                    deterministic=deterministic,
                    status_details={"test": True},
                )
                if deterministic:
                    self.assertTrue((final / "canonical/value.json").is_file())
                    self.assertFalse((final / "replay_01").exists())
                    self.assertFalse((final / "replay_02").exists())
                else:
                    self.assertTrue((final / "replay_01/value.json").is_file())
                    self.assertTrue((final / "replay_02/value.json").is_file())


if __name__ == "__main__":
    unittest.main()
