from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from gradio_project.memory.run_memory_online import CellSpec, read_manifest
from infra.rtx5090.run_n7_memory_campaign import Campaign
from utils.w0_parity_contract import token_pair_spec


class N7MemoryCampaignTest(unittest.TestCase):
    def test_window_matrix_matches_campaign_contract(self) -> None:
        campaign = Campaign("gemma4b_n7_w0_5_seed50_20990101T000000Z")
        counts = {window: len(campaign.specs_for_window(window)) for window in range(6)}
        self.assertEqual(counts, {0: 180, 1: 180, 2: 300, 3: 300, 4: 300, 5: 300})
        self.assertEqual(sum(counts.values()), 1560)
        for window in range(6):
            specs = campaign.specs_for_window(window)
            expected_seeds = set(range(21, 51)) if window in (0, 1) else set(range(1, 51))
            self.assertEqual({spec.seed for spec in specs}, expected_seeds)
            self.assertEqual({spec.memory_window for spec in specs}, {window})
            self.assertEqual({spec.token_pair for spec in specs}, {"01", "kz", "triangle_circle"})

    def test_every_pair_has_only_its_two_historical_variants(self) -> None:
        campaign = Campaign("gemma4b_n7_w0_5_seed50_20990101T000000Z")
        specs = campaign.specs_for_window(2)
        for pair in ("01", "kz", "triangle_circle"):
            self.assertEqual(
                {spec.variant for spec in specs if spec.token_pair == pair},
                set(token_pair_spec(pair)["variants"]),
            )

    def test_manifest_rejects_duplicate_cells(self) -> None:
        row = (
            '{"cell_id":"same","run_id":"run","token_pair":"01",'
            '"variant":"v9_lista_completa_meio_parity_01",'
            '"memory_window":2,"seed":1}\n'
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.jsonl"
            path.write_text(row + row, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_manifest(path)

    def test_campaign_manifest_is_stable_across_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            campaign = Campaign("gemma4b_n7_w0_5_seed50_20990101T000000Z")
            campaign.root = Path(temp) / "control"
            campaign.manifest_dir = campaign.root / "manifests"
            campaign.create_manifests()
            first = json.loads(
                (campaign.root / "campaign_manifest.json").read_text(encoding="utf-8")
            )
            campaign.create_manifests()
            second = json.loads(
                (campaign.root / "campaign_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first, second)
            self.assertEqual(
                sum(
                    len(path.read_text(encoding="utf-8").splitlines())
                    for path in campaign.manifest_dir.glob("W*.jsonl")
                ),
                1560,
            )

    def test_cell_spec_rejects_cross_pair_variant(self) -> None:
        with self.assertRaises(ValueError):
            CellSpec.from_dict(
                {
                    "cell_id": "bad",
                    "run_id": "run",
                    "token_pair": "kz",
                    "variant": "v21_zero_shot_cot_parity_01",
                    "memory_window": 2,
                    "seed": 1,
                }
            )


if __name__ == "__main__":
    unittest.main()
