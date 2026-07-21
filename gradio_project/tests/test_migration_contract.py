from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = REPO_ROOT / "gradio_project"


class MigrationContractTest(unittest.TestCase):
    def test_canonical_files_exist(self) -> None:
        required = (
            "interface/interface_v4_gradio.py",
            "memory/run_gradio_gemma4b_01_matrix.py",
            "memory/run_memory_online.py",
            "prompts/prompt_strategies.py",
            "prompts/prompt_templates.yaml",
            "utils/conformity_game_prompts.py",
            "launchers/launch_gradio_service.sh",
        )
        for relative in required:
            self.assertTrue((PHASE_ROOT / relative).is_file(), relative)

    def test_no_operational_streamlit_imports(self) -> None:
        for path in PHASE_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotIn("streamlit_test", node.module, str(path))
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn("streamlit_test", alias.name, str(path))

    def test_unchanged_prompt_sources_match_legacy_hashes(self) -> None:
        provenance = json.loads(
            (PHASE_ROOT / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8")
        )
        expected = provenance["source_files_sha256_before_path_migration"]
        for relative in (
            "prompts/prompt_strategies.py",
            "prompts/prompt_templates.yaml",
            "utils/conformity_game_prompts.py",
            "utils/initial_distribution.py",
            "utils/utils.py",
        ):
            digest = hashlib.sha256((PHASE_ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(expected[relative], digest, relative)

    def test_legacy_phase_directory_is_absent(self) -> None:
        self.assertFalse((REPO_ROOT / "streamlit_test").exists())

    def test_obsolete_memory_entrypoints_are_absent(self) -> None:
        obsolete = (
            "infra/rtx5090/run_memory_w1_pipeline.sh",
            "infra/rtx5090/run_token_pair_batch.sh",
            "infra/rtx5090/run_w0_rule_ca_parity_pipeline.sh",
            "infra/rtx5090/analyze_n7_ici_token_bias.py",
        )
        for relative in obsolete:
            self.assertFalse((REPO_ROOT / relative).exists(), relative)



if __name__ == "__main__":
    unittest.main()
