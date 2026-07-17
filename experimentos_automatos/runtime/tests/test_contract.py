import hashlib
import unittest
from pathlib import Path


RUNTIME = Path(__file__).resolve().parents[1]


class PhaseTwoContractTest(unittest.TestCase):
    def test_canonical_consumer_is_preserved(self):
        digest = hashlib.sha256((RUNTIME / "upstream/run_automaton_numba.py").read_bytes()).hexdigest()
        self.assertEqual(digest, "18b519a9f749ab1e5e27a91c67f3ab3ce06a11b6fd0263f87975f9de2079d746")

    def test_runtime_has_external_artifact_roots(self):
        source = (RUNTIME / "run_automaton_numba.py").read_text(encoding="utf-8")
        self.assertIn("AUTOMATON_RULES_ROOT", source)
        self.assertIn("AUTOMATON_OUTPUT_ROOT", source)


if __name__ == "__main__":
    unittest.main()
