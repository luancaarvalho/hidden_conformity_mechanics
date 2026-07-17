import csv
import hashlib
import json
import unittest
from pathlib import Path


RUNTIME = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class PhaseOneContractTest(unittest.TestCase):
    def test_upstream_and_frozen_prompt_hashes(self):
        provenance = json.loads((RUNTIME / "SOURCE_PROVENANCE.json").read_text())
        self.assertEqual(sha256(RUNTIME / "upstream/execucao_simultanea.py"), provenance["producer_upstream"]["sha256"])
        self.assertEqual(sha256(RUNTIME / "prompt_strategies.py"), provenance["prompt_strategies"]["sha256"])
        self.assertEqual(sha256(RUNTIME / "prompt_templates.yaml"), provenance["prompt_templates"]["sha256"])

    def test_canonical_manifest_matrix(self):
        path = RUNTIME / "manifests/gemma4b_canonical_54_cells.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 54)
        self.assertEqual({int(row["n_vizinhos"]) for row in rows}, {7, 9, 11})
        self.assertEqual(len({row["tokens_config"] for row in rows}), 9)
        self.assertEqual(len({row["id_experimento"] for row in rows}), 54)
        for n in (7, 9, 11):
            self.assertEqual(sum(int(row["n_vizinhos"]) == n for row in rows), 18)


if __name__ == "__main__":
    unittest.main()
