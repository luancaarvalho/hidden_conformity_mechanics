from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "run_w0_parity_automata.py"
SPEC = importlib.util.spec_from_file_location("run_w0_parity_automata", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class W0ParityAutomataTest(unittest.TestCase):
    def test_center_identity_rule_preserves_state_for_two_transitions(self) -> None:
        rules = {
            format(index, "07b"): int(format(index, "07b")[3])
            for index in range(128)
        }
        initial = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
        states, completed, stop_reason = MODULE.evolve(initial, rules, 2)
        self.assertEqual(completed, 2)
        self.assertEqual(stop_reason, "max_rounds")
        np.testing.assert_array_equal(states[0], initial)
        np.testing.assert_array_equal(states[1], initial)
        np.testing.assert_array_equal(states[2], initial)

    def test_shared_renderer_is_byte_deterministic(self) -> None:
        states = np.array([[0, 1, 0], [1, 1, 0]], dtype=float)
        first = MODULE.render_binary_trajectory_png(states)
        second = MODULE.render_binary_trajectory_png(states)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
