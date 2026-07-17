# Phase 2 Cellular Automata Runtime

`upstream/run_automaton_numba.py` is the canonical consumer with SHA-256 `18b519a9f749ab1e5e27a91c67f3ab3ce06a11b6fd0263f87975f9de2079d746`. The adjacent runtime changes only import and artifact-root paths so the code can execute from this repository layout.

Example after a Phase 1 table passes its gate:

```bash
AUTOMATON_RULES_ROOT="$PWD/artifacts/phase2_cellular_automata/RTX5090_liaan/rules" \
AUTOMATON_OUTPUT_ROOT="$PWD/artifacts/phase2_cellular_automata/RTX5090_liaan/runs" \
python experimentos_automatos/runtime/run_automaton_numba.py \
  --csv <validated-phase1-basic.csv> \
  --n_simulations 200 \
  --agents 100 \
  --max_iterations 200 \
  --initial-ratio 0.51
```

The Phase 2 manifest must record the SHA-256 of `<validated-phase1-basic.csv>`.
