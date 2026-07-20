# AGENTS.md - Phase 2 Cellular Automata

- Consume only Phase 1 rules whose manifest passed completeness and mechanical gates.
- Record the input rule SHA-256 in every Phase 2 run manifest and summary.
- The LLM must not be called during Phase 2.
- Use synchronous ring updates and cap each simulation at `2N` updates.
- Evaluate both initial majority labels for every reported condition.
- Separate convergence to the correct label, convergence to the wrong label, and no consensus.
- Keep bulk results under `artifacts/phase2_cellular_automata/RTX5090_liaan/`; never commit them.
- New runs use `artifacts/phase2_cellular_automata/rtx5090/n=<size>/...`; the uppercase legacy root remains read-only historical data.
