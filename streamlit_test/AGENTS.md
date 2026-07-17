# AGENTS.md - Phase 3 Memory Conformity

- Phase 3 re-queries the LLM every round; do not describe it as a frozen cellular-automaton rule.
- Preserve synchronous round barriers: all prompts in round `r` use only finalized state from `r-1` and its permitted memory window.
- Record `N`, neighborhood size, `W`, max rounds, stop reason, seed distribution, token pair, prompt strategy, model, endpoint, and sampling.
- On RTX 5090, parallel scientific runs require batch-invariant vLLM and two exact replays.
- Existing July 2026 runs remain in the legacy `temp_vllm/gradio_project/results` directory and are linked into `artifacts/phase3_memory/RTX5090_liaan`.
