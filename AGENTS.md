# AGENTS.md - Hidden Conformity Mechanics

## Canonical phases

1. `extract_rules/`: exhaustive LLM rule extraction.
2. `experimentos_automatos/`: Density Classification Task validation using frozen rules.
3. `gradio_project/`: memory-enabled conformity simulations with live LLM calls.

Read `docs/RESEARCH_ARCHITECTURE.md` and the phase-specific `AGENTS.md` before changing or running a phase.

## RTX 5090

- SSH: `ssh rtx5090` while the AnyDesk VPN is active.
- Canonical checkout: `/home/liaan/Documentos/Luan/hidden_conformity_mechanics`.
- Conda prefix: `artifacts/conda/runtime`; install with UV through `infra/rtx5090/create_runtime_env.sh`.
- Gradio code: `gradio_project/`.
- Phase 3 artifacts: `artifacts/phase3_memory/RTX5090_liaan`.
- vLLM endpoint: `http://127.0.0.1:8127/v1` when the managed Gemma service is active.
- Gradio service: tmux session `gradio_project_v4`, port `7860`.

Do not use `streamlit_test/` as an operational path. Historical Streamlit-era instructions are archived in `docs/legacy/AGENTS_PROJECT_FINAL_LEGACY.md` and do not override this file.

## Required conduct

- Do not kill, pause, renice, or modify processes not started by the current experiment.
- Check `nvidia-smi`, `tmux ls`, active ports, Git status, and endpoint model before a GPU run.
- Do not overwrite an existing run directory. Every run must have source, prompt, payload, model, backend, sampling, hardware, and Git provenance.
- Keep bulk outputs, weights, databases, Conda environments, logs, PNGs, and `.npy` files under ignored `artifacts/` paths.
- Use Conda plus `uv pip --python <prefix>/bin/python`; do not create a virtualenv.
- For deterministic parallel vLLM work, require `VLLM_BATCH_INVARIANT=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `temperature=0`, server/request seed `42`, and two exact replays.
- Never call a Phase 3 dynamic-memory run a static rule table. Phase 1 rules may enter Phase 2 only after the complete `2^n` mechanical gate passes.
- Work on `codex/<topic>` branches and run experiments only from a clean, recorded commit.

## Verification

Before Phase 3 work:

```bash
cd /home/liaan/Documentos/Luan/hidden_conformity_mechanics
bash infra/rtx5090/preflight.sh
bash gradio_project/tests/run_tests.sh
```
