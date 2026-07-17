# Phase 3 - Gradio Memory Experiments

This directory is the only canonical code root for memory-enabled conformity experiments.

## Boundaries

- `interface/`: interactive Gradio application and `SimulationRunner`.
- `memory/`: batch runners, replay comparators, and artifact finalizers.
- `prompts/`: prompt strategies and YAML templates.
- `utils/`: parsing, initial distributions, backend calls, and conformity prompts.
- `launchers/`: tmux-safe service and experiment entry points.
- `tests/`: migration and package contract checks.

The Phase 3 result root is `artifacts/phase3_memory/RTX5090_liaan`. It currently resolves to the preserved July 2026 result store. Never write results into the Git-tracked package.

## Runtime contract

- Launch the UI with `bash gradio_project/launchers/launch_gradio_service.sh`.
- Launch Python entry points from the repository root or use `python -m`.
- Do not add imports from `streamlit_test`, `projeto_final`, or the legacy `/home/liaan/Documentos/Luan/temp_vllm/gradio_project` code root.
- A launcher may refer to the legacy result directory only through the canonical artifact symlink.
- Preserve synchronous round semantics: prompts for round `r` are computed from the completed state at `r-1`.
- Preserve the scientific stop contract declared by each launcher. Do not silently add stability stopping or change maximum rounds.
- Prompt, parser, sampling, and source hashes belong in every run manifest.

## Determinism

Parallel outputs are deterministic evidence only after exact replay comparison covers classifications, rounds, trajectories, normalized logs, and rendered PNGs. Temperature zero by itself is insufficient.
