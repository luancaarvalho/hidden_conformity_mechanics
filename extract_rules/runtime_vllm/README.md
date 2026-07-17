# Phase 1 vLLM Runtime

`upstream/execucao_simultanea.py` is the newest unmodified producer located during the July 2026 audit. `execucao_simultanea_vllm.py` differs only where needed for the current RTX 5090 vLLM endpoint: external served-model aliases and configurable output length.

The frozen prompt files were copied from the validated Gemma 3 4B runtime on `liaan-006`. The canonical manifest contains 54 cells: two prompt strategies, nine token pairs, and `n=7,9,11`.

Run only after `infra/rtx5090/preflight.sh` passes and the current memory batch has released the GPU:

```bash
tmux new-session -d -s phase1_gemma4b_rules \
  "bash extract_rules/runtime_vllm/launch_rtx5090.sh"
```

The launcher uses the existing Conda environment, endpoint `8127`, model alias `gemma3-4b-temp0`, temperature zero, request seed 42, and no top-k/top-p/min-p/repetition overrides. Set `LLM_MAX_OUTPUT_TOKENS=3000` for the canonical matrix because it includes CoT cells.
