# Gradio Memory Runtime

Canonical Phase 3 implementation for the memory-enabled conformity experiments.

```text
gradio_project/
  interface/     Gradio v4 interface and simulation runner
  memory/        batch, replay, comparison, and finalization scripts
  prompts/       prompt strategies and templates
  utils/         shared simulation utilities
  launchers/     service and experiment launchers
  tests/         migration/package checks
```

On the RTX 5090:

```bash
cd /home/liaan/Documentos/Luan/hidden_conformity_mechanics
bash infra/rtx5090/create_runtime_env.sh
bash gradio_project/tests/run_tests.sh
tmux new-session -d -s gradio_project_v4 \
  'cd /home/liaan/Documentos/Luan/hidden_conformity_mechanics && bash gradio_project/launchers/launch_gradio_service.sh'
```

The UI listens on `127.0.0.1:7860` by default and uses vLLM at `127.0.0.1:8127`. Existing and new Phase 3 outputs are addressed through `artifacts/phase3_memory/RTX5090_liaan`.
