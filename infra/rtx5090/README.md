# RTX 5090 Deployment

Canonical checkout: `/home/liaan/Documentos/Luan/hidden_conformity_mechanics`.

Phase 3 code lives in the canonical checkout under `gradio_project/`. The July 2026 memory artifacts remain immutable at `/home/liaan/Documentos/Luan/temp_vllm/gradio_project/results`; `setup_artifact_layout.sh` exposes them through an ignored symbolic link.

Create a dedicated Conda prefix and install packages with UV:

```bash
bash infra/rtx5090/create_runtime_env.sh
```

Do not create a virtualenv and do not modify an environment used by an active experiment. Run `preflight.sh` before every GPU batch.

Start the canonical interface in tmux with:

```bash
tmux new-session -d -s gradio_project_v4 \
  'cd /home/liaan/Documentos/Luan/hidden_conformity_mechanics && bash gradio_project/launchers/launch_gradio_service.sh'
```
