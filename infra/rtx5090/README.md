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

Launch the complete Gemma 3 4B n=7 W=0..5, seeds 1..50 campaign with:

```bash
bash infra/rtx5090/launch_n7_memory_campaign_tmux.sh
```

The launcher creates one persistent tmux session and runs one resumable Python
orchestrator. Windows execute sequentially from W=0 through W=5. Within a replay,
two cells can be active while one shared executor caps inference at 32 requests.
New artifacts use `artifacts/<phase>/rtx5090/n=<size>/...`; exact replays are
reduced to one `canonical/` copy plus `determinism/` evidence.

After the remote run reaches a terminal state, mirror and verify it from the local workspace with:

```bash
bash infra/rtx5090/sync_n7_memory_campaign_to_local.sh <CAMPAIGN_ID>
```

Control files are written to:

```text
artifacts/orchestration/rtx5090/<CAMPAIGN_ID>/
```

The final aggregate is written to:

```text
artifacts/cross_window_validation/rtx5090/n=7/gemma-3-4b-it/W0-5_seed1-50_<timestamp>/
```
