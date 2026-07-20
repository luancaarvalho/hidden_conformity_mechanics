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

Launch the Gemma 3 4B W=0 rule/automaton parity campaign with:

```bash
bash infra/rtx5090/launch_w0_rule_ca_parity_tmux.sh
```

The launcher prints the tmux session, run ID, and cross-phase result root. New artifacts use `artifacts/<phase>/rtx5090/n=<size>/...`; exact replays are reduced to one `canonical/` copy plus `determinism/` evidence.

After the remote run reaches a terminal state, mirror and verify it from the local workspace with:

```bash
bash infra/rtx5090/sync_w0_rule_ca_parity_to_local.sh <RUN_ID>
```

Run the complete sequential `k/z` and `△/○` campaign, including rule extraction,
automata, online W=0 parity, W=1 impact, and the cross-token proofread, with:

```bash
bash infra/rtx5090/launch_token_pair_batch_tmux.sh
```

The launcher prints a `BATCH_ID`. After terminal completion, mirror every promoted
artifact and verify remote/local SHA-256 hashes from the local workspace with:

```bash
bash infra/rtx5090/sync_token_pair_batch_to_local.sh <BATCH_ID>
```
