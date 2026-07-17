# RTX 5090 Deployment

Canonical checkout: `/home/liaan/Documentos/Luan/hidden_conformity_mechanics`.

The July 2026 memory runtime remains at `/home/liaan/Documentos/Luan/temp_vllm/gradio_project`; it is not moved while jobs or analyses may reference it. `setup_artifact_layout.sh` exposes those results under the canonical repository through an ignored symbolic link.

Use the existing Conda environments. Install missing packages with:

```bash
uv pip install --python /home/liaan/Documentos/Luan/temp_vllm/conda_envs/gradio/bin/python \
  -r infra/rtx5090/requirements-runtime.txt
```

Do not create a virtualenv. Run `preflight.sh` before every GPU batch.
