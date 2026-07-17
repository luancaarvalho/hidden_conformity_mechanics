# RTX 5090 Deployment

Canonical checkout: `/home/liaan/Documentos/Luan/hidden_conformity_mechanics`.

The July 2026 memory runtime remains at `/home/liaan/Documentos/Luan/temp_vllm/gradio_project`; it is not moved while jobs or analyses may reference it. `setup_artifact_layout.sh` exposes those results under the canonical repository through an ignored symbolic link.

Create a dedicated Conda prefix and install packages with UV:

```bash
/home/liaan/miniconda3/bin/conda create -y \
  -p "$PWD/artifacts/conda/runtime" python=3.12
uv pip install --python "$PWD/artifacts/conda/runtime/bin/python" \
  -r infra/rtx5090/requirements-runtime.txt
```

Do not create a virtualenv and do not modify an environment used by an active experiment. Run `preflight.sh` before every GPU batch.
