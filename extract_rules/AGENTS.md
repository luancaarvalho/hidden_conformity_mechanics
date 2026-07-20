# AGENTS.md - Phase 1 Rule Extraction

- Read `../docs/RESEARCH_ARCHITECTURE.md` before changing or running extraction code.
- `runtime_vllm/upstream/` is immutable reference code. Apply RTX 5090 adaptations only to files outside `upstream/`.
- A complete cell has exactly `2^n` unique inputs and one strict parsed output for each input.
- The canonical matrix is 2 prompt styles x 9 visible token pairs x n={7,9,11}: 54 cells.
- Use the prompt files frozen beside the runtime and record their SHA-256 hashes.
- Do not infer or inject the correct majority label. The model must produce the rule output.
- Do not promote partial tables, parser failures, retries with changed prompts, or mixed source revisions.
- Keep run data under `artifacts/phase1_rule_extraction/RTX5090_liaan/`; never commit it.
- New runs use `artifacts/phase1_rule_extraction/rtx5090/n=<size>/...`; the uppercase legacy root remains read-only historical data.
