# Research Architecture

## Scientific Phases

### Phase 1 - Rule Extraction

The LLM is queried once for every binary local configuration. For neighborhood size `n`, a complete rule contains exactly `2^n` rows: 128 for `n=7`, 512 for `n=9`, and 2048 for `n=11`. The canonical matrix combines two prompt styles, nine visible token pairs, and the three neighborhood sizes.

Primary output: a frozen truth table plus prompt, response, parser, sampling, model, and backend provenance.

### Phase 2 - Cellular Automata Validation

The LLM is no longer queried. The simulator consumes a validated Phase 1 truth table and tests whether the induced local rule solves the Density Classification Task. The presentation contract uses ring sizes 100, 200, 400, 800, and 1600; initial majorities 51%, 55%, 60%, 65%, and 70%; balanced tests for both majority labels; and a maximum of `2N` synchronous updates.

Primary output: per-simulation trajectories and aggregated success/failure metrics tied to the exact Phase 1 rule hash.

### Phase 3 - Memory-Enabled Conformity

Each agent queries the LLM at every synchronous round using its current local neighborhood and up to `W` historical local snapshots. This is a dynamic LLM system, not a static `2^n` truth table. The canonical code lives in `gradio_project/`; existing RTX 5090 W=0..5 artifacts remain in their immutable result store and are exposed through an ignored artifact link.

## Handoff Contract

`Phase 1 -> Phase 2` requires:

- a unique `rule_id` derived from the rule CSV SHA-256;
- exactly `2^n` unique binary configurations;
- one parsed output per configuration;
- zero request, parse, or strict-contract failures;
- model, endpoint, prompt, sampling, and source-code hashes;
- deterministic replay evidence when inference was parallel.

Phase 3 does not consume Phase 1 rules. Comparisons across phases must join on model family, prompt family, visible token pair, neighborhood size, and sampling contract without treating the mechanisms as identical.

## Repository and Artifact Layout

```text
hidden_conformity_mechanics/
  extract_rules/                 # Phase 1 code and manifests
  experimentos_automatos/        # Phase 2 code
  gradio_project/                # Phase 3 code and Gradio interface
  docs/                          # scientific and operational contracts
  infra/rtx5090/                 # deployment and preflight scripts
  artifacts/                     # ignored run data, never committed
```

Every run directory must contain `run_manifest.json`, `source_hashes.json`, a machine-readable scoreboard, and a concise proofread summary. Existing run directories are immutable.

## Git Policy

- `main` contains reviewed, reproducible code and contracts.
- Work happens on `codex/<topic>` branches.
- Commits must not contain model weights, Conda environments, raw bulk outputs, tokens, passwords, or private keys.
- A run records the exact Git commit and refuses to start from a dirty worktree unless explicitly marked as a non-scientific smoke.
