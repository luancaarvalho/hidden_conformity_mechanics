# Hidden Conformity Research Pipeline

This repository separates the study into the three phases defined in the research presentation:

1. `extract_rules/`: exhaustive LLM rule extraction.
2. `experimentos_automatos/`: validation of frozen rules in cellular automata.
3. `gradio_project/`: memory-enabled conformity simulations and the Gradio interface.

Code and experiment contracts are versioned in Git. Large run artifacts live under `artifacts/`, which is intentionally ignored, and every run must carry a manifest with Git revision, input hashes, model, backend, sampling, and hardware provenance.

Read `AGENTS.md` and `docs/RESEARCH_ARCHITECTURE.md` before launching work.
