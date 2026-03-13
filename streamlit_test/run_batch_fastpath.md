# Run Batch Fast Path

This note documents the throughput-oriented changes applied to the default `run_batch_png.py` execution path via [llm_sim_runner.py](/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/llm_sim_runner.py).

## Goal

Reduce per-request overhead so batch simulation stays as close as possible to the request cadence observed in `execucao_simultanea.py`, while preserving the same simulation outputs and logging artifacts.

## Changes Applied

1. Neighbor indices are precomputed once per simulation.
   - Added `_neighbor_index_map(...)`.
   - Avoids recomputing left/right index lists for every agent request.

2. Memory prompt formatting now reuses precomputed neighbor indices.
   - Added `_format_memory_block_timeline_precomputed(...)`.
   - Removes repeated neighbor-index lookup from the hot path.

3. Prompts are prebuilt per round before issuing requests.
   - Added `_PromptJob`.
   - For each round, the runner now constructs all `(system_prompt, user_prompt)` pairs first, then issues requests.
   - This reduces Python work between one successful response and the next request.

4. Request logging is buffered.
   - `requests.jsonl` writes now accumulate in memory and flush every 64 records.
   - This removes synchronous file writes from the tightest request loop.

5. Removed an unnecessary copy of the previous round state.
   - The runner now reads `prev = states[r - 1]` instead of `prev = states[r - 1].copy()`.

## What Did Not Change

1. Simulation semantics.
   - Agent order is unchanged.
   - Consensus/stability checks are unchanged.
   - State updates remain sequential within each round.

2. Output artifacts.
   - `requests.jsonl`, `states.npy`, `result_meta.json`, `initial_config.json`, and heatmaps are still generated as before.

3. Prompt semantics.
   - `standard` and `conformity_game` prompt construction still follows the existing shared utilities.
   - `Qwen` still receives `/no_think` automatically.

## Observed Effect

Short validation on A100 with `Gemma 4B`, `v9`, `agents=10`, `rounds=2`, `memory=0`:

- `10` requests completed
- average request time around `0.13s`
- throughput around `7.66 req/s`

With `memory_w=3`, the same short validation stayed close:

- average request time around `0.14s`

This indicates the dominant slowdown was runner overhead, not token parsing itself.

## Operational Rule

This optimized path is now the default behavior of `run_batch_png.py` through `llm_sim_runner.py`.
There is no separate "fast mode" toggle.
