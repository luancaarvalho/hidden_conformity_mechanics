# TODO - vLLM A100 (2026-02-26)

## Checklist

- [x] Criado orquestrador de benchmark vLLM A100:
  - `streamlit_test/projecao_simulacao/run_singleload_benchmarks_vllm_a100.sh`
- [x] Garantido 1 modelo por vez na A100 (kill + redeploy por modelo).
- [x] Corrigido binding de GPU (A100) e evitado uso da T400.
- [x] Ajustado `run_batch_png.py` para permitir `max_rounds <= 2*agents`.
- [x] Rodado benchmark com `rounds=6` e `max_rounds=6` (pipeline validado; run completo ainda pendente).
- [x] `Qwen/Qwen3-32B` baixado no cache da A100 (`~/.cache/huggingface/hub/models--Qwen--Qwen3-32B`).
- [x] Daemon remoto de download em paralelo ativo na A100 (continua com PC local desligado).
- [ ] Finalizar benchmark A100 para todos os labels no mesmo run:
  - `gemma4b`, `qwen4b_no_think`, `qwen32b_no_think`, `llama8b`, `llama70b`, `gemma27b`
- [ ] Revisar estratégia de deploy "3 Gemma 4B reais" no vLLM:
  - hoje está em alias (`--served-model-name :4 :5 :6`) no mesmo processo
  - validar e migrar para deploy separado por servidor/processo (sem alias) antes do próximo benchmark
- [ ] Resolver acesso gated dos modelos Llama no Hugging Face (sem isso, `llama8b/70b` falham no deploy).

## Como retomar benchmark (rápido)

```bash
cd /Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/projecao_simulacao
./run_singleload_benchmarks_vllm_a100.sh
```

## Observações rápidas

- Últimos runs em:
  - `streamlit_test/projecao_simulacao/benchmark_results/singleload_queue_vllm_a100_*`
- Resultado consolidado por run:
  - `consolidated_tokens_per_second.csv`

## Downloads remotos (A100)

- Script daemon:
  - `~/download_models_daemon.sh`
- PID atual:
  - `~/download_logs/download_models_daemon.pid` (`19969` no momento da criação)
- Logs:
  - `~/download_logs/qwen3_32b.log`
  - `~/download_logs/llama3_1_8b.log`
  - `~/download_logs/llama3_1_70b.log`
  - `~/download_logs/download_models_daemon.nohup.log`
- Marcadores de concluído:
  - `~/download_logs/*.done`
- Monitorar:
  - `ssh ncdia@172.18.254.16 "pgrep -af download_models_daemon.sh; tail -n 50 ~/download_logs/qwen3_32b.log; tail -n 50 ~/download_logs/llama3_1_8b.log; tail -n 50 ~/download_logs/llama3_1_70b.log"`
