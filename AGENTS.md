# AGENTS.md - Projeto Final (Conformidade LLM)

## 0. Objetivo Atual (2026-02-26)

- Rodar benchmark `single-load` no vLLM da A100 (`172.18.254.16`) para os modelos alvo, com **1 modelo por vez**.
- Usar cenário `v21_zero_shot_cot` com `agents=30`, `neighbors=7`, `seed_distribution=1`, `memory_w=3`.
- Limite operacional atual para este ciclo: `rounds=6` e `max_rounds=6`.
- Consolidar métricas em:
  - `.../streamlit_test/projecao_simulacao/benchmark_results/singleload_queue_vllm_a100_*/consolidated_tokens_per_second.csv`

Este documento consolida o estado operacional do `projeto_final`:
- dois servidores principais usados no benchmark (`.17` RTX 6000 e `.18` Mac Studio),
- fluxo da `interface_v2`,
- fluxo de benchmark (single-load e paralelo),
- regras de determinismo e troubleshooting.

## 0.1 Atualização operacional (2026-03-04)

### A100 - vLLM Gemma 4B (em execução)
- Host: `172.18.254.16`
- Sessão: `tmux exp_gemma4b_vllm`
- Script: `execucao_simultanea_vllm.py --auto-orchestrate`
- CSV: `~/luan/experimentos/por_modelo/experimentos_gemma4b.csv`
- DB: `~/luan/experimentos/por_modelo/experimentos_gemma4b.db`
- Deploy ativo:
  - `google/gemma-3-4b-it` servido como `google/gemma-3-4b`
  - `tensor-parallel-size=1`, `dtype=bfloat16`, `gpu_memory_utilization=0.98`,
    `max_model_len=8192`, `max_num_seqs=64`, `max_num_batched_tokens=32768`,
    `enable_chunked_prefill`, `enable_prefix_caching`, `swap_space=8`,
    `generation-config=vllm`
- Estado do run no momento deste registro:
  - `concluido=3`, `executando=1`, `pendente=50`

### Automaton Gemma 27B (concluído)
- Base de entrada: `projeto_final/extract_rules/A100/gemma27b`
- Escopo: 2 estratégias (`v9_lista_completa_meio_kz`, `v21_zero_shot_cot`) em `n=7,9,11`
- Execução:
  - script: `projeto_final/experimentos_automatos/run_gemma27b_a100_kz_automaton_55_70_noplots.sh`
  - maiorias iniciais: `55%, 60%, 65%, 70%`
  - `n_simulations=50`, agentes `30,60,90,120`, `max_iterations=2N`
  - `generate_plot=False` (sem geração de PNG)
- Resultado:
  - `96/96` jobs concluídos, `0` falhas
  - log: `projeto_final/experimentos_automatos/logs/run_gemma27b_kz_v9_v21_55_70_noplots.log`
  - para cada maioria (`55/60/65/70`): `24` pastas de resultado e `0` PNG gerados

## 1. Escopo e pastas importantes

- Raiz desta documentação:
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final`
- Interface:
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/interface/interface_v2.py`
- Runner batch/heatmap:
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/run_batch_png.py`
- Runner core da simulação:
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/llm_sim_runner.py`
- Benchmark por cenário:
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/projecao_simulacao/benchmark_scenario_models.py`
- Orquestrador benchmark single-load (RTX):
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/projecao_simulacao/run_singleload_benchmarks_rtx6000.sh`
- Orquestrador benchmark paralelo por família (Mac):
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/projecao_simulacao/run_macstudio_parallel_family_pairs_v9.sh`
- Utilitários compartilhados:
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/utils/utils.py`
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/utils/initial_distribution.py`
- Regra curta:
  - Funções compartilhadas entre interface, batch e benchmark devem viver em `projeto_final/utils/`, não em pastas específicas como `streamlit_test/`.

## 2. Infraestrutura dos servidores

### 2.1 Linux Server 2 (RTX Pro 6000)
- Host: `172.18.254.17`
- Usuário: `ncdia`
- Login por senha: `ncdiam47`
- LM Studio base URL: `http://172.18.254.17:1234/v1`
- Projeto remoto: `~/luan`
- Ambiente: `conda activate luan_conformidade`

### 2.2 Mac Studio
- Host: `172.18.254.18`
- Usuário: `ncdia`
- Login por senha: `NCDIAM47` (maiúsculas)
- LM Studio base URL: `http://172.18.254.18:1234/v1`
- Projeto remoto: `~/luan`
- Ambiente: `conda activate luan_conformidade`

### 2.3 Linux Server (A100) - suporte
- Host: `172.18.254.16`
- Usuário: `ncdia`
- Senha: `ncdiam47`
- Usado quando necessário para `llama-server`/execuções auxiliares.

## 3. SSH (padrão: senha, sem chave)

Comandos mínimos de login:

```bash
ssh ncdia@172.18.254.17
ssh ncdia@172.18.254.18
ssh ncdia@172.18.254.16
```

Validação forçando autenticação por senha:

```bash
sshpass -p 'ncdiam47' ssh -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password ncdia@172.18.254.17 "echo LOGIN_OK && hostname"
sshpass -p 'NCDIAM47' ssh -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password ncdia@172.18.254.18 "echo LOGIN_OK && hostname"
```

Se houver erro de host key no `.17`:

```bash
ssh-keygen -R 172.18.254.17
ssh-keygen -R '[172.18.254.17]:22'
ssh-keyscan -T 5 -t ed25519,ecdsa,rsa 172.18.254.17 >> ~/.ssh/known_hosts
chmod 600 ~/.ssh/known_hosts
```

## 4. Contrato de determinismo (importante)

### 4.1 Chamada LLM compartilhada
- A chamada para API é centralizada em:
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/utils/utils.py`
- Função: `call_llm_responses(...)`
- Endpoint usado: `/v1/responses` (OpenAI-compatible / LM Studio)
- Interface e benchmark reutilizam essa mesma função.

### 4.2 Distribuição inicial compartilhada
- Geração inicial centralizada em:
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/utils/initial_distribution.py`
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/utils/utils.py` (`generate_initial_distribution_shared`)
- Regra atual:
  - `seed_distribution` ímpar -> maioria inicial = `1`
  - `seed_distribution` par -> maioria inicial = `0`
  - isso garante balanço 50/50 em intervalos de seeds consecutivos (ex.: 1..30).
- `half_split`:
  - só fica ativo quando `majority_ratio == 0.5` e número de agentes é par;
  - caso contrário, fallback para modo `ratio`.

### 4.3 Seeds e parsing
- Seed de request padrão: `42`
- `qwen3`: adiciona `"/no_think"` automaticamente no prompt (`llm_sim_runner.py`) para evitar blocos de raciocínio atrapalhando parse.
- Parse de resposta usa token entre colchetes, robusto a `<think>...</think>`.

### 4.4 Regras de rodada/tokens
- `run_batch_png.py` agora permite `--max-rounds <= 2 * --agents` (não pode exceder esse teto).
- Benchmark CoT (`v21_*`) força `max_tokens >= 3000` em:
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/projecao_simulacao/benchmark_scenario_models.py`

## 5. Interface v2 (estado atual)

Arquivo:
- `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/interface/interface_v2.py`

Pontos relevantes:
- Presets de servidor já incluem:
  - `http://172.18.254.18:1234/v1` (Mac Studio)
  - `http://172.18.254.17:1234/v1` (Linux Server 2 RTX 6000)
  - `http://172.18.254.16:8081/v1` (A100 llama-server)
- Suporta variantes `v9_*` e `v21_*`.
- Para `v21_*` (CoT): `max_output_tokens = 3000`.
- Para variantes não-CoT: `max_output_tokens = 50`.
- Par `yes/no` exibido como `no(0)/yes(1)` e mapeado em ordem fixa (`no=0`, `yes=1`).
- Distribuição inicial da simulação vem de `generate_initial_distribution_shared` (mesma base usada no benchmark).

## 6. Benchmark - single-load (RTX 6000 `.17`)

Script:
- `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/projecao_simulacao/run_singleload_benchmarks_rtx6000.sh`

Características:
- Carrega 1 modelo por vez no LM Studio remoto.
- Executa benchmark por cenário e consolida em:
  - `consolidated_tokens_per_second.csv`
- Usa `FORCE_QWEN_NO_THINK=1` no benchmark.

Exemplo de execução local:

```bash
cd /Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/projecao_simulacao
bash ./run_singleload_benchmarks_rtx6000.sh
```

Variáveis úteis:
- `MODELS_ONLY=gemma4b,qwen4b_no_think,...`
- `PROMPT_VARIANT_OVERRIDE=v21_zero_shot_cot`
- `MAX_TOKENS_OVERRIDE=3000`
- `SCENARIO_DIR=.../seed_distribution_0001/memory_w_3`

## 7. Benchmark - paralelo por família (Mac `.18`)

Script:
- `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/projecao_simulacao/run_macstudio_parallel_family_pairs_v9.sh`

Pares atuais:
- `pair_qwen4b_qwen32b_nothink`
- `pair_gemma4b_gemma27b`

Saídas esperadas:
- `pair_summary.csv`
- `impact_vs_singleload_v9.csv`
- logs em `orchestrator.log`

Observações operacionais:
- Quando rodar **no próprio Mac Studio**, usar `USE_LOCAL_LMS=1` (não depende de `sshpass` dentro do servidor).
- Quando rodar da máquina local contra o Mac remoto, manter padrão de `SSH_HOST/SSH_PASS`.

Exemplo recomendado (no `.18`, dentro de `tmux`):

```bash
cd ~/luan/projeto_final/streamlit_test/projecao_simulacao
source ~/miniconda3/etc/profile.d/conda.sh
conda activate luan_conformidade
USE_LOCAL_LMS=1 \
SCENARIO_DIR="/Users/ncdia/luan/streamlit_test/batch_outputs/macstudio_v9_kz_t0_seed42_a30_r30_20260206_104752/seed_distribution_0001/memory_w_3" \
BASE_URL="http://172.18.254.18:1234/v1" \
API_MODELS_URL="http://172.18.254.18:1234/api/v0/models" \
bash ./run_macstudio_parallel_family_pairs_v9.sh
```

## 8. Fluxo operacional recomendado

1. Sincronizar código para servidor (`rsync` para `~/luan`).
2. Ativar `conda` (`luan_conformidade`).
3. Rodar em `tmux` para execuções longas.
4. Monitorar progresso por:
   - `tmux capture-pane -pt <sessao> | tail -n 80`
   - crescimento de `requests.jsonl`
5. Coletar resultados (`rsync` de volta) e consolidar CSVs.

## 9. Troubleshooting conhecido

### 9.1 `Too many authentication failures`
Forçar:
- `-o IdentitiesOnly=yes`
- `-o PubkeyAuthentication=no`
- `-o PreferredAuthentications=keyboard-interactive,password`

### 9.2 `.17` com erro de host key / `preauth`
Aplicar limpeza/reseed de `known_hosts` (seção 3).

### 9.3 `run_batch_png_failed` por import incorreto no remoto
Sintoma típico:
- `ModuleNotFoundError: projeto_final.utils.utils; 'projeto_final.utils' is not a package`

Causa:
- arquivos soltos em `~/luan/projeto_final` (ex.: `utils.py`) conflitando com pacote `~/luan/projeto_final/utils/`.

Correção:

```bash
rm -f ~/luan/projeto_final/utils.py ~/luan/projeto_final/initial_distribution.py \
      ~/luan/projeto_final/run_batch_png.py ~/luan/projeto_final/benchmark_scenario_models.py \
      ~/luan/projeto_final/run_macstudio_parallel_family_pairs_v9.sh
```

Manter os scripts somente em:
- `~/luan/projeto_final/streamlit_test/...`
- `~/luan/projeto_final/utils/...`

### 9.4 `KeyError: max_rounds` em cenários antigos
- O benchmark foi ajustado para fallback:
  - `max_rounds = 2 * n_rounds` quando ausente no `config.json`.

## 10. Checklist rápido antes de rodar

- Servidor responde em SSH por senha.
- `curl http://<host>:1234/v1/models` responde.
- Modelo desejado carregado no LM Studio.
- Ambiente `luan_conformidade` ativo.
- `SCENARIO_DIR` existe e contém `config.json`.
- Sessão `tmux` criada.
- Pasta de saída limpa quando necessário para re-run comparável.

## 11. Execução por modelo (Mac Studio, 3 deploys)

Quando rodar experimentos por família/modelo no Mac Studio (`172.18.254.18`), usar este fluxo:

### 11.1 Limpeza de resultados (antes de novo ciclo)

- Encerrar sessões antigas:
  - `tmux kill-session -t exp_llama8b_3deploy || true`
  - `tmux kill-session -t exp_llama70b_3deploy || true`
- Limpar resultados antigos (sem apagar código):
  - `~/luan/resultados/*`
  - `~/luan/resultados_*/*`
  - `~/luan/projeto_final/streamlit_test/projecao_simulacao/benchmark_results/*`
  - `~/luan/projeto_final/streamlit_test/batch_outputs/*`
  - `~/luan/projeto_final/experimentos_automatos/resultados_experimentos/*`

### 11.2 CSVs por modelo (formato esperado pelo orquestrador)

Arquivos:
- `~/luan/experimentos/por_modelo/experimentos_llama8b.csv`
- `~/luan/experimentos/por_modelo/experimentos_llama70b.csv`

Regras:
- `status=pendente` no início.
- `modelo_executado` deve vir como pool:
  - `llama8b-pool` para CSV do Llama 8B.
  - `llama70b-pool` para CSV do Llama 70B.
- Coluna de vizinhos deve ser `num_vizinhos` (não `n_vizinhos`) para `db_sqlite.init_db(...)`.

### 11.3 Deploys LM Studio (preservando GPT-OSS)

- Não descarregar `gpt-oss` se estiver carregado.
- Descarregar outros modelos não-Llama quando necessário.
- Carregar 3 deploys de cada:
  - `meta-llama-3.1-8b-instruct`, `:2`, `:3`
  - `meta-llama-3.1-70b-instruct`, `:2`, `:3`

Observação de código:
- `execucao_simultanea.py` deve manter `llama70b-pool` com 3 entradas (`:1/:2/:3`) para paralelizar.

### 11.4 Sessões TMUX dedicadas

- Sessão 1:
  - `exp_llama8b_3deploy`
  - `EXPERIMENTOS_CSV_PATH=experimentos/por_modelo/experimentos_llama8b.csv`
  - `EXPERIMENTOS_DB_PATH=experimentos/por_modelo/experimentos_llama8b.db`
  - `ACTIVE_POOLS_OVERRIDE=llama8b-pool`
- Sessão 2:
  - `exp_llama70b_3deploy`
  - `EXPERIMENTOS_CSV_PATH=experimentos/por_modelo/experimentos_llama70b.csv`
  - `EXPERIMENTOS_DB_PATH=experimentos/por_modelo/experimentos_llama70b.db`
  - `ACTIVE_POOLS_OVERRIDE=llama70b-pool`

Ambiente comum:
- `conda activate luan_conformidade`
- `LMSTUDIO_BASE_URL=http://172.18.254.18:1234/v1`
- `GLOBAL_LLM_MAX_INFLIGHT=3`
- Inicializar DB com `db_sqlite.init_db(EXPERIMENTOS_CSV_PATH)` e usar `SYNC_DB_FROM_CSV=false`.

### 11.5 Pós-processamento (automatos)

Após concluir os CSVs por modelo, usar os `dados_combinados_*.csv` gerados para simulação em:
- `~/luan/projeto_final/experimentos_automatos/run_automaton_numba.py`

Comando base:
```bash
python ~/luan/projeto_final/experimentos_automatos/run_automaton_numba.py \
  --csv <caminho_do_dados_combinados.csv> \
  --n_simulations 100 \
  --agents 100 \
  --max_iterations 200 \
  --initial-ratio 0.51
```

## 12. Cenário Gemma 27B pool (RTX 6000 `.17`)

Objetivo:
- Rodar `v9` e `v21` para tokens `kz` e `no_yes` no RTX 6000, com 2 deploys de `gemma-3-27b` balanceados por seleção aleatória por request.

Pré-requisitos no LM Studio (`.17`):
- Dois modelos carregados:
  - `gemma-3-27b-it`
  - `gemma-3-27b-it:2`

Mudança de código aplicada:
- `run_batch_png.py` agora aceita `--model-pool` (ou `LLM_MODEL_POOL`).
- `llm_sim_runner.py` escolhe 1 model ID aleatório por request, registrando:
  - `selected_model`
  - `model_pool`
  em `requests.jsonl`.

Script runtime único (remoto):
- `~/luan/projeto_final/streamlit_test/batch_outputs/prod_alltokens_v9_20260210_153615_rtx6000_halfsplit_20260219_103805/scripts_runtime/gemma27b_pool_by_strategy.sh`
- uso:
  - `bash .../gemma27b_pool_by_strategy.sh v9`
  - `bash .../gemma27b_pool_by_strategy.sh v21`

Separação de saída (modelo/estratégia/token):
- `.../gemma27b/token/kz/agents_<N>/...`
- `.../gemma27b/token/no_yes/agents_<N>/...`
- `.../gemma27b/cot/kz/agents_<N>/...`
- `.../gemma27b/cot/no_yes/agents_<N>/...`

Validação automática de tokens:
- Após cada execução por `agents`, os scripts validam `requests.jsonl`:
  - `parsed_token` deve estar no par permitido (`k/z` ou `no/yes`).
  - exibem `ok_tokens`, `invalid_tokens` e `usage_rows`.
  - falham com exit code `2` se houver token inválido.

Execução em TMUX (uma sessão por estratégia):
```bash
tmux kill-session -t gemma27b_v9_pool || true
tmux kill-session -t gemma27b_v21_pool || true

tmux new-session -d -s gemma27b_v9_pool
tmux send-keys -t gemma27b_v9_pool \
  'cd ~/luan && source ~/miniconda3/etc/profile.d/conda.sh && conda activate luan_conformidade && bash ~/luan/projeto_final/streamlit_test/batch_outputs/prod_alltokens_v9_20260210_153615_rtx6000_halfsplit_20260219_103805/scripts_runtime/gemma27b_pool_by_strategy.sh v9' Enter

tmux new-session -d -s gemma27b_v21_pool
tmux send-keys -t gemma27b_v21_pool \
  'cd ~/luan && source ~/miniconda3/etc/profile.d/conda.sh && conda activate luan_conformidade && bash ~/luan/projeto_final/streamlit_test/batch_outputs/prod_alltokens_v9_20260210_153615_rtx6000_halfsplit_20260219_103805/scripts_runtime/gemma27b_pool_by_strategy.sh v21' Enter
```
