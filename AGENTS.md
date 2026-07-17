# AGENTS.md - Projeto Final (Conformidade LLM)

## 0. Pipeline Cientifico Canonico (RTX 5090)

O PDF `Hidden Conformity` define tres fases, que nao devem ser misturadas:

1. `extract_rules/`: Fase 1, extracao exaustiva de uma tabela de regra por modelo, prompt, par de tokens e n em 7/9/11.
2. `experimentos_automatos/`: Fase 2, validacao das regras congeladas no Density Classification Task.
3. `streamlit_test/`: Fase 3, jogo de conformidade com memoria e novas consultas ao LLM a cada rodada.

Regras obrigatorias para agentes:

- Ler `docs/RESEARCH_ARCHITECTURE.md` e o `AGENTS.md` da fase antes de agir.
- Nunca usar uma saida da Fase 1 na Fase 2 sem manifesto completo, tabela com `2^n` entradas e gate mecanico aprovado.
- Nunca apresentar a Fase 3 como tabela de regra estatica: memoria torna o estado dependente da historia.
- Codigo, contratos, manifests e summaries pequenos entram no Git; pesos, ambientes, logs, PNGs, CSVs extensos, `.npy` e bancos de execucao ficam em `artifacts/`.
- Cada run recebe diretorio novo e imutavel. Nao sobrescrever resultados anteriores.
- Na RTX 5090, nao matar nem alterar processos externos. Antes de usar GPU, verificar `nvidia-smi`, `tmux ls` e portas ativas.
- Para vLLM paralelo deterministico, exigir `VLLM_BATCH_INVARIANT=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `temperature=0`, seed `42` e dois replays exatamente iguais.
- Usar Conda e instalar dependencias com `uv pip --python <conda-prefix>/bin/python`; nao criar `.venv`.
- Branches de agentes usam prefixo `codex/`. Nao executar experimentos a partir de worktree suja.

Checkout canonico na RTX 5090:

- Git: `/home/liaan/Documentos/Luan/hidden_conformity_mechanics`
- Artefatos: `/home/liaan/Documentos/Luan/hidden_conformity_mechanics/artifacts`
- Runtime legado preservado: `/home/liaan/Documentos/Luan/temp_vllm/gradio_project`

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

## 0.0 Acesso SSH atual - maquinas via AnyDesk VPN

### RTX 6000 Pro Liaan

- Nome operacional: `rtx600`.
- Comando SSH recomendado:
  - `ssh rtx600`
- Alias legado ainda valido:
  - `ssh liaan-007-anydesk`
- Config local em `~/.ssh/config`:
  - `Host rtx600 liaan-007-anydesk`
  - `HostName 172.18.0.1`
  - `User liaan`
- Root remoto usado nos experimentos recentes:
  - `/home/liaan/Documentos/Luan`
- Root local para resultados importados dessa maquina:
  - `projeto_final/resultados_simulacoes_conformidade/RTX6000pro_liaan`
- Antes de iniciar novos experimentos, verificar `nvidia-smi`, sessoes `tmux`, portas vLLM ativas e processos externos; nao matar processos que nao foram criados pelo experimento atual.

### RTX 5090 Liaan 006

- Nome operacional: `rtx5090`.
- Comando SSH previsto:
  - `ssh rtx5090`
- Alias alternativo:
  - `ssh liaan-006-anydesk`
- Config local em `~/.ssh/config`:
  - `Host rtx5090 liaan-006-anydesk`
  - `HostName 172.19.0.1`
  - `User liaan`
  - `IdentityFile ~/.ssh/luan_006_ed25519`
- Status testado em 2026-07-16:
  - VPN AnyDesk estabelecida com IP local `172.19.0.2` e IP remoto `172.19.0.1`.
  - Porta `22/tcp` aberta.
  - O servidor SSH anuncia apenas `publickey`; login por senha nao foi oferecido.
  - A sessao de desktop confirmou o prompt `liaan@liaan-006`.
  - Login validado com `ssh rtx5090` usando `~/.ssh/luan_006_ed25519`.
  - Fingerprint validado: `SHA256:fnHdpgmqQ6d9yMAVEyLCXWkg7+69gUSnVZNKmm08QFk`.
  - Permissoes validadas: home `750`, `~/.ssh` `700`, `authorized_keys` `600`.
  - `AllowUsers` preserva `luan` e autoriza `liaan` somente a partir de `172.19.0.2`.
  - GPU confirmada: `NVIDIA GeForce RTX 5090`, `32607 MiB`, driver `580.159.03`.
- Requisito operacional: manter a sessao VPN do AnyDesk ativa antes de executar `ssh rtx5090`.
- Padrao de ambientes Python nesta maquina:
  - criar ambientes com Conda, preferencialmente por prefixo dentro do root do experimento;
  - instalar pacotes com `uv pip install --python <conda-prefix>/bin/python ...`;
  - nao criar `.venv` nem usar Python gerenciado pelo `uv` neste fluxo;
  - ambientes atuais do teste Gemma/Gradio: `/home/liaan/Documentos/Luan/temp_vllm/conda_envs/vllm` e `/home/liaan/Documentos/Luan/temp_vllm/conda_envs/gradio`.
- Regra de determinismo vLLM validada na RTX 5090:
  - iniciar o servidor com `VLLM_BATCH_INVARIANT=1` e `VLLM_USE_FLASHINFER_SAMPLER=0`;
  - manter `--generation-config vllm`, `--seed 42` e `temperature=0` nas requisicoes;
  - `temperature=0` e seed fixo, sem batch invariance, nao garantem trajetorias identicas sob concorrencia: o controle de 2026-07-16 teve somente `19/40` celulas exatamente iguais;
  - todo batch cientifico paralelo deve ser executado duas vezes e passar `100%` de igualdade em `states.npy`, PNG e log normalizado antes de ser tratado como deterministico;
  - o replay com batch invariance passou `40/40` celulas exatamente iguais nos quatro modos (`standard/conformity` x `only-token/CoT`), com menor throughput como tradeoff;
  - evidencias locais: `resultados_simulacoes_conformidade/RTX5090_liaan/gemma4b_01_n30_neigh7_W3_batch_invariant_comparison_20260716T222142Z/`.

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

## 0.2 Atualização operacional (2026-03-26)

### RTX 6000 Pro - paralelismo por rodada com múltiplas instâncias do mesmo Llama
- Host atual: `172.18.254.170`
- Usuário: `ncdia`
- Autenticacao: usar configuracao SSH local; credenciais nao pertencem ao Git.
- LM Studio base URL: `http://172.18.254.170:1234/v1`
- Branch de trabalho para esta mudança:
  - `codex/parallel-round-lane-scheduler`
- Script alterado:
  - `streamlit_test/llm_sim_runner.py`
- Semântica nova do runner batch:
  - a rodada `r` monta todos os prompts apenas com `states[r-1]`
  - existe uma lane por entrada em `model_pool`
  - quando uma lane termina, ela pega imediatamente o próximo agente pendente da mesma rodada
  - a próxima rodada só começa quando todos os agentes da rodada atual terminam
- Smoke de referência já validado:
  - `v21_zero_shot_cot`
  - `agents=30`
  - `seed_distribution=1`
  - `memory_w=3,5`
  - `model_pool=meta-llama-3.1-8b-instruct,meta-llama-3.1-8b-instruct:2,meta-llama-3.1-8b-instruct:3`
- Regra de prompt:
  - `"/no_think"` é somente para modelos `qwen`
  - rodando `run_batch_png.py` diretamente com Llama, o sufixo não deve aparecer
- Root de saída da RTX 6000:
  - remoto: `~/luan/projeto_final/streamlit_test/batch_outputs/rtx6000`
  - local: `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/batch_outputs/rtx6000`
- Regra de PNG:
  - o artefato oficial do batch é um único `sim*.png`
  - não depender mais de `heatmap_*.png` para considerar um run completo

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
- Host: `172.18.254.170`
- Usuário: `ncdia`
- Autenticacao: usar configuracao SSH local; credenciais nao pertencem ao Git.
- LM Studio base URL: `http://172.18.254.170:1234/v1`
- Projeto remoto: `~/luan`
- Ambiente: `conda activate luan_conformidade`

### 2.2 Mac Studio
- Host: `172.18.254.18`
- Usuário: `ncdia`
- Autenticacao: usar configuracao SSH local; credenciais nao pertencem ao Git.
- LM Studio base URL: `http://172.18.254.18:1234/v1`
- Projeto remoto: `~/luan`
- Ambiente: `conda activate luan_conformidade`

### 2.3 Linux Server (A100) - suporte
- Host: `172.18.254.16`
- Usuário: `ncdia`
- Autenticacao: usar configuracao SSH local; credenciais nao pertencem ao Git.
- Usado quando necessário para `llama-server`/execuções auxiliares.

## 3. SSH

Comandos mínimos de login:

```bash
ssh ncdia@172.18.254.170
ssh ncdia@172.18.254.18
ssh ncdia@172.18.254.16
```

As credenciais devem permanecer no agente SSH, keychain ou configuracao local ignorada pelo Git. Nunca registrar senha ou token neste arquivo.

Se houver erro de host key no `.170`:

```bash
ssh-keygen -R 172.18.254.170
ssh-keygen -R '[172.18.254.170]:22'
ssh-keyscan -T 5 -t ed25519,ecdsa,rsa 172.18.254.170 >> ~/.ssh/known_hosts
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
- Llama e Gemma não devem receber `"/no_think"` quando rodando `run_batch_png.py` diretamente.
- Parse de resposta usa token entre colchetes, robusto a `<think>...</think>`.

### 4.4 Regras de rodada/tokens
- `run_batch_png.py` agora permite `--max-rounds <= 2 * --agents` (não pode exceder esse teto).
- No modo com `model_pool`, `llm_sim_runner.py` usa paralelismo por rodada com barreira obrigatória:
  - cada rodada usa somente `states[r-1]`
  - cada lane usa uma entrada fixa de `model_pool`
  - quando uma lane termina, ela pega o próximo agente pendente da mesma rodada
  - a rodada seguinte só começa após o fechamento completo da rodada atual
- Benchmark CoT (`v21_*`) força `max_tokens >= 3000` em:
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/projecao_simulacao/benchmark_scenario_models.py`

## 5. Interface v2 (estado atual)

Arquivo:
- `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/interface/interface_v2.py`

Pontos relevantes:
- Presets de servidor já incluem:
  - `http://172.18.254.18:1234/v1` (Mac Studio)
  - `http://172.18.254.170:1234/v1` (Linux Server 2 RTX 6000)
  - `http://172.18.254.16:8081/v1` (A100 llama-server)
- Suporta variantes `v9_*` e `v21_*`.
- Para `v21_*` (CoT): `max_output_tokens = 3000`.
- Para variantes não-CoT: `max_output_tokens = 50`.
- Par `yes/no` exibido como `no(0)/yes(1)` e mapeado em ordem fixa (`no=0`, `yes=1`).
- Distribuição inicial da simulação vem de `generate_initial_distribution_shared` (mesma base usada no benchmark).

## 6. Benchmark - single-load (RTX 6000 `.170`)

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
   - lembrar que `requests.jsonl` faz flush em bloco; janela curta sem crescimento não prova travamento
5. Coletar resultados (`rsync` de volta) e consolidar CSVs.

## 9. Troubleshooting conhecido

### 9.1 `Too many authentication failures`
Forçar:
- `-o IdentitiesOnly=yes`
- `-o PubkeyAuthentication=no`
- `-o PreferredAuthentications=keyboard-interactive,password`

### 9.2 `.170` com erro de host key / `preauth`
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
