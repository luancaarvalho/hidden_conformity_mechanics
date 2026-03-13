# Status de Migração: Interface v2 (Streamlit) -> v3 (Gradio)

Data de referência: 2026-02-24
Autor da execução: Codex (GPT-5)

## 1) Objetivo recebido
Migrar a interface `interface_v2.py` (Streamlit) para uma nova interface Gradio (`interface_v3_gradio.py`) **sem alterar a lógica experimental**.

Requisitos principais do pedido:
- manter lógica de experimento idêntica (prompts, parsing, memória, seed, stopping, logs, chamada LLM);
- remover padrão Streamlit de polling/rerun;
- implementar streaming por generator com atualização após cada resposta do LLM;
- adicionar Stop/Cancel seguro via mecanismo de cancelamento do Gradio;
- manter controles e saídas equivalentes (heatmap, logs, status, teste de conexão, JSON, etc.).

---

## 2) Arquivos criados/alterados

### Criado
- `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/interface/interface_v3_gradio.py`

### Alterações dentro deste arquivo
- migração da camada de UI para Gradio;
- remoção da seção Streamlit (`session_state`, `st.rerun`, loop de polling);
- criação de execução streaming com `run_simulation_stream(...)`;
- implementação de `SimulationRunner` (versão streaming da execução);
- fallback de compatibilidade para Gradio 6 no `queue()`.

### Não alterado
- não houve mudança em `interface_v2.py`;
- não houve alteração em `prompt_strategies.py`, `prompt_templates.yaml`, utilitários de lógica, parsing, ou runner principal de experimento;
- nenhuma mudança na semântica da chamada de inferência (`call_llm_responses`) no fluxo da simulação.

---

## 3) Decisões de implementação

1. **Lógica experimental preservada**
- O núcleo de simulação foi mantido com a mesma estrutura (rodada 0, loop por rodada, loop por agente, memória W, parse, atualização de estado, conformidade, logs).
- O método de consulta LLM usado no caminho principal continua com `call_llm_responses(...)` + parse `parse_llm_response(...)`.

2. **Troca de infraestrutura de execução**
- Antes: `SimulationWorker` + `thread` + `queue.Queue` + `st.rerun` periódico.
- Agora: `SimulationRunner.run_stream()` + generator `run_simulation_stream(...)` para yields orientados a evento.

3. **Streaming por passo de agente**
- O generator faz yield:
  - após inicialização da rodada 0;
  - após cada atualização de agente;
  - ao final (conformidade ou limite de rodadas / erro / parada).

4. **Stop/Cancel**
- Botão `⏹️ Parar` usa `cancels=[run_event]` no Gradio e também seta `stop_event` global ativo.
- Não foi usado kill de thread/processo.

5. **Heatmap sem IO extra**
- Continua usando `generate_heatmap_png(...)` (bytes em memória).
- Conversão bytes -> PIL (`Image.open(BytesIO(...)).copy()`) para `gr.Image(type="pil")`.

6. **Compatibilidade de versão Gradio**
- O pedido exigia `demo.queue(concurrency_count=1)`.
- Em Gradio 6.6.0, esse argumento não existe.
- Foi adicionado fallback:
  - tenta `demo.queue(concurrency_count=1)`;
  - em `TypeError`, usa `demo.queue(default_concurrency_limit=1)`.

---

## 4) Paridade de comportamento preservada

Itens explicitamente preservados em `interface_v3_gradio.py`:
- `DEFAULT_BASE_URL` idêntico;
- `AVAILABLE_MODELS` idêntica;
- `SERVER_PRESETS` idêntico;
- parsing de tokens (`parse_llm_response`) idêntico;
- mapeamento de pares de opinião (`get_opinion_pair`) idêntico;
- memória local (`_format_memory_block`) idêntica no conteúdo;
- regras de conformidade e parada idênticas;
- formato e ordem de gravação de logs de prompt/resposta mantidos.

---

## 5) Validações executadas

## 5.1 Ambiente `luan_conformidade`
- `conda env` encontrado: `luan_conformidade`.
- `gradio` foi instalado no ambiente (versão detectada: `6.6.0`).

## 5.2 Subida da UI
- Execução de `interface_v3_gradio.py` validada.
- Endpoint local respondeu em `http://127.0.0.1:7860` (`SERVER_OK=1`).

## 5.3 Bateria de validação funcional automatizada
Resultado consolidado final: **19 PASS / 0 FAIL**.

Cobertura validada:
1. paridade de estado v2 vs v3 (`np.array_equal(..., equal_nan=True)`);
2. yields de streaming (init + per-agent + término);
3. memória (`W`) no fluxo e formato;
4. jogo de conformidade (`no/yes`) e convergência esperada;
5. Stop resultando em status `Parado`;
6. ajuste automático de vizinhos ímpares;
7. preview de maioria inicial;
8. upload JSON válido e inválido;
9. controle de seed;
10. fluxo de teste LLM (mock e UI formatting);
11. presença dos componentes/labels principais da UI.

## 5.4 Teste real de LLM (não-mock)
- `test_model_connection(DEFAULT_BASE_URL, DEFAULT_MODEL)` retornou sucesso:
  - `raw_response: [k]`
  - `token: k`
  - latência aproximada: `15304 ms`

---

## 6) Estado atual para próximo agente

- Arquivo principal pronto para validação manual:
  - `/Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados/projeto_final/streamlit_test/interface/interface_v3_gradio.py`
- Linha de compatibilidade de fila (Gradio 6):
  - fallback entre `concurrency_count` e `default_concurrency_limit`.
- Hash SHA-256 do arquivo no estado atual:
  - `52dd9272436383e50c09767badc21aca9be7104857d2b8cecfab314993d984a6`
- Tamanho atual:
  - `1548` linhas.

---

## 7) Como reproduzir rapidamente

```bash
source ~/opt/miniconda3/etc/profile.d/conda.sh
conda activate luan_conformidade
cd /Users/luancarvalho/PycharmProjects/conformidade_experimento_resultados
python projeto_final/streamlit_test/interface/interface_v3_gradio.py
```

Depois abrir:
- `http://127.0.0.1:7860`

---

## 8) Observações finais

- Escopo entregue: migração de infraestrutura de UI (Streamlit -> Gradio) com preservação da lógica de experimento.
- Não há evidência de regressão funcional nos cenários cobertos pelos testes automatizados e pelo teste real de conexão LLM.
- Qualquer verificação adicional recomendada daqui em diante é de UX visual manual (layout/comportamento interativo no navegador) com cenários longos de execução.
