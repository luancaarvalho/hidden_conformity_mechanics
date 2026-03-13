#!/usr/bin/env python3
"""
Streamlit App - Simulação de Conformidade com LLM (Versão Llama Server)
Adiciona suporte explícito e presets para clusters Llama Server (A100).
"""

import os
import sys
import re
import time
import random
import threading
import queue
import json
from io import BytesIO
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import yaml
import streamlit as st
from openai import OpenAI
import requests

def _find_repo_root() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(d, "prompt_strategies.py")) and os.path.exists(os.path.join(d, "prompt_templates.yaml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


REPO_ROOT = _find_repo_root()

# Adiciona o repo root ao path para importar módulos existentes
sys.path.insert(0, REPO_ROOT)

try:
    from prompt_strategies import get_prompt_strategy, PromptStrategy
except ImportError:
    get_prompt_strategy = None
    PromptStrategy = None

from projeto_final.utils.initial_distribution import generate_initial_distribution

# ==============================================================================
# CONSTANTES E CONFIGURAÇÃO
# ==============================================================================

DEFAULT_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://172.18.254.18:1234/v1")
DEFAULT_API_KEY = "lm-studio"
DEFAULT_MODEL = "google/gemma-3-12b" # Default atualizado para o experimento atual

# Presets de Servidores
SERVER_PRESETS = {
    "🚀 Llama Server (Porta 8085)": "http://172.18.254.16:8085/v1",
    "LM Studio (Mac Studio)": "http://172.18.254.18:1234/v1",
    "LM Studio (Linux Server 2 RTX Pro 6000)": "http://172.18.254.17:1234/v1",
    "🚀 Llama Server 1 (A100 - 8081)": "http://172.18.254.16:8081/v1",
    "🚀 Llama Server 2 (A100 - 8082)": "http://172.18.254.16:8082/v1",
    "🚀 Llama Server 3 (A100 - 8083)": "http://172.18.254.16:8083/v1",
    "✏️ Customizado": "custom"
}

AVAILABLE_MODELS = [
    "google/gemma-3-12b",
    "google/gemma-3-4b",
    "meta-llama-3.1-8b-instruct",
    "meta-llama-3.1-70b-instruct",
    "qwen3-4b",
    "qwen/qwen3-32b",
]

# Mapeamento de tokens de opinião para valores numéricos
OPINION_MAP = {
    'k': 0, 'z': 1,
    'a': 0, 'b': 1,
    '0': 0, '1': 1,
    'p': 0, 'q': 1,
    'α': 0, 'β': 1,
    '△': 0, '○': 1,
    '⊕': 0, '⊖': 1,
    'ł': 0, 'þ': 1,
    'no': 0, 'yes': 1,
}

# Variantes de prompt disponíveis (fallback se YAML não estiver disponível)
FALLBACK_VARIANTS = [
    'v5_original', 'v6_lista_indices', 'v7_offsets',
    'v8_visual', 'v9_lista_completa_meio', 'v10_lista_indice_especifico',
    'v11_lista_completa_meio_sem_current',
    'v12_python', 'v13_incidence', 'v14_json',
    'v15_compact_symbol', 'v16_cartesian',
    'v17_graph_of_thought', 'v18_rule', 'v19_lista_completa_meio_com_raciocinio',
    'v20_lista_completa_meio_raciocinio_primeiro',
    'v20_lista_completa_meio_raciocinio_primeiro_ab',
    'v20_lista_completa_meio_raciocinio_primeiro_01',
    'v9_lista_completa_meio_ab',
    'v9_lista_completa_meio_01',
    'v9_lista_completa_meio_kz',
    'v20_lista_completa_meio_raciocinio_primeiro_kz',
    'v21_zero_shot_cot',
    'v21_zero_shot_cot_ab',
    'v21_zero_shot_cot_01',
]

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def load_yaml_variants(yaml_path: str) -> List[str]:
    """Carrega lista de variantes do arquivo YAML."""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)
        return list(configs.keys()) if configs else FALLBACK_VARIANTS
    except Exception:
        return FALLBACK_VARIANTS


def get_yaml_path() -> str:
    """Retorna o caminho para o arquivo prompt_templates.yaml."""
    yaml_path = os.path.join(REPO_ROOT, "prompt_templates.yaml")
    if os.path.exists(yaml_path):
        return yaml_path
    # Tenta no diretório atual
    local_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_templates.yaml")
    if os.path.exists(local_yaml):
        return local_yaml
    return yaml_path  # Retorna o esperado mesmo se não existir


def parse_llm_response(response_text: str) -> Optional[str]:
    """
    Extrai a opinião da resposta do LLM.
    Aceita todos os tokens: k/z, a/b, 0/1, p/q, α/β, △/○, ⊕/⊖, ł/þ, yes/no
    Aceita respostas com espaços: [k], [ k ], etc.
    """
    response_clean = response_text.lower().strip()
    
    # Verifica formatos exatos
    exact_matches = {
        '[k]': 'k', '[z]': 'z',
        '[a]': 'a', '[b]': 'b',
        '[0]': '0', '[1]': '1',
        '[p]': 'p', '[q]': 'q',
        '[α]': 'α', '[β]': 'β',
        '[△]': '△', '[○]': '○',
        '[⊕]': '⊕', '[⊖]': '⊖',
        '[ł]': 'ł', '[þ]': 'þ',
        '[yes]': 'yes', '[no]': 'no'
    }
    
    if response_clean in exact_matches:
        return exact_matches[response_clean]
    
    # Fallback: busca padrão em qualquer lugar da resposta
    match = re.search(r'\\[\\s\\*(k|z|a|b|0|1|p|q|α|β|△|○|⊕|⊖|ł|þ|yes|no)\\\s*\\]', response_clean)
    if match:
        return match.group(1)
    
    return None


def get_default_opinion(variant: str) -> str:
    """Retorna a opinião padrão baseada na variante do prompt."""
    if '_ab' in variant:
        return 'a'
    elif '_01' in variant:
        return '0'
    elif '_αβ' in variant or 'greek' in variant.lower():
        return 'α'
    elif '_△○' in variant or 'geometric' in variant.lower():
        return '△'
    elif '_⊕⊖' in variant or 'math' in variant.lower():
        return '⊕'
    elif '_pq' in variant:
        return 'p'
    elif '_łþ' in variant:
        return 'ł'
    elif '_yesno' in variant or 'yesno' in variant.lower():
        return 'no'
    else:
        return 'k'


def get_opinion_pair(variant: str) -> Tuple[str, str]:
    """Retorna o par de opiniões baseado na variante."""
    if '_ab' in variant:
        return ('a', 'b')
    elif '_01' in variant:
        return ('0', '1')
    elif '_αβ' in variant:
        return ('α', 'β')
    elif '_△○' in variant:
        return ('△', '○')
    elif '_⊕⊖' in variant:
        return ('⊕', '⊖')
    elif '_pq' in variant:
        return ('p', 'q')
    elif '_łþ' in variant:
        return ('ł', 'þ')
    elif '_yesno' in variant:
        return ('no', 'yes')
    else:
        return ('k', 'z')


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Limita um valor entre min e max."""
    return max(min_val, min(max_val, value))


def generate_heatmap_png(
    states: np.ndarray,
    n_agents: int,
    n_rounds: int,
    current_round: int = -1,
    current_agent: int = -1
) -> bytes:
    """
    Gera uma imagem PNG do heatmap em memória.
    
    Args:
        states: Matriz de estados (n_rounds x n_agents)
        n_agents: Número de agentes
        n_rounds: Número de rodadas
        current_round: Rodada atual sendo processada (-1 = nenhuma)
        current_agent: Agente atual sendo processado (-1 = nenhum)
    
    Returns:
        bytes do PNG gerado
    """
    # Dimensionamento dinâmico
    cell_width = 0.15
    cell_height = 0.12
    fig_width = clamp(n_agents * cell_width + 2.5, 8, 25)
    fig_height = clamp(n_rounds * cell_height + 1.5, 6, 20)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Prepara dados para exibição (NaN = cinza claro)
    display_data = states.copy().astype(float)
    
    # Configura colormap: 0=branco (K/primeiro token), 1=preto (Z/segundo token), NaN=cinza claro
    # Colormap 'binary' padrão: 0=branco, 1=preto (igual ao run_automaton_numba.py)
    cmap = plt.cm.binary.copy()
    cmap.set_bad(color='#E0E0E0')  # Cinza claro para células não processadas
    
    # Aspect ratio dinâmico
    aspect_ratio = (n_agents / n_rounds) * 0.8 if n_rounds > 0 else 1.0
    
    im = ax.imshow(
        display_data,
        cmap=cmap,
        interpolation='nearest',
        origin='upper',
        aspect=aspect_ratio,
        vmin=0,
        vmax=1
    )
    
    # Ticks adaptativos
    max_x_ticks = min(12, n_agents)
    max_y_ticks = min(15, n_rounds)
    
    x_tick_step = max(1, n_agents // max_x_ticks)
    y_tick_step = max(1, n_rounds // max_y_ticks)
    
    ax.set_xticks(range(0, n_agents, x_tick_step))
    ax.set_yticks(range(0, n_rounds, y_tick_step))
    
    ax.set_xlabel('Agentes', fontsize=10, fontweight='bold')
    ax.set_ylabel('Rodadas', fontsize=10, fontweight='bold')
    ax.set_title('Simulação de Conformidade - Heatmap', fontsize=12, fontweight='bold')
    
    # Destaca célula atual com borda vermelha
    if current_round >= 0 and current_agent >= 0:
        rect = plt.Rectangle(
            (current_agent - 0.5, current_round - 0.5),
            1, 1,
            fill=False,
            edgecolor='red',
            linewidth=2
        )
        ax.add_patch(rect)
    
    # Grid sutil
    ax.set_xticks(np.arange(-0.5, n_agents, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rounds, 1), minor=True)
    ax.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    # Salva em memória como PNG
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue()


# ==============================================================================
# WORKER DE SIMULAÇÃO (THREAD BACKGROUND)
# ==============================================================================

class SimulationWorker:
    """Worker que executa a simulação em background com suporte a cancelamento."""
    
    def __init__(
        self,
        n_agents: int,
        n_rounds: int,
        n_neighbors: int,
        temperature: float,
        prompt_variant: str,
        base_url: str,
        model: str,
        delay_ms: int,
        initial_majority: int,
        update_queue: queue.Queue,
        stop_event: threading.Event,
        seed: Optional[int] = None,
        initial_config_json: Optional[Dict] = None,
        memory_window: int = 0,
        memory_format: str = "timeline"
    ):
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.n_neighbors = n_neighbors
        self.temperature = temperature
        self.prompt_variant = prompt_variant
        self.base_url = base_url
        self.model = model
        self.delay_ms = delay_ms
        self.initial_majority = initial_majority  # Porcentagem de opinião 0 (k/a/0)
        self.update_queue = update_queue
        self.stop_event = stop_event
        self.seed = seed
        self.initial_config_json = initial_config_json
        self.memory_window = memory_window  # W: número de turnos anteriores
        # Memory block format removed; keep a single stable prompt structure.
        self.memory_format = "timeline"
        
        # Estado da simulação
        self.states = np.full((n_rounds, n_agents), np.nan)
        self.client = None
        self.strategy = None
        self.opinion_pair = get_opinion_pair(prompt_variant)
        
        # Arquivo de log (TXT - formato legível)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"prompt_log_{self.prompt_variant}_W{self.memory_window}_{timestamp}.txt"
        
        # Define pasta de logs
        # Keep logs under `streamlit_test/logs` even though this file lives in `streamlit_test/interface/`.
        base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        save_dir = os.path.join(base_dir, "logs")
        os.makedirs(save_dir, exist_ok=True)
        
        self.log_filepath = os.path.join(save_dir, log_filename)
        
    def _log(self, message: str):
        """Envia mensagem de log para a fila."""
        self.update_queue.put({
            'type': 'log',
            'message': message
        })
    
    def _update_heatmap(self, round_idx: int, agent_idx: int):
        """Envia atualização do heatmap para a fila."""
        png_bytes = generate_heatmap_png(
            self.states,
            self.n_agents,
            self.n_rounds,
            round_idx,
            agent_idx
        )
        self.update_queue.put({
            'type': 'heatmap',
            'image': png_bytes,
            'round': round_idx,
            'agent': agent_idx
        })
    
    def _should_stop(self) -> bool:
        """Verifica se deve parar a execução."""
        return self.stop_event.is_set()
    
    def _check_conformity(self, round_idx: int) -> bool:
        """Verifica se a conformidade foi atingida."""
        round_states = self.states[round_idx]
        valid_states = round_states[~np.isnan(round_states)]
        if len(valid_states) == 0:
            return False
        return np.all(valid_states == valid_states[0])
    
    def _get_conformity_opinion(self, round_idx: int) -> str:
        """Retorna a opinião predominante quando há conformidade."""
        round_states = self.states[round_idx]
        valid_states = round_states[~np.isnan(round_states)]
        if len(valid_states) > 0:
            opinion_value = int(valid_states[0])
            return self.opinion_pair[opinion_value]
        return "?"
    
    def _get_neighbor_indices(self, agent_idx: int) -> Tuple[List[int], List[int]]:
        """Retorna índices dos vizinhos (esquerda, direita)."""
        half = self.n_neighbors // 2
        left_indices = [((agent_idx - i) % self.n_agents) for i in range(1, half + 1)]
        right_indices = [((agent_idx + i) % self.n_agents) for i in range(1, half + 1)]
        return left_indices[::-1], right_indices
    
    def _state_to_token(self, state_val: float) -> str:
        """Converte valor numérico para token."""
        if np.isnan(state_val):
            return "?"
        return self.opinion_pair[int(state_val)]
    
    def _get_neighbors(self, agent_idx: int, round_states: np.ndarray) -> Tuple[List[str], List[str]]:
        """Obtém vizinhos à esquerda e direita."""
        half = self.n_neighbors // 2
        left = []
        right = []
        for i in range(1, half + 1):
            neighbor_idx = (agent_idx - i) % self.n_agents
            val = round_states[neighbor_idx]
            if not np.isnan(val):
                left.insert(0, self.opinion_pair[int(val)])
        for i in range(1, half + 1):
            neighbor_idx = (agent_idx + i) % self.n_agents
            val = round_states[neighbor_idx]
            if not np.isnan(val):
                right.append(self.opinion_pair[int(val)])
        return left, right
    
    def _format_memory_block(self, agent_idx: int, current_round: int) -> str:
        """Constrói bloco de memória."""
        if self.memory_window == 0 or current_round == 0:
            return ""
        
        start_round = max(0, current_round - self.memory_window)
        end_round = current_round
        
        if start_round >= end_round:
            return ""
        
        left_indices, right_indices = self._get_neighbor_indices(agent_idx)
        memory_lines = []
        
        memory_lines.append("=== MEMORY (Previous Rounds) ===")
        for r in range(start_round, end_round):
            self_token = self._state_to_token(self.states[r, agent_idx])
            left_tokens = [self._state_to_token(self.states[r, idx]) for idx in left_indices]
            right_tokens = [self._state_to_token(self.states[r, idx]) for idx in right_indices]
            memory_lines.append(f"Round {r}:")
            memory_lines.append(f"  You: [{self_token}]")
            memory_lines.append(f"  Left: {left_tokens}")
            memory_lines.append(f"  Right: {right_tokens}")
        return "\n".join(memory_lines) if memory_lines else ""
    
    def _query_llm_llama_server(self, system_prompt: str, user_prompt: str, agent_idx: int, round_idx: int) -> Tuple[str, Optional[str]]:
        """
        Executa chamada ESPECÍFICA para llama-server (OpenAI Compatible) via Requests.
        Evita o client OpenAI para ter controle total sobre timeout e headers.
        """
        max_attempts = 5
        
        # Garante URL limpa para /v1/chat/completions
        base_url_clean = self.base_url.rstrip("/")
        if not base_url_clean.endswith("/v1"):
            base_url_clean += "/v1"
        url = f"{base_url_clean}/chat/completions"

        for attempt in range(max_attempts):
            if self._should_stop():
                return ("", None)
            
            try:
                # LOG DE DEBUG DA TEMPERATURA
                self._log(f"🌡️ Enviando: Temp={self.temperature}, ReqSeed=42")

                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 50  # Importante para conformidade
                }
                
                payload["seed"] = 42
                
                # Timeout agressivo para não travar a UI (120s)
                response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
                response.raise_for_status()
                
                data = response.json()
                raw_response = data['choices'][0]['message']['content'].strip()
                token = parse_llm_response(raw_response)
                
                # Log e Retorno
                if token is not None:
                    self._save_log(round_idx, agent_idx, attempt, system_prompt, user_prompt, raw_response, token, payload)
                    return (raw_response, token)
                
                self._log(f"⚠️ Tentativa {attempt + 1}: Resposta '{raw_response[:20]}...' sem token válido.")
                
            except Exception as e:
                self._log(f"❌ Erro Llama Server (Tentativa {attempt + 1}): {str(e)[:100]}")
                time.sleep(1) # Backoff simples
        
        return ("", None)

    def _save_log(self, round_idx, agent_idx, attempt, sys_prompt, user_prompt, raw_resp, token, payload):
        """Salva log detalhado da interação."""
        curr_time = time.strftime("%Y-%m-%d %H:%M:%S")
        separator = "═" * 80
        sub_separator = "-" * 80
        
        # Extrai configs relevantes do payload para o cabeçalho
        config_str = (
            f"Temp={payload.get('temperature')} | Top-K={payload.get('top_k')} | "
            f"Top-P={payload.get('top_p')} | Seed={payload.get('seed')}"
        )
        
        log_entry = (
            f"{separator}\n"
            f"TIMESTAMP: {curr_time} | ROUND: {round_idx} | AGENT: {agent_idx} | ATTEMPT: {attempt + 1}\n"
            f"CONFIG: {config_str}\n"
            f"{separator}\n\n"
            f"{sys_prompt}\n\n"
            f"{user_prompt}\n\n"
            f">>> RESPONSE (RAW)\n"
            f"{sub_separator}\n"
            f"{raw_resp}\n"
            f"{sub_separator}\n\n"
            f">>> RESULT: {token}\n\n"
        )
        with open(self.log_filepath, 'a', encoding='utf-8') as f:
            f.write(log_entry)

    def run(self):
        """Executa a simulação completa."""
        try:
            self._log("🚀 Iniciando simulação (Modo Llama Server)...")
            self._log(f"🔗 Endpoint: {self.base_url}")
            self._log(f"💾 Log: {self.log_filepath}")
            
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
                self._log(f"🎲 Seed: {self.seed}")
            
            # Carrega estratégia
            yaml_path = get_yaml_path()
            if get_prompt_strategy is not None:
                try:
                    self.strategy = get_prompt_strategy(self.prompt_variant, yaml_path)
                except Exception as e:
                    self._log(f"⚠️ Erro ao carregar estratégia: {e}")
                    self.update_queue.put({'type': 'error', 'message': str(e)})
                    return
            else:
                self.update_queue.put({'type': 'error', 'message': 'prompt_strategies não disponível'})
                return
            
            # Inicializa rodada 0
            if self.initial_config_json is not None:
                self._log(f"📂 Carregando distribuição do arquivo (sim #{self.initial_config_json['sim_number']})...")
                initial_opinions = np.array(self.initial_config_json['initial_config'], dtype=int)
                if len(initial_opinions) != self.n_agents:
                    # Ajuste de tamanho
                    if len(initial_opinions) > self.n_agents:
                        initial_opinions = initial_opinions[:self.n_agents]
                    else:
                        repeat_times = self.n_agents // len(initial_opinions) + 1
                        initial_opinions = np.tile(initial_opinions, repeat_times)[:self.n_agents]
            else:
                self._log(f"📊 Gerando nova distribuição ({self.initial_majority}% maioria inicial)...")
                majority_ratio = self.initial_majority / 100.0
                distribution_mode = "half_split" if int(self.initial_majority) == 50 else "ratio"
                seed_distribution = int(self.seed) if self.seed is not None else int(time.time() * 1000) % 2147483647
                try:
                    dist = generate_initial_distribution(
                        n_agents=self.n_agents,
                        seed_distribution=seed_distribution,
                        majority_ratio=majority_ratio,
                        mode=distribution_mode,
                    )
                except ValueError:
                    dist = generate_initial_distribution(
                        n_agents=self.n_agents,
                        seed_distribution=seed_distribution,
                        majority_ratio=majority_ratio,
                        mode="ratio",
                    )
                    distribution_mode = "ratio"
                initial_opinions = dist.opinions
                self._log(
                    f"📌 Distribuição inicial: mode={distribution_mode}, seed_dist={seed_distribution}, "
                    f"count_0={dist.count_0}, count_1={dist.count_1}"
                )
            
            # Aplica rodada 0
            for i in range(self.n_agents):
                self.states[0, i] = initial_opinions[i]
            self._update_heatmap(0, self.n_agents - 1)
            self._log("✅ Rodada 0 inicializada")
            
            # Loop Principal
            for r in range(1, self.n_rounds):
                if self._should_stop():
                    self.update_queue.put({'type': 'stopped'})
                    return
                
                self._log(f"🔄 Rodada {r}/{self.n_rounds - 1}...")
                prev_states = self.states[r - 1].copy()
                
                for i in range(self.n_agents):
                    if self._should_stop():
                        self.update_queue.put({'type': 'stopped'})
                        return
                    
                    left, right = self._get_neighbors(i, prev_states)
                    current_val = prev_states[i]
                    current_opinion = self.opinion_pair[int(current_val)] if not np.isnan(current_val) else get_default_opinion(self.prompt_variant)
                    
                    try:
                        system_prompt, user_prompt = self.strategy.build_prompt(left=left, right=right, current_opinion=current_opinion)
                    except Exception as e:
                        self._log(f"⚠️ Erro no prompt: {e}")
                        self.states[r, i] = np.nan
                        continue
                    
                    memory_block = self._format_memory_block(i, r)
                    if memory_block:
                        user_prompt = memory_block + "\n\n" + user_prompt
                    
                    # CHAMADA MODIFICADA: Usa método específico para Llama Server
                    raw_response, token = self._query_llm_llama_server(system_prompt, user_prompt, i, r)
                    
                    if token is not None and token in OPINION_MAP:
                        self.states[r, i] = OPINION_MAP[token]
                        self._log(f"✅ Agente {i}: [{token}]")
                    else:
                        self.states[r, i] = np.nan
                        self._log(f"⚠️ Agente {i}: Falha na resposta")
                    
                    self._update_heatmap(r, i)
                
                if self._check_conformity(r):
                    opinion = self._get_conformity_opinion(r)
                    self._log(f"🎯 CONFORMIDADE ATINGIDA na rodada {r}!")
                    self.update_queue.put({'type': 'conformity', 'round': r, 'opinion': opinion})
                    return
            
            self._log("✅ Simulação concluída!")
            self.update_queue.put({'type': 'completed'})
            
        except Exception as e:
            self._log(f"❌ Erro fatal: {str(e)}")
            self.update_queue.put({'type': 'error', 'message': str(e)})


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

def init_session_state():
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = None
    if 'update_queue' not in st.session_state:
        st.session_state.update_queue = None
    if 'worker_thread' not in st.session_state:
        st.session_state.worker_thread = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = "Desconectado"
    if 'last_test_result' not in st.session_state:
        st.session_state.last_test_result = None
    if 'simulation_status' not in st.session_state:
        st.session_state.simulation_status = "Parado"


def process_queue_updates():
    if st.session_state.update_queue is None:
        return False
    try:
        while True:
            update = st.session_state.update_queue.get_nowait()
            if update['type'] == 'heatmap':
                st.session_state.current_image = update['image']
            elif update['type'] == 'log':
                st.session_state.log_messages.append(update['message'])
                if len(st.session_state.log_messages) > 50:
                    st.session_state.log_messages = st.session_state.log_messages[-50:]
            elif update['type'] in ('completed', 'stopped', 'error', 'conformity'):
                st.session_state.simulation_running = False
                if update['type'] == 'completed':
                    st.session_state.simulation_status = "Concluído"
                elif update['type'] == 'conformity':
                    st.session_state.simulation_status = f"🎯 Conformidade [{update.get('opinion')}]"
                elif update['type'] == 'error':
                    st.session_state.simulation_status = "Erro"
                else:
                    st.session_state.simulation_status = "Parado"
    except queue.Empty:
        pass


def test_model_connection(base_url: str, model: str) -> Dict[str, Any]:
    try:
        # Teste via Requests direto para validar endpoint Llama Server
        base_url_clean = base_url.rstrip("/")
        if not base_url_clean.endswith("/v1"):
            base_url_clean += "/v1"
        url = f"{base_url_clean}/chat/completions"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Say [k]"}],
            "max_tokens": 10,
            "temperature": 0.0
        }
        
        start = time.time()
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        lat = (time.time() - start) * 1000
        resp.raise_for_status()
        
        raw = resp.json()['choices'][0]['message']['content'].strip()
        token = parse_llm_response(raw)
        
        return {'success': True, 'raw_response': raw, 'token': token, 'latency_ms': lat}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def start_simulation(n_agents, n_rounds, n_neighbors, temperature, prompt_variant, base_url, model, delay_ms, initial_majority, seed, initial_config_json, memory_window, memory_format):
    st.session_state.log_messages = []
    st.session_state.current_image = None
    st.session_state.simulation_status = "Rodando"
    st.session_state.update_queue = queue.Queue()
    st.session_state.stop_event = threading.Event()
    
    worker = SimulationWorker(
        n_agents, n_rounds, n_neighbors, temperature, prompt_variant, base_url, model, delay_ms, 
        initial_majority, st.session_state.update_queue, st.session_state.stop_event, 
        seed, initial_config_json, memory_window, memory_format
    )
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    st.session_state.worker_thread = thread
    st.session_state.simulation_running = True


def stop_simulation():
    if st.session_state.stop_event:
        st.session_state.stop_event.set()
    st.session_state.simulation_status = "Parando..."


def main():
    st.set_page_config(page_title="Llama Server Sim", page_icon="🚀", layout="wide")
    init_session_state()
    
    st.title("🚀 Simulador Llama Server (A100)")
    st.markdown("---")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📊 Heatmap")
        heatmap = st.empty()
        if st.session_state.current_image:
            heatmap.image(st.session_state.current_image, use_container_width=True)
        else:
            heatmap.info("Aguardando...")
            
        st.subheader("📝 Log")
        log_text = "\n".join(st.session_state.log_messages[-20:]) if st.session_state.log_messages else ""
        st.text_area("Status", value=log_text, height=200, disabled=True)
    
    with col_right:
        st.subheader("⚙️ Configurações")
        
        # --- SELEÇÃO DE SERVIDOR ---
        selected_server = st.selectbox("Servidor", list(SERVER_PRESETS.keys()), index=1)
        if selected_server == "✏️ Customizado":
            base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL)
        else:
            base_url = SERVER_PRESETS[selected_server]
            st.code(base_url, language="text")
        
        model = st.selectbox("Modelo", AVAILABLE_MODELS, index=0)
        
        n_agents = st.number_input("Agentes", 3, 100, 10)
        n_rounds = st.number_input("Rodadas", 2, 1000, 10)
        n_neighbors = st.number_input("Vizinhos", 1, 21, 3, step=2)
        if n_neighbors % 2 == 0: n_neighbors -= 1
        
        temp = st.slider("Temperatura", 0.0, 2.0, 0.0, 0.1)
        
        # Variants
        yaml_path = get_yaml_path()
        variants = [v for v in load_yaml_variants(yaml_path) if v.startswith(("v9_", "v21_"))]
        variant = st.selectbox("Prompt", variants, index=0 if variants else 0)
        
        # Memory
        use_mem = st.checkbox("Memória (W)", False)
        if use_mem:
            max_mem = max(1, int(n_rounds) - 1)
            mem_w = st.number_input("W (Turnos)", 1, max_mem, min(1, max_mem))
            mem_fmt = "timeline"
        else:
            mem_w = 0
            mem_fmt = "timeline"
        
        # Seed & Init
        init_maj = st.slider("Maioria Inicial %", 0, 100, 50)
        use_seed = st.checkbox("Seed Fixo", True)
        seed = st.number_input("Valor Seed", 1, 999999, 42) if use_seed else None
        
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        if c1.button("🔍 Testar"):
            res = test_model_connection(base_url, model)
            st.session_state.last_test_result = res
            st.session_state.connection_status = "OK" if res['success'] else "Erro"
            
        if c2.button("▶️ Rodar", disabled=st.session_state.simulation_running):
            start_simulation(n_agents, n_rounds, n_neighbors, temp, variant, base_url, model, 0, init_maj, seed, None, mem_w, mem_fmt)
            st.rerun()
            
        if c3.button("⏹️ Parar", disabled=not st.session_state.simulation_running):
            stop_simulation()
            st.rerun()
            
        st.markdown(f"Status: **{st.session_state.connection_status}**")
        
        if st.session_state.last_test_result:
            r = st.session_state.last_test_result
            if r['success']:
                st.success(f"{r['latency_ms']:.0f}ms | Resp: {r['token']}")
            else:
                st.error(r['error'])

    if st.session_state.simulation_running:
        process_queue_updates()
        time.sleep(0.2)
        st.rerun()

if __name__ == "__main__":
    main()
