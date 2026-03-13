#!/usr/bin/env python3
"""
Streamlit App - Simulação de Conformidade com LLM
Interface minimalista para simulação com heatmap PRETO/BRANCO que atualiza célula-a-célula.
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
DEFAULT_MODEL = "google/gemma-3-4b"

SERVER_PRESETS = {
    "LM Studio (Mac Studio)": "http://172.18.254.18:1234/v1",
    "LM Studio (Linux Server 2 RTX Pro 6000)": "http://172.18.254.17:1234/v1",
    "🚀 Llama Server (A100 - 8081)": "http://172.18.254.16:8081/v1",
    "✏️ Customizado": "custom",
}

AVAILABLE_MODELS = [
    "google/gemma-3-4b",
    "google/gemma-3-12b",
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
    match = re.search(r'\[\s*(k|z|a|b|0|1|p|q|α|β|△|○|⊕|⊖|ł|þ|yes|no)\s*\]', response_clean)
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
        """
        Verifica se a conformidade foi atingida (todos os agentes têm a mesma opinião).
        
        Args:
            round_idx: Índice da rodada a verificar
            
        Returns:
            True se todos os agentes têm a mesma opinião, False caso contrário
        """
        round_states = self.states[round_idx]
        
        # Ignora células NaN
        valid_states = round_states[~np.isnan(round_states)]
        
        if len(valid_states) == 0:
            return False
        
        # Verifica se todos os valores são iguais
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
        """Retorna índices dos vizinhos (esquerda, direita) para topologia circular."""
        half = self.n_neighbors // 2
        left_indices = [((agent_idx - i) % self.n_agents) for i in range(1, half + 1)]
        right_indices = [((agent_idx + i) % self.n_agents) for i in range(1, half + 1)]
        return left_indices[::-1], right_indices  # left em ordem reversa
    
    def _state_to_token(self, state_val: float) -> str:
        """Converte valor de estado numérico para token de opinião."""
        if np.isnan(state_val):
            return "?"
        return self.opinion_pair[int(state_val)]
    
    def _get_neighbors(self, agent_idx: int, round_states: np.ndarray) -> Tuple[List[str], List[str]]:
        """
        Obtém vizinhos à esquerda e direita de um agente (topologia circular).
        """
        half = self.n_neighbors // 2
        left = []
        right = []
        
        # Vizinhos à esquerda (sentido anti-horário)
        for i in range(1, half + 1):
            neighbor_idx = (agent_idx - i) % self.n_agents
            val = round_states[neighbor_idx]
            if not np.isnan(val):
                left.insert(0, self.opinion_pair[int(val)])
        
        # Vizinhos à direita (sentido horário)
        for i in range(1, half + 1):
            neighbor_idx = (agent_idx + i) % self.n_agents
            val = round_states[neighbor_idx]
            if not np.isnan(val):
                right.append(self.opinion_pair[int(val)])
        
        return left, right
    
    def _format_memory_block(self, agent_idx: int, current_round: int) -> str:
        """
        Constrói bloco de memória com W rodadas anteriores (self + vizinhos).
        Retorna string vazia se W=0 ou não há histórico suficiente.
        """
        if self.memory_window == 0 or current_round == 0:
            return ""
        
        # Determinar intervalo de rodadas: [current_round - W, current_round - 1]
        start_round = max(0, current_round - self.memory_window)
        end_round = current_round  # exclusive
        
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
    
    def _query_llm_native(self, system_prompt: str, user_prompt: str) -> str:
        """
        Executa chamada via API nativa do LM Studio (igual ao execucao_simultanea.py).
        Endpoint: /api/v1/chat
        Payload: { "model": ..., "input": "SYSTEM:...\nUSER:...", "temperature": ... }
        """
        # Remove /v1 do final se existir para pegar a raiz, depois adiciona /api/v1/chat
        base_url_root = self.base_url.rstrip("/")
        if base_url_root.endswith("/v1"):
            base_url_root = base_url_root[:-3]
            
        url = f"{base_url_root}/api/v1/chat"
        
        # Monta input combinado exatamente como no script original
        # Note: escaping newlines might be tricky if not careful, but f-string handles it literally
        combined_input = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
        
        payload = {
            "model": self.model,
            "input": combined_input,
            "temperature": self.temperature
        }

        # Adiciona seed se configurado para determinismo
        if self.seed is not None:
            payload["seed"] = self.seed
        
        # DEBUG
        print(f"DEBUG_NATIVE: URL={url} | Model={self.model}")
        
        try:
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            # Lógica de extração idêntica ao script
            content_text = ""
            if isinstance(data.get("output"), list):
                for item in data["output"]:
                    if isinstance(item, dict) and isinstance(item.get("content"), str):
                        content_text = item.get("content", "")
                        break
            if not content_text:
                content_text = data.get("content", "") if isinstance(data.get("content"), str) else ""
                
            return content_text
            
        except Exception as e:
            print(f"ERROR_NATIVE: {e}")
            raise e

    def _query_llm(self, system_prompt: str, user_prompt: str, agent_idx: int, round_idx: int) -> Tuple[str, Optional[str]]:
        """
        Faz chamada ao LLM e retorna (resposta_raw, token_extraído).
        Retenta até 5 vezes em caso de formato inválido.
        Registra prompt completo e resposta no log.
        """
        max_attempts = 5
        
        for attempt in range(max_attempts):
            if self._should_stop():
                return ("", None)
            
            try:
                # DEBUG: Print actual parameters being sent
                # print(f"DEBUG_CALL: Model={self.model} | Temp={self.temperature} | MaxTokens=512 | Seed=42")
                
                # --- ANTIGO (OpenAI Client) ---
                # response = self.client.chat.completions.create(
                #     model=self.model,
                #     messages=[
                #         {"role": "system", "content": system_prompt},
                #         {"role": "user", "content": user_prompt}
                #     ],
                #     temperature=self.temperature,
                #     seed=42
                # )
                # raw_response = response.choices[0].message.content.strip()
                
                # --- NOVO (Native HTTP) ---
                # raw_response = self._query_llm_native(system_prompt, user_prompt).strip()
                
                # metodo para forcar o determinismo (HTTP RAW)
                # Garante URL correta para endpoint OpenAI-compatible
                base_url_clean = self.base_url.rstrip("/")
                if not base_url_clean.endswith("/v1"):
                    base_url_clean += "/v1"
                url = f"{base_url_clean}/chat/completions"

                top_k_value = 40 if self.temperature and self.temperature > 0 else 1
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 50,
                    "top_p": 1,
                    "top_k": top_k_value,
                    "repeat_penalty": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
                
                if self.seed is not None:
                    payload["seed"] = self.seed
                
                try:
                    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
                    response.raise_for_status()
                    data = response.json()
                    raw_response = data['choices'][0]['message']['content'].strip()
                except Exception as e:
                    self._log(f"❌ Erro HTTP RAW: {e}")
                    raise e
                
                token = parse_llm_response(raw_response)
                
                # Registra no log (TXT formatado)
                if token is not None:
                    curr_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    separator = "═" * 80
                    sub_separator = "-" * 80
                    
                    log_entry = (
                        f"{separator}\n"
                        f"TIMESTAMP: {curr_time} | ROUND: {round_idx} | AGENT: {agent_idx} | ATTEMPT: {attempt + 1}\n"
                        f"{separator}\n\n"
                        f"{system_prompt}\n\n"
                        f"{user_prompt}\n\n"
                        f">>> RESPONSE (RAW)\n"
                        f"{sub_separator}\n"
                        f"{raw_response}\n"
                        f"{sub_separator}\n\n"
                        f">>> RESULT: {token}\n\n"
                    )
                    
                    # Salva imediatamente no arquivo TXT
                    with open(self.log_filepath, 'a', encoding='utf-8') as f:
                        f.write(log_entry)
                    return (raw_response, token)
                
                self._log(f"⚠️ Tentativa {attempt + 1}/{max_attempts}: formato inválido")
                
            except Exception as e:
                self._log(f"❌ Erro LLM (tentativa {attempt + 1}): {str(e)[:50]}")
                time.sleep(0.5)
        
        return ("", None)
    
    def run(self):
        """Executa a simulação completa."""
        try:
            self._log("🚀 Iniciando simulação...")
            self._log(f"💾 Log incremental: {self.log_filepath}")
            
            # Configurar seeds se fornecido
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
                self._log(f"🎲 Seed configurado: {self.seed}")
            
            # Inicializa cliente OpenAI
            self.client = OpenAI(base_url=self.base_url, api_key=DEFAULT_API_KEY)
            
            # Carrega estratégia de prompt
            yaml_path = get_yaml_path()
            if get_prompt_strategy is not None:
                try:
                    self.strategy = get_prompt_strategy(self.prompt_variant, yaml_path)
                except Exception as e:
                    self._log(f"⚠️ Erro ao carregar estratégia: {e}")
                    self.update_queue.put({'type': 'error', 'message': str(e)})
                    return
            else:
                self._log("⚠️ prompt_strategies.py não disponível")
                self.update_queue.put({'type': 'error', 'message': 'prompt_strategies não disponível'})
                return
            
            # Inicializa rodada 0
            if self.initial_config_json is not None:
                # Carrega distribuição do JSON
                self._log(f"📂 Carregando distribuição do arquivo (sim #{self.initial_config_json['sim_number']})...")
                initial_opinions = np.array(self.initial_config_json['initial_config'], dtype=int)
                
                if len(initial_opinions) != self.n_agents:
                    self._log(f"⚠️ AVISO: JSON tem {len(initial_opinions)} agentes, ajustando para {self.n_agents}")
                    # Trunca ou repete para ajustar ao tamanho
                    if len(initial_opinions) > self.n_agents:
                        initial_opinions = initial_opinions[:self.n_agents]
                    else:
                        repeat_times = self.n_agents // len(initial_opinions) + 1
                        initial_opinions = np.tile(initial_opinions, repeat_times)[:self.n_agents]
                
                n_opinion_0 = int(np.sum(initial_opinions == 0))
                n_opinion_1 = self.n_agents - n_opinion_0
                self._log(f"   → Distribuição carregada: {n_opinion_0} x '{self.opinion_pair[0]}', {n_opinion_1} x '{self.opinion_pair[1]}'")
            else:
                # Gera nova distribuição (mesmo método do run_automaton_numba.py)
                self._log(f"📊 Gerando nova distribuição ({self.initial_majority}% maioria inicial)...")
                
                # Converte porcentagem para ratio (igual ao run_automaton_numba.py)
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
                n_opinion_0 = int(dist.count_0)
                n_opinion_1 = int(dist.count_1)
                self._log(
                    f"   → {n_opinion_0} agentes com '{self.opinion_pair[0]}', {n_opinion_1} com '{self.opinion_pair[1]}' "
                    f"(mode={distribution_mode}, seed_dist={seed_distribution})"
                )
            
            # Aplicar distribuição TODA DE UMA VEZ (sem delay)
            for i in range(self.n_agents):
                self.states[0, i] = initial_opinions[i]
            
            # Atualizar heatmap UMA ÚNICA VEZ para rodada 0 completa
            self._update_heatmap(0, self.n_agents - 1)  # Mostra rodada 0 completa
            self._log("✅ Rodada 0 inicializada")
            
            # Executa rodadas 1 a n_rounds-1
            for r in range(1, self.n_rounds):
                if self._should_stop():
                    self._log("🛑 Simulação interrompida pelo usuário")
                    self.update_queue.put({'type': 'stopped'})
                    return
                
                self._log(f"🔄 Rodada {r}/{self.n_rounds - 1}...")
                prev_states = self.states[r - 1].copy()
                
                for i in range(self.n_agents):
                    if self._should_stop():
                        self._log("🛑 Simulação interrompida pelo usuário")
                        self.update_queue.put({'type': 'stopped'})
                        return
                    
                    # Obtém vizinhos da rodada anterior
                    left, right = self._get_neighbors(i, prev_states)
                    
                    # Opinião atual do agente (da rodada anterior)
                    current_val = prev_states[i]
                    current_opinion = self.opinion_pair[int(current_val)] if not np.isnan(current_val) else get_default_opinion(self.prompt_variant)
                    
                    # Constrói prompt base
                    try:
                        system_prompt, user_prompt = self.strategy.build_prompt(
                            left=left,
                            right=right,
                            current_opinion=current_opinion
                        )
                    except Exception as e:
                        self._log(f"⚠️ Erro no prompt: {e}")
                        self.states[r, i] = np.nan
                        self._update_heatmap(r, i)
                        continue
                    
                    # Injeta memória no início do user_prompt, se W > 0
                    memory_block = self._format_memory_block(i, r)
                    if memory_block:
                        user_prompt = memory_block + "\n\n" + user_prompt
                    
                    # Consulta LLM
                    raw_response, token = self._query_llm(system_prompt, user_prompt, i, r)
                    
                    if self._should_stop():
                        self._log("🛑 Simulação interrompida pelo usuário")
                        self.update_queue.put({'type': 'stopped'})
                        return
                    
                    # Atualiza estado
                    if token is not None and token in OPINION_MAP:
                        self.states[r, i] = OPINION_MAP[token]
                        self._log(f"✅ Agente {i}: [{token}]")
                    else:
                        self.states[r, i] = np.nan
                        self._log(f"⚠️ Agente {i}: resposta inválida")
                    
                    # Atualiza heatmap (o delay vem naturalmente da latência do LLM)
                    self._update_heatmap(r, i)
                
                # Verifica se a conformidade foi atingida após completar a rodada
                if self._check_conformity(r):
                    opinion = self._get_conformity_opinion(r)
                    self._log(f"🎯 CONFORMIDADE ATINGIDA na rodada {r}!")
                    self._log(f"   → Todos os agentes escolheram [{opinion}]")
                    self.update_queue.put({'type': 'conformity', 'round': r, 'opinion': opinion})
                    return
            
            self._log("✅ Simulação concluída (limite de rodadas)!")
            self._log(f"💾 Log salvo: {self.log_filepath}")
            self.update_queue.put({'type': 'completed'})
            
        except Exception as e:
            error_msg = f"❌ Erro fatal: {str(e)}"
            self._log(error_msg)
            
            # Salva erro em arquivo
            try:
                error_file = self.log_filepath.replace('.jsonl', '_ERROR.txt')
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"ERRO NA SIMULAÇÃO\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Erro: {str(e)}\n")
                    f.write(f"Tipo: {type(e).__name__}\n")
                    import traceback
                    f.write(f"\nTraceback completo:\n{traceback.format_exc()}")
                self._log(f"💾 Erro salvo: {error_file}")
            except:
                pass
            
            self.update_queue.put({'type': 'error', 'message': str(e)})


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

def init_session_state():
    """Inicializa variáveis de sessão do Streamlit."""
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
    """Processa atualizações da fila do worker."""
    if st.session_state.update_queue is None:
        return False
    
    has_updates = False
    try:
        while True:
            update = st.session_state.update_queue.get_nowait()
            has_updates = True
            
            if update['type'] == 'heatmap':
                st.session_state.current_image = update['image']
            elif update['type'] == 'log':
                st.session_state.log_messages.append(update['message'])
                # Mantém apenas as últimas 50 mensagens
                if len(st.session_state.log_messages) > 50:
                    st.session_state.log_messages = st.session_state.log_messages[-50:]
            elif update['type'] in ('completed', 'stopped', 'error', 'conformity'):
                st.session_state.simulation_running = False
                if update['type'] == 'completed':
                    st.session_state.simulation_status = "Concluído (limite de rodadas)"
                elif update['type'] == 'conformity':
                    round_num = update.get('round', '?')
                    opinion = update.get('opinion', '?')
                    st.session_state.simulation_status = f"🎯 Conformidade na rodada {round_num} [{opinion}]"
                elif update['type'] == 'stopped':
                    st.session_state.simulation_status = "Parado"
                else:
                    st.session_state.simulation_status = f"Erro: {update.get('message', 'Desconhecido')}"
    except queue.Empty:
        pass
    
    return has_updates


def test_model_connection(base_url: str, model: str) -> Dict[str, Any]:
    """Testa conexão com o modelo LLM."""
    try:
        client = OpenAI(base_url=base_url, api_key=DEFAULT_API_KEY)
        
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say only: [k]"}
            ],
            temperature=0.0,
            max_tokens=50
        )
        latency = (time.time() - start_time) * 1000  # ms
        
        raw_response = response.choices[0].message.content.strip()
        token = parse_llm_response(raw_response)
        
        return {
            'success': True,
            'raw_response': raw_response,
            'token': token,
            'latency_ms': latency
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def start_simulation(
    n_agents: int,
    n_rounds: int,
    n_neighbors: int,
    temperature: float,
    prompt_variant: str,
    base_url: str,
    model: str,
    delay_ms: int,
    initial_majority: int,
    seed: Optional[int] = None,
    initial_config_json: Optional[Dict] = None,
    memory_window: int = 0,
    memory_format: str = "timeline"
):
    """Inicia a simulação em background."""
    # Limpa estado anterior
    st.session_state.log_messages = []
    st.session_state.current_image = None
    st.session_state.simulation_status = "Rodando"
    
    # Cria fila e evento de parada
    st.session_state.update_queue = queue.Queue()
    st.session_state.stop_event = threading.Event()
    
    # Cria e inicia worker
    worker = SimulationWorker(
        n_agents=n_agents,
        n_rounds=n_rounds,
        n_neighbors=n_neighbors,
        temperature=temperature,
        prompt_variant=prompt_variant,
        base_url=base_url,
        model=model,
        delay_ms=delay_ms,
        initial_majority=initial_majority,
        update_queue=st.session_state.update_queue,
        stop_event=st.session_state.stop_event,
        seed=seed,
        initial_config_json=initial_config_json,
        memory_window=memory_window,
        memory_format=memory_format
    )
    
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    
    st.session_state.worker_thread = thread
    st.session_state.simulation_running = True


def stop_simulation():
    """Para a simulação em andamento."""
    if st.session_state.stop_event is not None:
        st.session_state.stop_event.set()
    st.session_state.simulation_status = "Parando..."


def main():
    """Função principal da aplicação Streamlit."""
    st.set_page_config(
        page_title="LLM Conformidade Simulator",
        page_icon="🧠",
        layout="wide"
    )
    
    init_session_state()
    
    # Header
    st.title("🧠 Simulador de Conformidade com LLM")
    st.markdown("---")
    
    # Layout em 2 colunas
    col_left, col_right = st.columns([2, 1])
    
    # ==========================================================================
    # COLUNA ESQUERDA: Heatmap + Log
    # ==========================================================================
    with col_left:
        st.subheader("📊 Heatmap da Simulação")
        
        # Placeholder para o heatmap
        heatmap_placeholder = st.empty()
        
        # Exibe imagem atual ou placeholder
        if st.session_state.current_image is not None:
            heatmap_placeholder.image(
                st.session_state.current_image,
                use_container_width=True
            )
        else:
            heatmap_placeholder.info("⏳ Aguardando início da simulação...")
        
        st.subheader("📝 Log de Execução")
        
        # Área de log
        log_text = "\n".join(st.session_state.log_messages[-20:]) if st.session_state.log_messages else "Nenhuma mensagem ainda..."
        st.text_area(
            "Status",
            value=log_text,
            height=200,
            disabled=True,
            key="log_area"
        )
    
    # ==========================================================================
    # COLUNA DIREITA: Controles
    # ==========================================================================
    with col_right:
        st.subheader("⚙️ Configurações")
        
        # Inputs
        n_agents = st.number_input(
            "Número de Agentes",
            min_value=3,
            max_value=100,
            value=10,
            step=1,
            help="Quantidade de agentes na simulação"
        )
        
        n_rounds = st.number_input(
            "Número de Rodadas",
            min_value=2,
            max_value=1000,
            value=10,
            step=1,
            help="Quantidade de rodadas da simulação"
        )
        
        # Controles de memória (W)
        use_memory = st.checkbox(
            "🧠 Habilitar Memória Local (Rolling Window)",
            value=False,
            help="Permitir que agentes vejam suas decisões anteriores e dos vizinhos"
        )
        
        if use_memory:
            max_memory = max(1, int(n_rounds) - 1)
            memory_window = st.number_input(
                "Memória: turnos anteriores (W)",
                min_value=1,
                max_value=max_memory,
                value=min(3, max_memory),
                step=1,
                help="Número de rodadas anteriores para incluir no prompt"
            )
            memory_format = "timeline"
            st.caption(f"💭 Memória ativa: últimas {memory_window} rodada(s) (self + vizinhos)")
        else:
            memory_window = 0
            memory_format = "timeline"
            st.caption("💭 Memória desativada")
        
        n_neighbors = st.number_input(
            "Número de Vizinhos",
            min_value=1,
            max_value=min(n_agents - 1, 21),
            value=3,
            step=2,
            help="Deve ser ímpar e menor que o número de agentes"
        )
        
        # Validação: n_neighbors deve ser ímpar
        if n_neighbors % 2 == 0:
            st.warning("⚠️ Número de vizinhos deve ser ímpar! Ajustando...")
            n_neighbors = n_neighbors + 1 if n_neighbors + 1 < n_agents else n_neighbors - 1
        
        temperature = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Temperatura do modelo LLM"
        )
        
        # Carrega variantes do YAML e filtra apenas v9 e v21
        yaml_path = get_yaml_path()
        available_variants = load_yaml_variants(yaml_path)
        filtered_variants = [v for v in available_variants if v.startswith("v9_") or v.startswith("v21_")]
        
        def _token_suffix(variant: str) -> str:
            token_suffixes = [
                "_ab", "_01", "_αβ", "_△○", "_⊕⊖", "_pq", "_łþ", "_yesno"
            ]
            for suffix in token_suffixes:
                if suffix in variant:
                    if suffix == "_yesno":
                        return "noyes"
                    return suffix[1:]  # remove leading underscore
            return "kz"
        
        def _display_name(variant: str) -> str:
            if variant.startswith("v9_"):
                base = "token"
            elif variant.startswith("v21_"):
                base = "cot"
            else:
                base = variant
            token_suffix = _token_suffix(variant)
            return f"{base} ({token_suffix})"
        
        display_to_variant = { _display_name(v): v for v in filtered_variants }
        display_options = list(display_to_variant.keys())
        
        default_variant = "v9_lista_completa_meio"
        default_display = _display_name(default_variant) if default_variant in filtered_variants else (display_options[0] if display_options else "")
        
        selected_display = st.selectbox(
            "Variante de Prompt",
            options=display_options,
            index=display_options.index(default_display) if default_display in display_options else 0,
            help="Estratégia de prompt a ser utilizada"
        )
        prompt_variant = display_to_variant.get(selected_display, default_variant)
        
        # Delay removido - rodada 0 aparece instantaneamente
        delay_ms = 0
        
        initial_majority = st.slider(
            "Maioria Inicial (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            help="Porcentagem de agentes que começam com a primeira opinião (k/a/0) na Rodada 0"
        )
        
        # Mostra preview da distribuição inicial (maioria real quando > 50%)
        ratio_preview = initial_majority / 100.0
        if ratio_preview > 0.5:
            n_opinion_0 = int(np.ceil(n_agents * ratio_preview))
            min_majority = n_agents // 2 + 1
            if n_opinion_0 < min_majority:
                n_opinion_0 = min_majority
        else:
            n_opinion_0 = int(np.floor(n_agents * ratio_preview))
        n_opinion_0 = max(0, min(n_agents, n_opinion_0))
        n_opinion_1 = n_agents - n_opinion_0
        st.caption(f"📊 Rodada 0: {n_opinion_0} agentes com opinião primária, {n_opinion_1} com secundária")
        
        st.markdown("---")
        st.subheader("🎲 Controle de Reprodutibilidade")
        
        # Seed
        use_seed = st.checkbox("Usar seed fixo", value=False)
        seed_value = None
        if use_seed:
            seed_value = st.number_input(
                "Seed",
                min_value=1,
                max_value=999999,
                value=42,
                step=1,
                help="Seed para garantir mesma distribuição inicial"
            )
            st.caption(f"🎲 Seed: {seed_value}")
        
        # Upload de configuração inicial
        uploaded_file = st.file_uploader(
            "Carregar distribuição inicial (JSON)",
            type=['json'],
            help="Arquivo JSON exportado do run_automaton_numba.py"
        )
        
        initial_config_data = None
        if uploaded_file is not None:
            try:
                initial_config_data = json.loads(uploaded_file.read())
                
                # Validação
                if 'initial_config' not in initial_config_data:
                    st.error("❌ JSON inválido: falta 'initial_config'")
                    initial_config_data = None
                else:
                    st.success(f"✅ Distribuição carregada: sim #{initial_config_data.get('sim_number', '?')}")
                    st.caption(f"Agentes: {len(initial_config_data['initial_config'])}")
                    st.caption(f"Seed original: {initial_config_data.get('seed', 'N/A')}")
                    
                    # Visualizar distribuição
                    config_array = np.array(initial_config_data['initial_config'])
                    n_0 = int(np.sum(config_array == 0))
                    n_1 = len(config_array) - n_0
                    st.caption(f"Distribuição: {n_0} x '0' ({n_0/len(config_array)*100:.1f}%), {n_1} x '1' ({n_1/len(config_array)*100:.1f}%)")
                    
            except Exception as e:
                st.error(f"❌ Erro ao ler JSON: {str(e)}")
                initial_config_data = None
        
        st.markdown("---")
        st.subheader("🔌 Conexão LLM")

        selected_server = st.selectbox("Servidor", list(SERVER_PRESETS.keys()), index=0)
        if selected_server == "✏️ Customizado":
            base_url = st.text_input(
                "Base URL (opcional)",
                value=DEFAULT_BASE_URL,
                help="URL do servidor LM Studio"
            )
        else:
            base_url = SERVER_PRESETS[selected_server]
            st.code(base_url, language="text")
        
        model = st.selectbox(
            "Modelo",
            options=AVAILABLE_MODELS,
            index=0,
            help="Modelo LLM a ser utilizado"
        )
        
        st.markdown("---")
        
        # Botões
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🔍 Testar", use_container_width=True, disabled=st.session_state.simulation_running):
                with st.spinner("Testando conexão..."):
                    result = test_model_connection(base_url, model)
                    st.session_state.last_test_result = result
                    if result['success']:
                        st.session_state.connection_status = "Conectado ✅"
                    else:
                        st.session_state.connection_status = "Erro ❌"
        
        with col_btn2:
            if st.button("▶️ Rodar", use_container_width=True, disabled=st.session_state.simulation_running):
                start_simulation(
                    n_agents=n_agents,
                    n_rounds=n_rounds,
                    n_neighbors=n_neighbors,
                    temperature=temperature,
                    prompt_variant=prompt_variant,
                    base_url=base_url,
                    model=model,
                    delay_ms=delay_ms,
                    initial_majority=initial_majority,
                    seed=seed_value,
                    initial_config_json=initial_config_data,
                    memory_window=memory_window,
                    memory_format=memory_format
                )
                st.rerun()
        
        with col_btn3:
            if st.button("⏹️ Parar", use_container_width=True, disabled=not st.session_state.simulation_running):
                stop_simulation()
                st.rerun()
        
        st.markdown("---")
        st.subheader("📈 Status")
        
        # Status de conexão
        status_color = "green" if "Conectado" in st.session_state.connection_status else "gray"
        st.markdown(f"**Conexão:** :{status_color}[{st.session_state.connection_status}]")
        
        # Status da simulação
        sim_status = st.session_state.simulation_status
        if sim_status == "Rodando":
            st.markdown(f"**Simulação:** :blue[{sim_status}] 🔄")
        elif sim_status == "Concluído":
            st.markdown(f"**Simulação:** :green[{sim_status}] ✅")
        elif "Erro" in sim_status:
            st.markdown(f"**Simulação:** :red[{sim_status}]")
        else:
            st.markdown(f"**Simulação:** :gray[{sim_status}]")
        
        # Resultado do teste
        if st.session_state.last_test_result is not None:
            result = st.session_state.last_test_result
            st.markdown("---")
            st.subheader("🧪 Resultado do Teste")
            if result['success']:
                st.success(f"Latência: {result['latency_ms']:.1f}ms")
                st.text(f"Resposta: {result['raw_response'][:100]}")
                st.text(f"Token extraído: {result['token']}")
            else:
                st.error(f"Erro: {result['error']}")
    
    # ==========================================================================
    # AUTO-REFRESH durante simulação
    # ==========================================================================
    if st.session_state.simulation_running:
        # Processa atualizações da fila
        process_queue_updates()
        
        # Auto-refresh a cada 200ms
        time.sleep(0.2)
        st.rerun()


if __name__ == "__main__":
    main()
