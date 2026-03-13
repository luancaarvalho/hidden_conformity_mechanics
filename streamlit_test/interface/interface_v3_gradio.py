#!/usr/bin/env python3
"""
Gradio App - Simulação de Conformidade com LLM

Mudanças aplicadas nesta versão (v3):
- Infra de UI migrada de Streamlit para Gradio Blocks.
- Loop de polling/rerun removido; updates via generator `run_simulation_stream`.
- Controle de parada via botão Stop com `cancels` do Gradio + stop event.

Intencionalmente NÃO alterado:
- Lógica de experimento (prompts, parsing, sementes, memória, critérios de parada).
- Formato/chamada do LLM e gravação de logs de prompt/resposta.
- Defaults centrais (modelos, base URL padrão, ranges e comportamento de tokens/COT).
"""

import os
import sys
import re
import time
import random
import threading
import json
from io import BytesIO
from typing import Optional, List, Tuple, Dict, Any

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import yaml
from PIL import Image
from openai import OpenAI

def _find_repo_root() -> str:
    """
    Resolve repo root regardless of whether this file is executed from the project
    root (`streamlit_test/`) or from a subfolder (`streamlit_test/interface/`).
    """
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(d, "prompt_strategies.py")) and os.path.exists(os.path.join(d, "prompt_templates.yaml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # Fallback: assume `streamlit_test/interface/<file>.py`
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


REPO_ROOT = _find_repo_root()

# Adiciona o repo root ao path para importar prompt_strategies.py e outros módulos.
sys.path.insert(0, REPO_ROOT)

try:
    from prompt_strategies import get_prompt_strategy, PromptStrategy
except ImportError:
    get_prompt_strategy = None
    PromptStrategy = None

from projeto_final.utils.utils import (
    call_llm_responses,
    generate_initial_distribution_shared,
    parse_opinion_token,
)

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
    "google/gemma-3-4b:2",
    "google/gemma-3-12b",
    "google/gemma-3-12b:2",
    "gemma-3-27b-it",
    "gemma-3-27b-it:2",
    "meta-llama-3.1-8b-instruct",
    "meta-llama-3.1-70b-instruct",
    "qwen3-4b",
    "qwen/qwen3-32b",
]

def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _apply_sampling_overrides(payload: Dict[str, Any]) -> Dict[str, Any]:
    top_k = _env_int("LLM_TOP_K")
    if top_k is not None:
        payload["top_k"] = top_k

    top_p = _env_float("LLM_TOP_P")
    if top_p is not None:
        payload["top_p"] = top_p

    min_p = _env_float("LLM_MIN_P")
    if min_p is not None:
        payload["min_p"] = min_p

    repeat_penalty = _env_float("LLM_REPEAT_PENALTY")
    if repeat_penalty is not None:
        payload["repeat_penalty"] = repeat_penalty

    return payload

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
    # Repo root (onde ficam prompt_strategies.py e prompt_templates.yaml)
    yaml_path = os.path.join(REPO_ROOT, "prompt_templates.yaml")
    if os.path.exists(yaml_path):
        return yaml_path
    # Tenta no diretório atual
    local_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_templates.yaml")
    if os.path.exists(local_yaml):
        return local_yaml
    return yaml_path  # Retorna o esperado mesmo se não existir


CONFORMITY_GAME_MODE_A = "Normative DCT Convergence"
CONFORMITY_GAME_MODE_B = "DCT Consensus Protocol"

CONFORMITY_GAME_SYSTEM_PROMPT_MODE_A = (
    "You are playing a Conformity Game inside a small social group.\n\n"
    "The episode has a correct consensus target:\n"
    "the symbol that was more prevalent in the population at the beginning.\n"
    "Agreement on the correct target improves coordination and reduces friction.\n\n"
    "You only see local snapshots and an optional rolling memory. Your objective is to infer the correct target\n"
    "from repeated local evidence and align your public output to accelerate stable convergence.\n\n"
    "IMPORTANT OUTPUT RULE:\n"
    "Reply with ONLY one token in square brackets: [{T0}] or [{T1}]. No extra text."
)

CONFORMITY_GAME_SYSTEM_PROMPT_MODE_A_NO_MEMORY = (
    "You are playing a Conformity Game inside a small social group.\n\n"
    "The episode has a correct consensus target:\n"
    "the symbol that was more prevalent in the population at the beginning.\n"
    "Agreement on the correct target improves coordination and reduces friction.\n\n"
    "You only see local snapshots. Your objective is to infer the correct target\n"
    "from repeated local evidence and align your public output to accelerate stable convergence.\n\n"
    "IMPORTANT OUTPUT RULE:\n"
    "Reply with ONLY one token in square brackets: [{T0}] or [{T1}]. No extra text."
)

CONFORMITY_GAME_SYSTEM_PROMPT_MODE_B = (
    "You are a node in a local-to-global density-based consensus protocol.\n\n"
    "Each round you must output exactly one symbol: [{T0}] or [{T1}].\n"
    "The correct final consensus is the symbol that was globally more frequent at the beginning of the episode.\n"
    "However you do not see the full initial state. You only receive a local neighborhood snapshot and an optional rolling memory.\n\n"
    "IMPORTANT OUTPUT RULE:\n"
    "Reply with ONLY one token in square brackets: [{T0}] or [{T1}]. No extra text."
)

CONFORMITY_GAME_SYSTEM_PROMPT_MODE_B_NO_MEMORY = (
    "You are a node in a local-to-global density-based consensus protocol.\n\n"
    "Each round you must output exactly one symbol: [{T0}] or [{T1}].\n"
    "The correct final consensus is the symbol that was globally more frequent at the beginning of the episode.\n"
    "However you do not see the full initial state. You only receive a local neighborhood snapshot.\n\n"
    "IMPORTANT OUTPUT RULE:\n"
    "Reply with ONLY one token in square brackets: [{T0}] or [{T1}]. No extra text."
)

CONFORMITY_GAME_USER_TEMPLATE_WITH_MEMORY = (
    "Below you can see the current complete list of opinions from your shown neighborhood (including your own).\n\n"
    "**Complete Opinion List:** {NEIGHBORHOOD_LIST}\n"
    "**Your Position:** You are the opinion in the middle\n"
    "**Your Current Opinion:** '{CURRENT_OPINION}'\n\n"
    "Reply ONLY with [{T0}] or [{T1}]."
)

CONFORMITY_GAME_USER_TEMPLATE_NO_MEMORY = (
    "Below you can see the current complete list of opinions from your shown neighborhood (including your own).\n\n"
    "**Complete Opinion List:** {NEIGHBORHOOD_LIST}\n"
    "**Your Position:** You are the opinion in the middle\n"
    "**Your Current Opinion:** '{CURRENT_OPINION}'\n\n"
    "Reply ONLY with [{T0}] or [{T1}]."
)


def render_conformity_game_system_prompt(mode: str, token0: str, token1: str, memory_enabled: bool) -> str:
    if mode == CONFORMITY_GAME_MODE_B:
        tpl = CONFORMITY_GAME_SYSTEM_PROMPT_MODE_B if memory_enabled else CONFORMITY_GAME_SYSTEM_PROMPT_MODE_B_NO_MEMORY
    else:
        tpl = CONFORMITY_GAME_SYSTEM_PROMPT_MODE_A if memory_enabled else CONFORMITY_GAME_SYSTEM_PROMPT_MODE_A_NO_MEMORY
    return tpl.format(T0=token0, T1=token1)


def render_conformity_game_user_prompt(
    neighborhood_list: List[str],
    current_opinion: str,
    token0: str,
    token1: str,
    memory_enabled: bool,
) -> str:
    tpl = CONFORMITY_GAME_USER_TEMPLATE_WITH_MEMORY if memory_enabled else CONFORMITY_GAME_USER_TEMPLATE_NO_MEMORY
    return tpl.format(
        NEIGHBORHOOD_LIST=repr(neighborhood_list),
        CURRENT_OPINION=current_opinion,
        T0=token0,
        T1=token1,
    )


def parse_llm_response(response_text: str) -> Optional[str]:
    """
    Extrai a opinião da resposta do LLM.
    Aceita todos os tokens: k/z, a/b, 0/1, p/q, α/β, △/○, ⊕/⊖, ł/þ, yes/no
    Aceita respostas com espaços: [k], [ k ], etc.
    Para CoT, prioriza o último token válido entre colchetes.
    """
    allowed_tokens = (
        "k", "z",
        "a", "b",
        "0", "1",
        "p", "q",
        "α", "β",
        "△", "○",
        "⊕", "⊖",
        "ł", "þ",
        "no", "yes",
    )
    return parse_opinion_token(
        response_text,
        allowed_tokens=allowed_tokens,
        prefer_last=True,
    )


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


def is_cot_variant(variant: str) -> bool:
    """
    Detecta variantes CoT.
    Neste projeto, as variantes CoT em uso na interface começam com "v21_".
    """
    v = (variant or "").strip().lower()
    return v.startswith("v21_")


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
# EXECUCAO DA SIMULACAO (GRADIO STREAMING)
# ==============================================================================

_ACTIVE_STOP_EVENT_LOCK = threading.Lock()
_ACTIVE_STOP_EVENT: Optional[threading.Event] = None


TOKEN_MAP_OPTIONS = {
    "k/z": ("k", "z"),
    "a/b": ("a", "b"),
    "0/1": ("0", "1"),
    "p/q": ("p", "q"),
    "α/β": ("α", "β"),
    "△/○": ("△", "○"),
    "⊕/⊖": ("⊕", "⊖"),
    "ł/þ": ("ł", "þ"),
    "no(0)/yes(1)": ("no", "yes"),
}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _running_controls():
    return (
        gr.update(interactive=False),  # test_btn
        gr.update(interactive=False),  # run_btn
        gr.update(interactive=True),   # stop_btn
    )


def _idle_controls():
    return (
        gr.update(interactive=True),   # test_btn
        gr.update(interactive=True),   # run_btn
        gr.update(interactive=False),  # stop_btn
    )


def _set_active_stop_event(stop_event: Optional[threading.Event]) -> None:
    global _ACTIVE_STOP_EVENT
    with _ACTIVE_STOP_EVENT_LOCK:
        _ACTIVE_STOP_EVENT = stop_event


def _get_active_stop_event() -> Optional[threading.Event]:
    with _ACTIVE_STOP_EVENT_LOCK:
        return _ACTIVE_STOP_EVENT


def _clear_active_stop_event(stop_event: threading.Event) -> None:
    global _ACTIVE_STOP_EVENT
    with _ACTIVE_STOP_EVENT_LOCK:
        if _ACTIVE_STOP_EVENT is stop_event:
            _ACTIVE_STOP_EVENT = None


class SimulationRunner:
    """Executa a simulação com lógica idêntica ao worker original, emitindo updates por yield."""

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
        stop_event: threading.Event,
        seed: Optional[int] = None,
        initial_config_json: Optional[Dict] = None,
        memory_window: int = 0,
        memory_format: str = "timeline",
        jogo_conformidade: bool = False,
        jogo_conformidade_modo: str = CONFORMITY_GAME_MODE_A,
        token0: str = "k",
        token1: str = "z",
    ):
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.n_neighbors = n_neighbors
        self.temperature = temperature
        self.prompt_variant = prompt_variant
        self.base_url = base_url
        self.model = model
        self.delay_ms = delay_ms
        self.initial_majority = initial_majority
        self.stop_event = stop_event
        self.seed = seed
        self.initial_config_json = initial_config_json
        self.memory_window = memory_window
        self.memory_format = "timeline"
        self.jogo_conformidade = jogo_conformidade
        self.jogo_conformidade_modo = jogo_conformidade_modo
        self.max_output_tokens = 3000 if is_cot_variant(self.prompt_variant) else 50

        self.states = np.full((n_rounds, n_agents), np.nan)
        self.client = None
        self.strategy = None
        if self.jogo_conformidade:
            self.opinion_pair = (token0, token1)
        else:
            self.opinion_pair = get_opinion_pair(prompt_variant)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"prompt_log_{self.prompt_variant}_W{self.memory_window}_{timestamp}.txt"

        base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        save_dir = os.path.join(base_dir, "logs")
        os.makedirs(save_dir, exist_ok=True)
        self.log_filepath = os.path.join(save_dir, log_filename)

        self.log_messages: List[str] = []
        self.current_image: Optional[bytes] = None

    def _log(self, message: str):
        self.log_messages.append(message)
        if len(self.log_messages) > 50:
            self.log_messages = self.log_messages[-50:]

    def _update_heatmap(self, round_idx: int, agent_idx: int):
        self.current_image = generate_heatmap_png(
            self.states,
            self.n_agents,
            self.n_rounds,
            round_idx,
            agent_idx,
        )

    def _should_stop(self) -> bool:
        return self.stop_event.is_set()

    def _check_conformity(self, round_idx: int) -> bool:
        round_states = self.states[round_idx]
        valid_states = round_states[~np.isnan(round_states)]
        if len(valid_states) == 0:
            return False
        return np.all(valid_states == valid_states[0])

    def _get_conformity_opinion(self, round_idx: int) -> str:
        round_states = self.states[round_idx]
        valid_states = round_states[~np.isnan(round_states)]
        if len(valid_states) > 0:
            opinion_value = int(valid_states[0])
            return self.opinion_pair[opinion_value]
        return "?"

    def _get_neighbor_indices(self, agent_idx: int) -> Tuple[List[int], List[int]]:
        half = self.n_neighbors // 2
        left_indices = [((agent_idx - i) % self.n_agents) for i in range(1, half + 1)]
        right_indices = [((agent_idx + i) % self.n_agents) for i in range(1, half + 1)]
        return left_indices[::-1], right_indices

    def _state_to_token(self, state_val: float) -> str:
        if np.isnan(state_val):
            return "?"
        return self.opinion_pair[int(state_val)]

    def _get_neighbors(self, agent_idx: int, round_states: np.ndarray) -> Tuple[List[str], List[str]]:
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

    def _query_llm_native(self, system_prompt: str, user_prompt: str) -> str:
        sampling_overrides = _apply_sampling_overrides({})
        result = call_llm_responses(
            base_url=self.base_url,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            seed=42,
            max_output_tokens=self.max_output_tokens,
            timeout_s=120,
            top_k=sampling_overrides.get("top_k"),
            top_p=sampling_overrides.get("top_p"),
            min_p=sampling_overrides.get("min_p"),
            repeat_penalty=sampling_overrides.get("repeat_penalty"),
        )
        return str(result["raw_response"])

    def _query_llm(self, system_prompt: str, user_prompt: str, agent_idx: int, round_idx: int) -> Tuple[str, Optional[str]]:
        max_attempts = 5
        last_raw_response = ""

        for attempt in range(max_attempts):
            if self._should_stop():
                return ("", None)

            try:
                sampling_overrides = _apply_sampling_overrides({})
                try:
                    result = call_llm_responses(
                        base_url=self.base_url,
                        model=self.model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=self.temperature,
                        seed=42,
                        max_output_tokens=self.max_output_tokens,
                        timeout_s=120,
                        top_k=sampling_overrides.get("top_k"),
                        top_p=sampling_overrides.get("top_p"),
                        min_p=sampling_overrides.get("min_p"),
                        repeat_penalty=sampling_overrides.get("repeat_penalty"),
                    )
                    raw_response = str(result["raw_response"]).strip()
                    last_raw_response = raw_response
                except Exception as e:
                    self._log(f"❌ Erro HTTP RAW: {e}")
                    raise e

                token = parse_llm_response(raw_response)

                curr_time = time.strftime("%Y-%m-%d %H:%M:%S")
                separator = "═" * 80
                sub_separator = "-" * 80
                result_text = token if token is not None else "INVALID_PARSE"
                parse_note = (
                    ""
                    if token is not None
                    else f">>> PARSE_ERROR: expected one token in brackets, got invalid format\n\n"
                )
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
                    f">>> RESULT: {result_text}\n"
                    f"{parse_note}"
                )

                with open(self.log_filepath, 'a', encoding='utf-8') as f:
                    f.write(log_entry)

                if token is not None:
                    return (raw_response, token)

                self._log(f"⚠️ Tentativa {attempt + 1}/{max_attempts}: formato inválido")

            except Exception as e:
                self._log(f"❌ Erro LLM (tentativa {attempt + 1}): {str(e)[:50]}")
                time.sleep(0.5)

        raise RuntimeError(
            f"Falha fatal de parse após {max_attempts} tentativas "
            f"(round={round_idx}, agent={agent_idx}). Última resposta RAW: {last_raw_response!r}"
        )

    def _build_yield_payload(self, status_text: str) -> Tuple[Optional['Image.Image'], str, str]:
        pil_image = None
        if self.current_image is not None:
            pil_image = Image.open(BytesIO(self.current_image)).copy()
        logs_text = "\n".join(self.log_messages[-20:]) if self.log_messages else "Nenhuma mensagem ainda..."
        return pil_image, logs_text, status_text

    def run_stream(self):
        status_text = "Rodando"
        try:
            self._log("🚀 Iniciando simulação...")
            self._log(f"💾 Log incremental: {self.log_filepath}")

            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
                self._log(f"🎲 Seed configurado: {self.seed}")

            self.client = OpenAI(base_url=self.base_url, api_key=DEFAULT_API_KEY)

            if not self.jogo_conformidade:
                yaml_path = get_yaml_path()
                if get_prompt_strategy is not None:
                    try:
                        self.strategy = get_prompt_strategy(self.prompt_variant, yaml_path)
                    except Exception as e:
                        self._log(f"⚠️ Erro ao carregar estratégia: {e}")
                        status_text = f"Erro: {str(e)}"
                        yield self._build_yield_payload(status_text)
                        return
                else:
                    self._log("⚠️ prompt_strategies.py não disponível")
                    status_text = "Erro: prompt_strategies não disponível"
                    yield self._build_yield_payload(status_text)
                    return

            if self.initial_config_json is not None:
                self._log(f"📂 Carregando distribuição do arquivo (sim #{self.initial_config_json['sim_number']})...")
                initial_opinions = np.array(self.initial_config_json['initial_config'], dtype=int)

                if len(initial_opinions) != self.n_agents:
                    self._log(f"⚠️ AVISO: JSON tem {len(initial_opinions)} agentes, ajustando para {self.n_agents}")
                    if len(initial_opinions) > self.n_agents:
                        initial_opinions = initial_opinions[:self.n_agents]
                    else:
                        repeat_times = self.n_agents // len(initial_opinions) + 1
                        initial_opinions = np.tile(initial_opinions, repeat_times)[:self.n_agents]

                n_opinion_0 = int(np.sum(initial_opinions == 0))
                n_opinion_1 = self.n_agents - n_opinion_0
                self._log(f"   → Distribuição carregada: {n_opinion_0} x '{self.opinion_pair[0]}', {n_opinion_1} x '{self.opinion_pair[1]}'")
            else:
                self._log(f"📊 Gerando nova distribuição ({self.initial_majority}% maioria inicial)...")

                majority_ratio = self.initial_majority / 100.0
                seed_distribution = int(self.seed) if self.seed is not None else int(time.time() * 1000) % 2147483647
                dist = generate_initial_distribution_shared(
                    n_agents=self.n_agents,
                    seed_distribution=seed_distribution,
                    majority_ratio=majority_ratio,
                    requested_mode="auto",
                )
                distribution_mode = str(dist.mode)

                initial_opinions = dist.opinions
                n_opinion_0 = int(dist.count_0)
                n_opinion_1 = int(dist.count_1)
                self._log(
                    f"   → {n_opinion_0} agentes com '{self.opinion_pair[0]}', {n_opinion_1} com '{self.opinion_pair[1]}' "
                    f"(mode={distribution_mode}, seed_dist={seed_distribution})"
                )

            for i in range(self.n_agents):
                self.states[0, i] = initial_opinions[i]

            self._update_heatmap(0, self.n_agents - 1)
            self._log("✅ Rodada 0 inicializada")
            status_text = f"Rodando | Rodada 0/{self.n_rounds - 1} inicializada"
            yield self._build_yield_payload(status_text)

            for r in range(1, self.n_rounds):
                if self._should_stop():
                    self._log("🛑 Simulação interrompida pelo usuário")
                    status_text = "Parado"
                    yield self._build_yield_payload(status_text)
                    return

                self._log(f"🔄 Rodada {r}/{self.n_rounds - 1}...")
                prev_states = self.states[r - 1].copy()

                for i in range(self.n_agents):
                    if self._should_stop():
                        self._log("🛑 Simulação interrompida pelo usuário")
                        status_text = "Parado"
                        yield self._build_yield_payload(status_text)
                        return

                    left, right = self._get_neighbors(i, prev_states)

                    current_val = prev_states[i]
                    if not np.isnan(current_val):
                        current_opinion = self.opinion_pair[int(current_val)]
                    else:
                        current_opinion = self.opinion_pair[0] if self.jogo_conformidade else get_default_opinion(self.prompt_variant)

                    if self.jogo_conformidade:
                        memory_enabled = self.memory_window > 0
                        neighborhood_list = left + [current_opinion] + right
                        system_prompt = render_conformity_game_system_prompt(
                            self.jogo_conformidade_modo,
                            self.opinion_pair[0],
                            self.opinion_pair[1],
                            memory_enabled,
                        )
                        user_prompt = render_conformity_game_user_prompt(
                            neighborhood_list,
                            current_opinion,
                            self.opinion_pair[0],
                            self.opinion_pair[1],
                            memory_enabled,
                        )
                    else:
                        try:
                            system_prompt, user_prompt = self.strategy.build_prompt(
                                left=left,
                                right=right,
                                current_opinion=current_opinion,
                            )
                        except Exception as e:
                            self._log(f"⚠️ Erro no prompt: {e}")
                            self.states[r, i] = np.nan
                            self._update_heatmap(r, i)
                            status_text = f"Rodando | Rodada {r}/{self.n_rounds - 1} | Agente {i + 1}/{self.n_agents}"
                            yield self._build_yield_payload(status_text)
                            continue

                    memory_block = self._format_memory_block(i, r)
                    if memory_block:
                        user_prompt = memory_block + "\n\n" + user_prompt

                    raw_response, token = self._query_llm(system_prompt, user_prompt, i, r)

                    if self._should_stop():
                        self._log("🛑 Simulação interrompida pelo usuário")
                        status_text = "Parado"
                        yield self._build_yield_payload(status_text)
                        return

                    if token is not None and token in OPINION_MAP and token in self.opinion_pair:
                        self.states[r, i] = OPINION_MAP[token]
                        self._log(f"✅ Agente {i}: [{token}]")
                    else:
                        error_msg = (
                            f"Resposta inválida do LLM para agente {i} na rodada {r}: "
                            f"token={token!r}, esperado um de {self.opinion_pair}"
                        )
                        self._log(f"❌ {error_msg}")
                        raise RuntimeError(error_msg)

                    self._update_heatmap(r, i)
                    status_text = f"Rodando | Rodada {r}/{self.n_rounds - 1} | Agente {i + 1}/{self.n_agents}"
                    yield self._build_yield_payload(status_text)

                if self._check_conformity(r):
                    opinion = self._get_conformity_opinion(r)
                    self._log(f"🎯 CONFORMIDADE ATINGIDA na rodada {r}!")
                    self._log(f"   → Todos os agentes escolheram [{opinion}]")
                    status_text = f"🎯 Conformidade na rodada {r} [{opinion}]"
                    yield self._build_yield_payload(status_text)
                    return

            self._log("✅ Simulação concluída (limite de rodadas)!")
            self._log(f"💾 Log salvo: {self.log_filepath}")
            status_text = "Concluído (limite de rodadas)"
            yield self._build_yield_payload(status_text)

        except GeneratorExit:
            self.stop_event.set()
            raise
        except Exception as e:
            error_msg = f"❌ Erro fatal: {str(e)}"
            self._log(error_msg)

            try:
                log_base, _ = os.path.splitext(self.log_filepath)
                error_file = f"{log_base}_ERROR.txt"
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write("ERRO NA SIMULAÇÃO\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Erro: {str(e)}\n")
                    f.write(f"Tipo: {type(e).__name__}\n")
                    import traceback
                    f.write(f"\nTraceback completo:\n{traceback.format_exc()}")
                self._log(f"💾 Erro salvo: {error_file}")
            except Exception:
                pass

            status_text = f"Erro: {str(e)}"
            yield self._build_yield_payload(status_text)


def test_model_connection(base_url: str, model: str) -> Dict[str, Any]:
    """Testa conexão com o modelo LLM."""
    try:
        start_time = time.time()
        result = call_llm_responses(
            base_url=base_url,
            model=model,
            system_prompt="You are a helpful assistant.",
            user_prompt="Say only: [k]",
            temperature=0.0,
            seed=42,
            max_output_tokens=50,
            timeout_s=120,
        )
        latency = (time.time() - start_time) * 1000

        raw_response = str(result["raw_response"]).strip()
        token = parse_llm_response(raw_response)

        return {
            'success': True,
            'raw_response': raw_response,
            'token': token,
            'latency_ms': latency,
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }


def _token_suffix(variant: str) -> str:
    token_suffixes = [
        "_ab", "_01", "_αβ", "_△○", "_⊕⊖", "_pq", "_łþ", "_yesno",
    ]
    for suffix in token_suffixes:
        if suffix in variant:
            if suffix == "_yesno":
                return "noyes"
            return suffix[1:]
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


def _build_variant_options() -> Tuple[List[str], Dict[str, str], str]:
    yaml_path = get_yaml_path()
    available_variants = load_yaml_variants(yaml_path)
    filtered_variants = [v for v in available_variants if v.startswith("v9_") or v.startswith("v21_")]
    display_to_variant = {_display_name(v): v for v in filtered_variants}
    display_options = list(display_to_variant.keys())

    default_variant = "v9_lista_completa_meio"
    default_display = (
        _display_name(default_variant)
        if default_variant in filtered_variants
        else (display_options[0] if display_options else "")
    )
    return display_options, display_to_variant, default_display


def _resolve_ui_tokens(jogo_conformidade: bool, jogo_conformidade_tokens: str) -> Tuple[str, str]:
    if jogo_conformidade:
        return TOKEN_MAP_OPTIONS.get(jogo_conformidade_tokens, ("k", "z"))
    return ("k", "z")


def _initial_distribution_preview(
    n_agents: int,
    initial_majority: int,
    jogo_conformidade: bool,
    jogo_conformidade_tokens: str,
) -> str:
    token0, token1 = _resolve_ui_tokens(jogo_conformidade, jogo_conformidade_tokens)
    n_agents_i = _safe_int(n_agents, 10)
    initial_majority_i = _safe_int(initial_majority, 50)
    ratio_preview = float(initial_majority_i) / 100.0

    if ratio_preview > 0.5:
        n_opinion_0 = int(np.ceil(n_agents_i * ratio_preview))
        min_majority = n_agents_i // 2 + 1
        if n_opinion_0 < min_majority:
            n_opinion_0 = min_majority
    else:
        n_opinion_0 = int(np.floor(n_agents_i * ratio_preview))

    n_opinion_0 = max(0, min(n_agents_i, n_opinion_0))
    n_opinion_1 = n_agents_i - n_opinion_0
    return f"📊 Rodada 0: {n_opinion_0} agentes com '{token0}' (0), {n_opinion_1} com '{token1}' (1)"


def _validate_and_adjust_neighbors(n_agents: int, n_neighbors: int) -> Tuple[int, str]:
    n_agents = _safe_int(n_agents, 10)
    n_neighbors = _safe_int(n_neighbors, 3)
    warning = ""

    max_neighbors = max(1, min(n_agents - 1, 21))
    n_neighbors = max(1, min(n_neighbors, max_neighbors))

    if n_neighbors % 2 == 0:
        warning = "⚠️ Número de vizinhos deve ser ímpar! Ajustando..."
        n_neighbors = n_neighbors + 1 if n_neighbors + 1 < n_agents else n_neighbors - 1
        if n_neighbors < 1:
            n_neighbors = 1

    return n_neighbors, warning


def _on_agents_or_neighbors_change(n_agents: int, n_neighbors: int):
    n_agents_i = _safe_int(n_agents, 10)
    max_neighbors = max(1, min(n_agents_i - 1, 21))
    adjusted_neighbors, warning = _validate_and_adjust_neighbors(n_agents, n_neighbors)
    return (
        gr.update(maximum=max_neighbors, value=adjusted_neighbors),
        warning,
    )


def _on_memory_toggle(use_memory: bool, n_rounds: int, memory_window: int):
    if use_memory:
        memory_window_i = _safe_int(memory_window, 0)
        adjusted = memory_window_i if memory_window_i > 0 else 3
        caption = f"💭 Memória ativa: últimas {adjusted} rodada(s) (self + vizinhos)"
        return (
            gr.update(visible=True, minimum=1, value=adjusted),
            caption,
        )

    return (
        gr.update(visible=False, value=1),
        "💭 Memória desativada",
    )


def _on_server_change(selected_server: str):
    if selected_server == "✏️ Customizado":
        return gr.update(value=DEFAULT_BASE_URL, interactive=True), ""

    base_url = SERVER_PRESETS.get(selected_server, DEFAULT_BASE_URL)
    return gr.update(value=base_url, interactive=False), base_url


def _on_conformity_toggle(jogo_conformidade: bool, jogo_conformidade_tokens: str):
    token_note = "Ordem fixa: no = 0 (primeiro token), yes = 1 (segundo token)." if jogo_conformidade_tokens == "no(0)/yes(1)" else ""
    return (
        gr.update(visible=jogo_conformidade),
        gr.update(visible=jogo_conformidade),
        token_note if jogo_conformidade else "",
    )


def _on_conformity_tokens_change(jogo_conformidade_tokens: str):
    if jogo_conformidade_tokens == "no(0)/yes(1)":
        return "Ordem fixa: no = 0 (primeiro token), yes = 1 (segundo token)."
    return ""


def _parse_initial_config(uploaded_file_path: Optional[str]) -> Tuple[Optional[Dict], str]:
    if not uploaded_file_path:
        return None, ""

    try:
        with open(uploaded_file_path, 'r', encoding='utf-8') as f:
            initial_config_data = json.load(f)

        if 'initial_config' not in initial_config_data:
            return None, "❌ JSON inválido: falta 'initial_config'"

        config_array = np.array(initial_config_data['initial_config'])
        n_0 = int(np.sum(config_array == 0))
        n_1 = len(config_array) - n_0

        msg = (
            f"✅ Distribuição carregada: sim #{initial_config_data.get('sim_number', '?')}\n"
            f"Agentes: {len(initial_config_data['initial_config'])}\n"
            f"Seed original: {initial_config_data.get('seed', 'N/A')}\n"
            f"Distribuição: {n_0} x '0' ({n_0/len(config_array)*100:.1f}%), "
            f"{n_1} x '1' ({n_1/len(config_array)*100:.1f}%)"
        )
        return initial_config_data, msg
    except Exception as e:
        return None, f"❌ Erro ao ler JSON: {str(e)}"


def _on_initial_config_upload(uploaded_file):
    # Não valida JSON nesta etapa para evitar erro durante configuração.
    # A validação acontece somente ao clicar em Rodar.
    if uploaded_file:
        return "Arquivo JSON selecionado. Validação será feita ao rodar."
    return ""


def _test_connection_ui(base_url: str, model: str):
    result = test_model_connection(base_url, model)
    if result['success']:
        connection_text = "Conectado ✅"
        details = (
            f"Latência: {result['latency_ms']:.1f}ms\n"
            f"Resposta: {result['raw_response'][:100]}\n"
            f"Token extraído: {result['token']}"
        )
    else:
        connection_text = "Erro ❌"
        details = f"Erro: {result['error']}"

    return connection_text, details


def _request_stop():
    # Sinaliza parada e imediatamente restaura idle (igual ao v2: clicar Parar
    # reabilita Rodar/Testar e desabilita Parar na mesma resposta, sem race).
    stop_event = _get_active_stop_event()
    if stop_event is not None:
        stop_event.set()
    test_upd, run_upd, stop_upd = _idle_controls()
    return "Parado", test_upd, run_upd, stop_upd


def _on_run_clicked_ui():
    test_upd, run_upd, stop_upd = _running_controls()
    return "Rodando", test_upd, run_upd, stop_upd


def run_simulation_stream(
    n_agents: int,
    n_rounds: int,
    use_memory: bool,
    memory_window: int,
    jogo_conformidade: bool,
    jogo_conformidade_modo: str,
    jogo_conformidade_tokens: str,
    n_neighbors: int,
    temperature: float,
    selected_display_variant: str,
    initial_majority: int,
    use_seed: bool,
    seed_value: int,
    uploaded_file,
    selected_server: str,
    base_url: str,
    model: str,
):
    try:
        n_agents_i = _safe_int(n_agents, 10)
        n_rounds_i = _safe_int(n_rounds, 10)
        n_neighbors_i = _safe_int(n_neighbors, 3)
        initial_majority_i = _safe_int(initial_majority, 50)
        memory_window_i = _safe_int(memory_window, 0)
        seed_i = _safe_int(seed_value, 42)
        temperature_f = _safe_float(temperature, 0.7)
    except Exception as e:
        test_upd, run_upd, stop_upd = _idle_controls()
        yield None, "Nenhuma mensagem ainda...", f"Erro: parâmetros inválidos ({e})", test_upd, run_upd, stop_upd
        return

    display_options, display_to_variant, default_display = _build_variant_options()
    if selected_display_variant in display_to_variant:
        prompt_variant = display_to_variant[selected_display_variant]
    elif default_display in display_to_variant:
        prompt_variant = display_to_variant[default_display]
    elif display_options:
        prompt_variant = display_to_variant[display_options[0]]
    else:
        prompt_variant = "v9_lista_completa_meio"

    _ = selected_server

    delay_ms = 0
    memory_format = "timeline"
    seed = seed_i if use_seed else None

    token0, token1 = _resolve_ui_tokens(jogo_conformidade, jogo_conformidade_tokens)

    adjusted_neighbors, _ = _validate_and_adjust_neighbors(n_agents_i, n_neighbors_i)
    effective_memory_window = memory_window_i if use_memory else 0

    uploaded_file_path = None
    if isinstance(uploaded_file, str):
        uploaded_file_path = uploaded_file
    elif hasattr(uploaded_file, "name"):
        uploaded_file_path = uploaded_file.name

    initial_config_json, initial_config_msg = _parse_initial_config(uploaded_file_path)

    if uploaded_file_path and initial_config_json is None:
        test_upd, run_upd, stop_upd = _idle_controls()
        yield None, initial_config_msg or "Nenhuma mensagem ainda...", "Erro: JSON inválido", test_upd, run_upd, stop_upd
        return

    stop_event = threading.Event()
    _set_active_stop_event(stop_event)

    runner = SimulationRunner(
        n_agents=n_agents_i,
        n_rounds=n_rounds_i,
        n_neighbors=int(adjusted_neighbors),
        temperature=temperature_f,
        prompt_variant=prompt_variant,
        base_url=str(base_url),
        model=str(model),
        delay_ms=delay_ms,
        initial_majority=initial_majority_i,
        stop_event=stop_event,
        seed=seed,
        initial_config_json=initial_config_json,
        memory_window=int(effective_memory_window),
        memory_format=memory_format,
        jogo_conformidade=bool(jogo_conformidade),
        jogo_conformidade_modo=str(jogo_conformidade_modo),
        token0=token0,
        token1=token1,
    )

    if initial_config_msg:
        runner._log(initial_config_msg)

    try:
        test_upd, run_upd, stop_upd = _running_controls()
        yield None, "Nenhuma mensagem ainda...", "Rodando", test_upd, run_upd, stop_upd

        for payload in runner.run_stream():
            status_text = str(payload[2] if len(payload) > 2 else "")
            if status_text.startswith("Rodando"):
                test_upd, run_upd, stop_upd = _running_controls()
            else:
                test_upd, run_upd, stop_upd = _idle_controls()
            yield payload[0], payload[1], payload[2], test_upd, run_upd, stop_upd
    finally:
        _clear_active_stop_event(stop_event)


# ==============================================================================
# GRADIO UI
# ==============================================================================


def build_demo() -> gr.Blocks:
    display_options, _, default_display = _build_variant_options()
    initial_preview = _initial_distribution_preview(10, 50, False, "k/z")

    with gr.Blocks(title="LLM Conformidade Simulator") as demo:
        gr.Markdown("# 🧠 Simulador de Conformidade com LLM")

        with gr.Row():
            with gr.Column(scale=2):
                heatmap_output = gr.Image(label="📊 Heatmap da Simulação", type="pil")
                logs_output = gr.Textbox(
                    label="📝 Log de Execução",
                    value="Nenhuma mensagem ainda...",
                    lines=14,
                    max_lines=20,
                    interactive=False,
                )
                status_output = gr.Textbox(
                    label="📈 Status",
                    value="Parado",
                    interactive=False,
                )

            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Configurações")

                n_agents = gr.Number(
                    label="Número de Agentes",
                    value=10,
                    minimum=3,
                    maximum=100,
                    precision=0,
                )

                n_rounds = gr.Number(
                    label="Número de Rodadas",
                    value=10,
                    minimum=2,
                    maximum=1000,
                    precision=0,
                )

                use_memory = gr.Checkbox(
                    label="🧠 Habilitar Memória Local (Rolling Window)",
                    value=False,
                )

                memory_window = gr.Number(
                    label="Memória: turnos anteriores (W)",
                    value=1,
                    minimum=1,
                    precision=0,
                    visible=False,
                )
                memory_caption = gr.Markdown("💭 Memória desativada")

                jogo_conformidade = gr.Checkbox(
                    label="🎮 Jogo de Conformidade (Density-based)",
                    value=False,
                )

                jogo_conformidade_modo = gr.Dropdown(
                    label="Modo do jogo",
                    choices=[CONFORMITY_GAME_MODE_A, CONFORMITY_GAME_MODE_B],
                    value=CONFORMITY_GAME_MODE_A,
                    visible=False,
                )

                jogo_conformidade_tokens = gr.Dropdown(
                    label="Par de tokens",
                    choices=list(TOKEN_MAP_OPTIONS.keys()),
                    value="k/z",
                    visible=False,
                )
                token_caption = gr.Markdown("")

                n_neighbors = gr.Number(
                    label="Número de Vizinhos",
                    value=3,
                    minimum=1,
                    maximum=9,
                    step=2,
                    precision=0,
                )
                neighbors_warning = gr.Markdown("")

                temperature = gr.Slider(
                    label="Temperatura",
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                )

                prompt_variant = gr.Dropdown(
                    label="Variante de Prompt",
                    choices=display_options,
                    value=default_display,
                )

                initial_majority = gr.Slider(
                    label="Maioria Inicial (%)",
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=1,
                )

                initial_preview_md = gr.Markdown(initial_preview)

                gr.Markdown("---")
                gr.Markdown("## 🎲 Controle de Reprodutibilidade")

                use_seed = gr.Checkbox(label="Usar seed fixo", value=False)
                seed_value = gr.Number(
                    label="Seed",
                    value=42,
                    minimum=1,
                    maximum=999999,
                    precision=0,
                    visible=False,
                )

                uploaded_file = gr.File(
                    label="Carregar distribuição inicial (JSON)",
                    file_types=[".json"],
                    type="filepath",
                )
                upload_status = gr.Textbox(label="Status JSON", value="", interactive=False)

                gr.Markdown("---")
                gr.Markdown("## 🔌 Conexão LLM")

                selected_server = gr.Dropdown(
                    label="Servidor",
                    choices=list(SERVER_PRESETS.keys()),
                    value="LM Studio (Mac Studio)",
                )
                base_url = gr.Textbox(label="Base URL (opcional)", value=SERVER_PRESETS["LM Studio (Mac Studio)"], interactive=False)
                base_url_hint = gr.Markdown(SERVER_PRESETS["LM Studio (Mac Studio)"])

                model = gr.Dropdown(
                    label="Modelo",
                    choices=AVAILABLE_MODELS,
                    value=DEFAULT_MODEL,
                )

                with gr.Row():
                    test_btn = gr.Button("🔍 Testar")
                    run_btn = gr.Button("▶️ Rodar", variant="primary", interactive=True)
                    stop_btn = gr.Button("⏹️ Parar", variant="stop", interactive=False)

                connection_status = gr.Textbox(label="Conexão", value="Desconectado", interactive=False)
                test_result = gr.Textbox(label="🧪 Resultado do Teste", value="", lines=4, interactive=False)

        use_memory.change(
            _on_memory_toggle,
            inputs=[use_memory, n_rounds, memory_window],
            outputs=[memory_window, memory_caption],
        )
        n_rounds.change(
            _on_memory_toggle,
            inputs=[use_memory, n_rounds, memory_window],
            outputs=[memory_window, memory_caption],
        )
        memory_window.change(
            _on_memory_toggle,
            inputs=[use_memory, n_rounds, memory_window],
            outputs=[memory_window, memory_caption],
        )

        jogo_conformidade.change(
            _on_conformity_toggle,
            inputs=[jogo_conformidade, jogo_conformidade_tokens],
            outputs=[jogo_conformidade_modo, jogo_conformidade_tokens, token_caption],
        )
        jogo_conformidade_tokens.change(
            _on_conformity_tokens_change,
            inputs=[jogo_conformidade_tokens],
            outputs=[token_caption],
        )

        n_agents.change(
            _on_agents_or_neighbors_change,
            inputs=[n_agents, n_neighbors],
            outputs=[n_neighbors, neighbors_warning],
        )
        n_neighbors.change(
            _on_agents_or_neighbors_change,
            inputs=[n_agents, n_neighbors],
            outputs=[n_neighbors, neighbors_warning],
        )

        initial_majority.change(
            _initial_distribution_preview,
            inputs=[n_agents, initial_majority, jogo_conformidade, jogo_conformidade_tokens],
            outputs=[initial_preview_md],
        )
        n_agents.change(
            _initial_distribution_preview,
            inputs=[n_agents, initial_majority, jogo_conformidade, jogo_conformidade_tokens],
            outputs=[initial_preview_md],
        )
        jogo_conformidade.change(
            _initial_distribution_preview,
            inputs=[n_agents, initial_majority, jogo_conformidade, jogo_conformidade_tokens],
            outputs=[initial_preview_md],
        )
        jogo_conformidade_tokens.change(
            _initial_distribution_preview,
            inputs=[n_agents, initial_majority, jogo_conformidade, jogo_conformidade_tokens],
            outputs=[initial_preview_md],
        )

        use_seed.change(
            lambda v: gr.update(visible=bool(v), value=42 if bool(v) else 42),
            inputs=[use_seed],
            outputs=[seed_value],
        )

        selected_server.change(
            _on_server_change,
            inputs=[selected_server],
            outputs=[base_url, base_url_hint],
        )

        uploaded_file.change(
            _on_initial_config_upload,
            inputs=[uploaded_file],
            outputs=[upload_status],
        )

        test_btn.click(
            _test_connection_ui,
            inputs=[base_url, model],
            outputs=[connection_status, test_result],
        )

        run_pre_event = run_btn.click(
            _on_run_clicked_ui,
            inputs=None,
            outputs=[status_output, test_btn, run_btn, stop_btn],
            queue=False,
        )

        run_event = run_pre_event.then(
            run_simulation_stream,
            inputs=[
                n_agents,
                n_rounds,
                use_memory,
                memory_window,
                jogo_conformidade,
                jogo_conformidade_modo,
                jogo_conformidade_tokens,
                n_neighbors,
                temperature,
                prompt_variant,
                initial_majority,
                use_seed,
                seed_value,
                uploaded_file,
                selected_server,
                base_url,
                model,
            ],
            outputs=[heatmap_output, logs_output, status_output, test_btn, run_btn, stop_btn],
        )

        stop_btn.click(
            _request_stop,
            inputs=None,
            outputs=[status_output, test_btn, run_btn, stop_btn],
            cancels=[run_event],
            queue=False,
        )

    return demo


demo = build_demo()
try:
    demo.queue(concurrency_count=1)
except TypeError:
    # Gradio >= 6 usa default_concurrency_limit no lugar de concurrency_count.
    demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch()


# ==============================================================================
# README (quick start)
# ==============================================================================
# Run:
#   python projeto_final/streamlit_test/interface/interface_v3_gradio.py
# Optional:
#   export LMSTUDIO_BASE_URL="http://172.18.254.18:1234/v1"
# Notes:
#   - Uses Gradio streaming generator (`run_simulation_stream`) for per-agent updates.
#   - Stop button sets stop event and cancels the running Gradio event.
