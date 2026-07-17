import os
import time
import re
import random
import argparse
import csv
import json
import datetime
import hashlib
import yaml
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para threads
import matplotlib.pyplot as plt
from openai import OpenAI
import httpx
from types import SimpleNamespace
from prompt_strategies import get_prompt_strategy, PromptStrategy
import pandas as pd
import fcntl
import psutil
# threading removido - usando apenas asyncio
import asyncio
from contextvars import ContextVar
import db_sqlite

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

# ===============================================================================
# LOGPROBS (LM Studio Responses)
# ===============================================================================
# Default: always request logprobs from /v1/responses and store them in CSV basic.
# Observado no LM Studio (Mac Studio) que valores >=20 podem retornar 500.
LOGPROBS_TOP = int(os.getenv("LOGPROBS_TOP", "10"))

# Variável global para configuração de prompts
PROMPT_VARIANT = "v20_lista_completa_meio_raciocinio_primeiro"  # Variante padrão de prompt

# ===============================================================================
# PARÂMETROS DE SAMPLING (ENV OVERRIDES)
# ===============================================================================

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
    """
    Applies optional sampling overrides to the LM Studio OpenAI-compatible payload.
    Only sets keys when env vars are present, to preserve legacy behavior by default.
    """
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


def get_llm_api_format() -> str:
    """Backend de inferência. Default preserva o fluxo LM Studio /v1/responses."""
    return os.getenv("LLM_API_FORMAT", "responses").strip().lower()


def is_sglang_backend() -> bool:
    return get_llm_api_format() == "sglang"


def is_vllm_backend() -> bool:
    return get_llm_api_format() == "vllm"


def _looks_like_qwen_model(model_name: str) -> bool:
    return "qwen" in (model_name or "").lower()


def _append_no_think_if_needed(user_prompt: str, model_name: str) -> str:
    """Qwen only: append /no_think unless explicitly disabled or already present."""
    if not _looks_like_qwen_model(model_name):
        return user_prompt
    flag = os.getenv("ENABLE_QWEN_NO_THINK", "true").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return user_prompt
    if "/no_think" in user_prompt or "/no_thinking" in user_prompt:
        return user_prompt
    return f"{user_prompt.rstrip()}\n\n/no_think"


def _sglang_model_path(model_name: str) -> str:
    """Resolve aliases antigos da execução simultânea para model-path/tokenizer HF."""
    explicit = os.getenv("SGLANG_MODEL_PATH", "").strip()
    if explicit:
        return explicit

    model_lower = (model_name or "").lower()
    if "llama" in model_lower and "8b" in model_lower:
        return "meta-llama/Llama-3.1-8B-Instruct"
    if "gemma" in model_lower and "4b" in model_lower:
        return "google/gemma-3-4b-it"
    if "qwen" in model_lower and "4b" in model_lower:
        return "Qwen/Qwen3-4B"
    return model_name


TOKENIZER_CACHE: Dict[str, Any] = {}


def _get_sglang_tokenizer(model_name: str):
    if AutoTokenizer is None:
        raise RuntimeError("transformers.AutoTokenizer indisponível; necessário para SGLang /generate com chat template")
    tokenizer_path = os.getenv("SGLANG_TOKENIZER_PATH", "").strip() or _sglang_model_path(model_name)
    tok = TOKENIZER_CACHE.get(tokenizer_path)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        TOKENIZER_CACHE[tokenizer_path] = tok
    return tok


def _token_pair_for_prompt_variant(prompt_variant: str):
    """Mantém o stop_regex alinhado aos tokens das variantes v9/v21 atuais."""
    v = (prompt_variant or "").strip()
    mapping = {
        "v9_lista_completa_meio": ("k", "z"),
        "v9_lista_completa_meio_kz": ("k", "z"),
        "v21_zero_shot_cot": ("k", "z"),
        "v21_zero_shot_cot_kz": ("k", "z"),
        "v9_lista_completa_meio_ab": ("a", "b"),
        "v21_zero_shot_cot_ab": ("a", "b"),
        "v9_lista_completa_meio_01": ("0", "1"),
        "v21_zero_shot_cot_01": ("0", "1"),
        "v9_lista_completa_meio_pq": ("p", "q"),
        "v21_zero_shot_cot_pq": ("p", "q"),
        "v9_lista_completa_meio_yesno": ("no", "yes"),
        "v21_zero_shot_cot_yesno": ("no", "yes"),
        "v9_lista_completa_meio_αβ": ("α", "β"),
        "v21_zero_shot_cot_αβ": ("α", "β"),
        "v9_lista_completa_meio_△○": ("△", "○"),
        "v21_zero_shot_cot_△○": ("△", "○"),
        "v9_lista_completa_meio_⊕⊖": ("⊕", "⊖"),
        "v21_zero_shot_cot_⊕⊖": ("⊕", "⊖"),
        "v9_lista_completa_meio_łþ": ("ł", "þ"),
        "v21_zero_shot_cot_łþ": ("ł", "þ"),
    }
    if v not in mapping:
        raise ValueError(f"Variante não suportada para SGLang stop_regex: {prompt_variant!r}")
    return mapping[v]


def _make_sglang_stop_regex(prompt_variant: str) -> str:
    escaped = sorted([re.escape(t) for t in _token_pair_for_prompt_variant(prompt_variant)], key=len, reverse=True)
    return r"\[(?:" + "|".join(escaped) + r")\]"

# ===============================================================================
# HELPERS PARA BASE URL DO LM STUDIO
# ===============================================================================

def _derive_base_url_root(openai_base_url: str) -> str:
    """Converte base_url OpenAI (/v1) para raiz do servidor."""
    url = openai_base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url.rstrip("/")

# Lista de todas as variantes disponíveis para teste
VARIANTES_TESTE = [
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
    # Adicionando variantes com sufixo _kz para consistência
    'v9_lista_completa_meio_kz',
    'v20_lista_completa_meio_raciocinio_primeiro_kz'
]

# Variante equivalente à v20 mas SEM raciocínio (apenas [k] ou [z])
# Útil para comparar diretamente com v20 sem alterar o restante do fluxo
VARIANTE_V20_EQUIVALENTE_SEM_RACIOCINIO = 'v9_lista_completa_meio'
VARIANTES_TESTE_V20_SEM_RACIOCINIO = [VARIANTE_V20_EQUIVALENTE_SEM_RACIOCINIO]
# Lista para testar a variante V20 (com raciocínio primeiro)
VARIANTES_TESTE_V20_COM_RACIOCINIO = ['v20_lista_completa_meio_raciocinio_primeiro']
# Importa a configuração se disponível, caso contrário usa configurações locais
try:
    from config import ConformityConfig
    config = ConformityConfig()
    client = config.get_client()
    # Deriva base URL do cliente, com fallback para env
    _env_base = os.getenv("LMSTUDIO_BASE_URL")
    if _env_base:
        BASE_URL_ROOT = _derive_base_url_root(_env_base)
    else:
        try:
            BASE_URL_ROOT = _derive_base_url_root(str(client.base_url))
        except Exception:
            BASE_URL_ROOT = "http://172.18.254.16:1234"
except ImportError:
    # Detectar se estamos executando no servidor local ou remotamente
    import socket
    
    # 1. Tentar pegar via variável de ambiente (prioridade máxima)
    _env_base = os.getenv("LMSTUDIO_BASE_URL")
    
    if _env_base:
        BASE_URL_ROOT = _derive_base_url_root(_env_base)
    else:
        # 2. Detecção automática baseada no IP
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
            # IPs conhecidos onde o LM Studio roda localmente
            if local_ip == "172.18.254.16" or local_ip == "172.18.254.18":
                # Executando no servidor Linux ou Mac Studio - usar localhost
                BASE_URL_ROOT = "http://127.0.0.1:1234"
            else:
                # Executando remotamente de outro lugar - usar IP do servidor Linux por padrão
                BASE_URL_ROOT = "http://172.18.254.16:1234"
        except:
            # Em caso de erro, assumir execução remota padrão
            BASE_URL_ROOT = "http://172.18.254.16:1234"
            
    local_client = OpenAI(base_url=f"{BASE_URL_ROOT}/v1", api_key="lm-studio")
    client = local_client

# Cliente assíncrono para execução com asyncio
try:
    from openai import AsyncOpenAI
    # Usa a mesma lógica de detecção de IP/Env para o cliente assíncrono
    import socket
    
    # 1. Tentar pegar via variável de ambiente (prioridade máxima)
    _env_base_async = os.getenv("LMSTUDIO_BASE_URL")
    
    if _env_base_async:
        # Se definido no env, usa ele
        _base_url_async = _derive_base_url_root(_env_base_async)
        async_client = AsyncOpenAI(base_url=f"{_base_url_async}/v1", api_key="lm-studio")
    else:
        # 2. Detecção automática
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
            if local_ip == "172.18.254.16" or local_ip == "172.18.254.18":
                # Executando no servidor Linux ou Mac Studio - usar localhost
                async_client = AsyncOpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
            else:
                # Executando remotamente - usar IP do servidor
                async_client = AsyncOpenAI(base_url="http://172.18.254.16:1234/v1", api_key="lm-studio")
        except:
            # Em caso de erro, assumir execução remota
            async_client = AsyncOpenAI(base_url="http://172.18.254.16:1234/v1", api_key="lm-studio")
except ImportError:
    async_client = None  # fallback: usar asyncio.to_thread com client síncrono

# Cliente HTTP async para API nativa do LM Studio
NATIVE_HTTP_CLIENT: Optional[httpx.AsyncClient] = None

def _get_native_http_client() -> httpx.AsyncClient:
    """Cria/retorna client HTTP async reutilizável."""
    global NATIVE_HTTP_CLIENT
    if NATIVE_HTTP_CLIENT is None:
        NATIVE_HTTP_CLIENT = httpx.AsyncClient(timeout=120.0)
    return NATIVE_HTTP_CLIENT

# Constantes
# Mapeamento completo incluindo tokens especiais e yes/no
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
REVERSE_OPINION_MAP = {0: 'k', 1: 'z'}  # Mapeia valores numéricos para letras (mantém k/z como padrão)
# TEMPERATURES = [0.0, 0.2, 0.5, 0.8, 1.0]  # Diferentes temperaturas para teste
# TEMPERATURES = [0.0, 1.0]  # Diferentes temperaturas para teste
TEMPERATURES = [0.0]  # Diferentes temperaturas para teste
# TEMPERATURES = [round(x * 0.5, 2) for x in range(0, 7)]  # [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Modelos disponíveis no LM Studio (3 deploys de cada para paralelização)
# Pools: workers do mesmo pool pegam experimentos em shuffle
MODEL_POOLS = {
    "gemma4b-pool": [
        "google/gemma-3-4b",              # Gemma 4B - Deploy 1
        "google/gemma-3-4b:2",            # Gemma 4B - Deploy 2
        "google/gemma-3-4b:3"             # Gemma 4B - Deploy 3
    ],
    "gemma12b-pool": [
        "google/gemma-3-12b",             # Gemma 12B - Deploy 1
        "google/gemma-3-12b:2",           # Gemma 12B - Deploy 2
        "google/gemma-3-12b:3",           # Gemma 12B - Deploy 3
        "google/gemma-3-12b:4",           # Gemma 12B - Deploy 4
        "google/gemma-3-12b:5",           # Gemma 12B - Deploy 5
        "google/gemma-3-12b:6",           # Gemma 12B - Deploy 6
        "google/gemma-3-12b:7",           # Gemma 12B - Deploy 7
        "google/gemma-3-12b:8",           # Gemma 12B - Deploy 8
        "google/gemma-3-12b:9"            # Gemma 12B - Deploy 9
    ],
    "llama8b-pool": [
        "meta-llama-3.1-8b-instruct",     # Llama 8B - Deploy 1
        "meta-llama-3.1-8b-instruct:2",   # Llama 8B - Deploy 2
        "meta-llama-3.1-8b-instruct:3"    # Llama 8B - Deploy 3
    ],
    "llama70b-pool": [
        "meta-llama-3.1-70b-instruct"     # Llama 70B - Deploy único (modelo grande)
    ],
    "qwen4b-pool": [
        "qwen3-4b"                        # Qwen 3 4B - Deploy único
    ],
    "qwen32b-pool": [
        "qwen/qwen3-32b"                  # Qwen 3 32B - Deploy único (modelo normal, sem QwQ)
    ]
}

# Lista plana de todos os modelos (para compatibilidade)
# Detecta automaticamente quais pools usar baseado no DB
DB_PATH = os.getenv('EXPERIMENTOS_DB_PATH', os.path.join("experimentos", "experimentos.db"))
_active_pools_override = os.getenv("ACTIVE_POOLS_OVERRIDE", "").strip()
if _active_pools_override:
    # Allows running in environments where only a subset of models is loaded in LM Studio.
    # Example: ACTIVE_POOLS_OVERRIDE="gemma12b-pool"
    pools = [p.strip() for p in _active_pools_override.split(",") if p.strip()]
    invalid = [p for p in pools if p not in MODEL_POOLS]
    if invalid:
        raise RuntimeError(
            f"❌ ACTIVE_POOLS_OVERRIDE inválido: {invalid}. "
            f"Pools válidos: {sorted(MODEL_POOLS.keys())}"
        )
    ACTIVE_POOLS = pools
    print(f"🔧 ACTIVE_POOLS_OVERRIDE={ACTIVE_POOLS}")
elif 'qwen' in DB_PATH.lower():
    # Se for DB do Qwen, usar apenas pools Qwen
    ACTIVE_POOLS = ["qwen4b-pool", "qwen32b-pool"]
    print(f"🔍 Detectado DB Qwen: {DB_PATH} - Usando apenas pools Qwen")
elif 'llama70b' in DB_PATH.lower():
    # Se for DB do Llama 70B, usar apenas pool Llama 70B
    ACTIVE_POOLS = ["llama70b-pool"]
    print(f"🔍 Detectado DB Llama 70B: {DB_PATH} - Usando apenas pool Llama 70B")
else:
    # DB padrão: usar Gemma 4B, Llama 8B e Gemma 12B
    ACTIVE_POOLS = ["gemma4b-pool", "llama8b-pool", "gemma12b-pool"]
    print(f"🔍 Detectado DB padrão: {DB_PATH} - Usando pools Gemma 4B, Llama 8B e Gemma 12B")

AVAILABLE_MODELS = [model for pool_name in ACTIVE_POOLS for model in MODEL_POOLS[pool_name]]
_available_models_override = os.getenv("AVAILABLE_MODELS_OVERRIDE", "").strip()
if _available_models_override:
    requested_models = [m.strip() for m in _available_models_override.split(",") if m.strip()]
    known_models = {m for pool in MODEL_POOLS.values() for m in pool}
    invalid_models = [m for m in requested_models if m not in known_models]
    if invalid_models and not (is_sglang_backend() or is_vllm_backend()):
        raise RuntimeError(
            f"❌ AVAILABLE_MODELS_OVERRIDE inválido: {invalid_models}. "
            f"Modelos válidos: {sorted(known_models)}"
        )
    if invalid_models and (is_sglang_backend() or is_vllm_backend()):
        print(f"🔧 LLM_API_FORMAT={get_llm_api_format()}: aceitando aliases fora de MODEL_POOLS: {invalid_models}")
    AVAILABLE_MODELS = requested_models
    print(f"🔧 AVAILABLE_MODELS_OVERRIDE={AVAILABLE_MODELS}")
print(f"📋 Modelos ativos: {AVAILABLE_MODELS}")

MODEL = "gemma-12b"  # Modelo padrão (usado apenas para compatibilidade)
NUM_ITERATIONS = 1  # Número de iterações por configuração
NUM_NEIGHBORS = 3   # Número TOTAL de vizinhos no experimento (sempre ímpar) - TESTE DE VALIDAÇÃO
TOTAL_PARTICIPANTS = 10  # Número total de participantes no experimento


# Configuração dos arquivos de log
LOG_FILE = None  
PROMPT_LOG_FILE = None

# ContextVars para isolamento de contexto em tarefas assíncronas
CURRENT_MODEL: ContextVar[str] = ContextVar("CURRENT_MODEL", default=None)
CURRENT_EXPERIMENT_ID: ContextVar[int] = ContextVar("CURRENT_EXPERIMENT_ID", default=None)
CURRENT_LOG_FILE: ContextVar[str] = ContextVar("CURRENT_LOG_FILE", default=None)
CURRENT_PROMPT_LOG_FILE: ContextVar[str] = ContextVar("CURRENT_PROMPT_LOG_FILE", default=None)
CURRENT_PROMPT_VARIANT: ContextVar[str] = ContextVar("CURRENT_PROMPT_VARIANT", default=None)

# Lock global para métricas de tokens em ambiente assíncrono (será inicializado quando necessário)
TOKEN_STATS_LOCK = None

def get_token_stats_lock():
    """Retorna o lock de estatísticas de tokens, criando-o se necessário"""
    global TOKEN_STATS_LOCK
    if TOKEN_STATS_LOCK is None:
        TOKEN_STATS_LOCK = asyncio.Lock()
    return TOKEN_STATS_LOCK

# Lock global por modelo para garantir 1 request por vez em cada deploy
MODEL_REQUEST_LOCKS: Dict[str, asyncio.Lock] = {}

def get_model_request_lock(model_name: str) -> asyncio.Lock:
    """Retorna o lock associado ao modelo para garantir execução síncrona por modelo."""
    lock = MODEL_REQUEST_LOCKS.get(model_name)
    if lock is None:
        lock = asyncio.Lock()
        MODEL_REQUEST_LOCKS[model_name] = lock
    return lock

# Gate global para evitar concorrência entre workers de modelos diferentes.
# Em alguns setups do LM Studio, /v1/responses com logprobs fica instável com requests paralelos.
GLOBAL_LLM_GATE = None

def get_global_llm_gate() -> asyncio.Semaphore:
    """Retorna um semáforo global (lazy-init) para limitar requests concorrentes ao servidor."""
    global GLOBAL_LLM_GATE
    if GLOBAL_LLM_GATE is None:
        max_inflight = int(os.getenv("GLOBAL_LLM_MAX_INFLIGHT", "1"))
        if max_inflight < 1:
            max_inflight = 1
        GLOBAL_LLM_GATE = asyncio.Semaphore(max_inflight)
    return GLOBAL_LLM_GATE

# ===============================================================================
# FUNÇÕES PARA CONTEXTO ASSÍNCRONO (ContextVars)
# ===============================================================================

def set_task_model(model_name: str) -> None:
    """Define o modelo para a tarefa atual, garantindo binding rígido modelo↔experimento"""
    CURRENT_MODEL.set(model_name)
    print(f"🔒 Task {asyncio.current_task()}: Binding modelo={model_name}")

def get_task_model() -> str:
    """Retorna o modelo vinculado à tarefa atual"""
    model = CURRENT_MODEL.get()
    if not model:
        raise RuntimeError(f"❌ Task {asyncio.current_task()}: Nenhum modelo vinculado! Use set_task_model() primeiro.")
    return model

def set_task_experiment_id(exp_id: int) -> None:
    """Define o ID do experimento para a tarefa atual"""
    CURRENT_EXPERIMENT_ID.set(exp_id)
    print(f"🔒 Task {asyncio.current_task()}: Binding experiment_id={exp_id}")

def get_task_experiment_id() -> int:
    """Retorna o ID do experimento vinculado à tarefa atual"""
    exp_id = CURRENT_EXPERIMENT_ID.get()
    if exp_id is None:
        raise RuntimeError(f"❌ Task {asyncio.current_task()}: Nenhum experiment_id vinculado!")
    return exp_id

def set_task_log_files(log_file_path: str, prompt_log_file_path: str) -> None:
    """Configura os caminhos de log na tarefa atual."""
    CURRENT_LOG_FILE.set(log_file_path)
    CURRENT_PROMPT_LOG_FILE.set(prompt_log_file_path)

def get_current_log_file():
    """Retorna o caminho do arquivo de log para o contexto atual (tarefa assíncrona)"""
    return CURRENT_LOG_FILE.get()

def get_current_prompt_log_file():
    """Retorna o caminho do arquivo de log de prompt para o contexto atual (tarefa assíncrona)"""
    return CURRENT_PROMPT_LOG_FILE.get()

def set_task_prompt_variant(variant: str) -> None:
    """Define a variante de prompt para a tarefa atual"""
    CURRENT_PROMPT_VARIANT.set(variant)
    print(f"🎯 Task {asyncio.current_task()}: Binding variant={variant}")

def get_task_prompt_variant() -> str:
    """Retorna a variante de prompt da tarefa atual"""
    variant = CURRENT_PROMPT_VARIANT.get()
    if variant is None:
        # Fallback para a variável global se não estiver definida no contexto
        variant = PROMPT_VARIANT
    return variant

# ===============================================================================
# SISTEMA DE ORQUESTRAÇÃO DE EXPERIMENTOS
# ===============================================================================

def process_exists(pid_str):
    """Verifica se um processo ainda existe localmente"""
    # Lida com valores NaN do pandas
    if pd.isna(pid_str) or not pid_str:
        return False
    
    # Converte para string se for float/int
    if isinstance(pid_str, (int, float)):
        pid_str = str(int(pid_str))
    
    if not pid_str or pid_str.strip() == '':
        return False
    
    try:
        # CORREÇÃO: Extrai apenas o PID real (antes do _) para verificação
        # Novo formato: "PID_hash" (ex: "2223372_a1b2c3d4")
        if '_' in str(pid_str):
            actual_pid = int(str(pid_str).split('_')[0])
        else:
            # Formato antigo: apenas PID
            actual_pid = int(pid_str)
        
        return psutil.pid_exists(actual_pid)
    except (ValueError, TypeError, IndexError) as e:
        print(f"⚠️  Erro ao verificar PID {pid_str}: {e}")
        return False


def clear_experiment_fields(df, idx):
    """Limpa campos de um experimento de forma type-safe"""
    df.loc[idx, 'status'] = 'pendente'
    df.loc[idx, 'process_id'] = pd.NA
    df.loc[idx, 'hora_inicio'] = pd.NA
    df.loc[idx, 'modelo_executado'] = pd.NA
    df.loc[idx, 'caminho_csv_saida'] = pd.NA
    df.loc[idx, 'caminho_log_saida'] = pd.NA
    df.loc[idx, 'caminho_log_prompt_saida'] = pd.NA

def delete_experiment_files(row, verbose=False):
    """Deleta arquivos de um experimento órfão"""
    files_to_delete = [
        row.get('caminho_csv_saida', ''),
        row.get('caminho_log_saida', ''),
        row.get('caminho_log_prompt_saida', '')
    ]
    
    deleted_files = []
    for file_path in files_to_delete:
        # Converte para string e remove valores NaN
        if pd.isna(file_path) or not file_path:
            continue
            
        file_path = str(file_path).strip()
        if not file_path:
            continue
            
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_files.append(os.path.basename(file_path))
                if verbose:
                    print(f"  🗑️  Deletado: {os.path.basename(file_path)}")
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Erro ao deletar {os.path.basename(file_path)}: {e}")
    
    if verbose and deleted_files:
        print(f"  📁 {len(deleted_files)} arquivo(s) deletado(s) do experimento {row['id_experimento']}")
    
    return len(deleted_files)

class ExperimentManager:
    """Gerencia experimentos usando SQLite como fonte da verdade"""
    
    def __init__(self, csv_path=None):
        # CSV não é mais usado internamente, apenas para compatibilidade
        # Usa variável de ambiente se disponível, senão usa o padrão
        if csv_path is None:
            csv_path = os.getenv('EXPERIMENTOS_CSV_PATH', 'experimentos/experimentos_master.csv')
        self.csv_path = csv_path
        self.process_id = str(os.getpid())
        
        # SQLite é a fonte da verdade
        db_sqlite.init_db()
    
    def _create_empty_csv(self):
        """Cria um CSV vazio com o cabeçalho correto"""
        headers = [
            'id_experimento', 'conjunto_experimento', 'status', 'process_id', 
            'hora_inicio', 'hora_fim', 'modelo', 'modelo_executado',
            'num_vizinhos', 'num_iteracoes', 'temperaturas', 'variante_prompt', 
            'notas', 'caminho_csv_saida', 'caminho_log_saida', 'caminho_log_prompt_saida'
        ]
        
        empty_df = pd.DataFrame(columns=headers)
        empty_df.to_csv(self.csv_path, index=False)
        print(f"Criado arquivo CSV vazio: {self.csv_path}")
    
    def _atomic_csv_operation_DEPRECATED(self, operation_func):
        """Executa operação no CSV com lock atômico + busy flag + rename atômico"""
        max_retries = 10  # Mais tentativas para concorrência
        retry_delay = 0.05  # Delay menor inicial
        
        busy_flag = self.csv_path + '.busy'
        temp_file = self.csv_path + '.tmp'
        
        for attempt in range(max_retries):
            try:
                # Cria busy flag para indicar operação em andamento
                with open(busy_flag, 'w') as bf:
                    try:
                        task_id = id(asyncio.current_task()) if asyncio.current_task() else "main"
                    except RuntimeError:
                        task_id = "thread"
                    bf.write(f"{os.getpid()}_{task_id}")
                
                print(f"🔒 CSV busy flag ativado: {busy_flag}")
                
                with open(self.csv_path, 'r+', encoding='utf-8') as f:
                    # Tenta obter lock exclusivo
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    # Lê o CSV
                    f.seek(0)
                    df = pd.read_csv(f)

                    # Normaliza tipos e espaços em colunas críticas para evitar falhas de filtro
                    try:
                        if 'status' in df.columns:
                            df['status'] = df['status'].fillna('').astype(str).str.strip().str.lower()
                        if 'id_experimento' in df.columns:
                            df['id_experimento'] = pd.to_numeric(df['id_experimento'], errors='coerce')
                        if 'modelo_executado' in df.columns:
                            df['modelo_executado'] = df['modelo_executado'].fillna('').astype(str).str.strip()
                        if 'conjunto_experimento' in df.columns:
                            df['conjunto_experimento'] = df['conjunto_experimento'].fillna('').astype(str).str.strip()
                        # CORREÇÃO PANDAS: Garantir que colunas críticas tenham dtypes corretos
                        if 'process_id' in df.columns:
                            df['process_id'] = df['process_id'].astype('object')  # Permite strings e NaN
                        if 'hora_inicio' in df.columns:
                            df['hora_inicio'] = df['hora_inicio'].astype('object')  # Permite strings e vazios
                        if 'hora_fim' in df.columns:
                            df['hora_fim'] = df['hora_fim'].astype('object')  # Permite strings e vazios
                    except Exception:
                        # Em caso de erro na normalização, segue com o df original
                        pass
                    
                    # OTIMIZAÇÃO: Executa operação o mais rápido possível para minimizar lock time
                    start_time = time.time()
                    result = operation_func(df)
                    operation_time = time.time() - start_time
                    
                    # Salva as mudanças usando ESCRITA ATÔMICA se necessário
                    if result.get('save_changes', False):
                        # Escreve para arquivo temporário primeiro
                        result['df'].to_csv(temp_file, index=False)
                        
                        # RENAME ATÔMICO: Substitui o arquivo original
                        os.replace(temp_file, self.csv_path)
                        
                        print(f"⚡ CSV atomicamente atualizado via {temp_file}")
                    
                    # Log timing para otimização
                    if operation_time > 0.1:  # Log apenas operações lentas
                        try:
                            task_name = getattr(asyncio.current_task(), 'get_name', lambda: 'main')() if asyncio.current_task() else 'main'
                        except RuntimeError:
                            task_name = 'thread'
                        print(f"⚡ CSV operation took {operation_time:.3f}s (task: {task_name})")
                    
                    # Remove busy flag
                    try:
                        os.remove(busy_flag)
                        print(f"🔓 CSV busy flag removido")
                    except:
                        pass  # Flag pode já ter sido removido
                    
                    # Retorna o resultado
                    return result.get('return_value')
                    
            except (IOError, OSError) as e:
                # Remove busy flag em caso de erro
                try:
                    os.remove(busy_flag)
                    print(f"🔓 CSV busy flag removido (erro)")
                except:
                    pass
                
                if attempt < max_retries - 1:
                    # Backoff exponencial com jitter para reduzir contenção
                    import random
                    jitter = random.uniform(0.8, 1.2)  # ±20% de variação aleatória
                    sleep_time = retry_delay * (2 ** attempt) * jitter
                    time.sleep(sleep_time)
                    print(f"🔄 Retry #{attempt+1}/{max_retries} após {sleep_time:.3f}s (lock CSV ocupado)")
                    continue
                else:
                    # Falha crítica - log detalhado para debugging
                    print(f"❌ FALHA CRÍTICA: Lock CSV falhou após {max_retries} tentativas")
                    print(f"   Erro: {type(e).__name__}: {e}")
                    try:
                        task_name = getattr(asyncio.current_task(), 'get_name', lambda: 'main')() if asyncio.current_task() else 'main'
                    except RuntimeError:
                        task_name = 'thread'
                    print(f"   Task: {task_name}")
                    print(f"   PID: {os.getpid()}")
                    return None
            except Exception as e:
                # Remove busy flag em caso de erro inesperado
                try:
                    os.remove(busy_flag)
                    print(f"🔓 CSV busy flag removido (erro inesperado)")
                except:
                    pass
                
                # Qualquer outro erro inesperado
                print(f"❌ ERRO INESPERADO na operação atômica: {type(e).__name__}: {e}")
                return None
    
    def recover_orphaned_experiments(self):
        """Recupera experimentos órfãos que ficaram em status 'executando' usando SQLite"""
        status = db_sqlite.get_status()
        recovered = 0
        
        for exp in status["experiments"]:
            if exp["status"] == "executando":
                should_recover = False
                reason = ""
                
                # Verifica se processo ainda existe
                if not process_exists(exp.get("process_id")):
                    should_recover = True
                    reason = f"Processo {exp.get('process_id')} não existe mais"
                
                
                if should_recover:
                    print(f"🔄 Recuperando experimento {exp['id_experimento']}: {reason}")
                    
                    # Deleta arquivos existentes do experimento órfão
                    delete_experiment_files(exp, verbose=True)
                    
                    # Reseta status no SQLite
                    success = db_sqlite.update_experiment_status(
                        exp["id_experimento"], "pendente",
                        notas="RECUPERADO: processo inexistente",
                        modelo_executado=None, caminho_csv_saida=None,
                        caminho_log_saida=None, caminho_log_prompt_saida=None
                    )
                    if success:
                        recovered += 1
        
        if recovered > 0:
            print(f"✅ Recuperados {recovered} experimento(s) órfão(s)")
        
        return recovered
    
    def get_next_experiment(self, preferred_model=None, start_id=None, end_id=None):
        """Obtém o próximo experimento disponível de forma atômica
        
        Args:
            preferred_model: Modelo específico para reservar (opcional)
            start_id: ID mínimo para considerar (opcional)
            end_id: ID máximo para considerar (opcional)
        """
        # asyncio task ID para identificação única (com tratamento para threads)
        try:
            task_id = id(asyncio.current_task()) if asyncio.current_task() else "main"
        except RuntimeError:
            # Estamos em uma thread sem loop de eventos (asyncio.to_thread)
            task_id = "thread"
        task_hash = hashlib.md5(str(task_id).encode()).hexdigest()[:8]
        process_id = f"{os.getpid()}_{task_hash}"
        
        return db_sqlite.reserve_next_experiment(
            preferred_model=preferred_model,
            start_id=start_id,
            end_id=end_id,
            available_models=AVAILABLE_MODELS,
            process_id=process_id
        )
    
    def get_experiment_by_id(self, experiment_id, preferred_model=None):
        """Obtém um experimento específico por ID se estiver pendente usando SQLite
        
        Args:
            experiment_id: ID do experimento específico
            preferred_model: Modelo específico para usar (opcional)
        """
        # asyncio task ID para identificação única (com tratamento para threads)
        try:
            task_id = id(asyncio.current_task()) if asyncio.current_task() else "main"
        except RuntimeError:
            # Estamos em uma thread sem loop de eventos (asyncio.to_thread)
            task_id = "thread"
        task_hash = hashlib.md5(str(task_id).encode()).hexdigest()[:8]
        process_id = f"{os.getpid()}_{task_hash}"
        
        # Usa a função do db_sqlite que já faz tudo isso de forma atômica
        return db_sqlite.reserve_next_experiment(
            preferred_model=preferred_model,
            start_id=experiment_id,  # Força buscar apenas este ID específico
            end_id=experiment_id,    # Força buscar apenas este ID específico
            available_models=AVAILABLE_MODELS,
            process_id=process_id
        )
    
    def update_experiment_status(self, experiment_id, status, **kwargs):
        """Atualiza o status de um experimento usando SQLite + cria manifest atômico"""
        success = db_sqlite.update_experiment_status(experiment_id, status, **kwargs)
        
        # MANIFEST ATÔMICO: Se concluído, cria manifest APÓS DB ser atualizado
        # Manifest removido - SQLite é a fonte da verdade suficiente
        
        return success
        
    # Função _create_experiment_manifest removida - SQLite é suficiente como fonte da verdade
    pass
    
    def get_experiments_status(self):
        """Retorna o status atual de todos os experimentos usando SQLite"""
        return db_sqlite.get_status()

# Variáveis globais para tracking de performance
TOKEN_STATS = {
    'total_tokens': 0,
    'total_time': 0.0,
    'request_count': 0,
    'running_avg_tps': 0.0
}  

def clean_old_logs(experiment_id=None, prompt_variant=None, num_neighbors=None, model_name=None):
    """Limpa logs antigos de um experimento específico para evitar acumulação"""
    try:
        # Usa os valores fornecidos ou os globais
        variant_for_name = prompt_variant if prompt_variant is not None else PROMPT_VARIANT
        neighbors_for_name = num_neighbors if num_neighbors is not None else NUM_NEIGHBORS
        model_for_name = (model_name if model_name else MODEL or 'unknown').replace('-', '_').replace('/', '_').replace(':', '_')
        exp_suffix = f"_exp{experiment_id}" if experiment_id else ""
        
        logs_dir = os.path.join(os.path.dirname(__file__), 'resultados', 'logs')
        prompt_logs_dir = os.path.join(os.path.dirname(__file__), 'resultados', 'prompt_logs')
        
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(prompt_logs_dir, exist_ok=True)
        
        # Padrões mais específicos para encontrar arquivos
        log_prefixes = [
            f"respostas_log_{variant_for_name}_n{neighbors_for_name}{exp_suffix}_",
            f"prompts_log_{variant_for_name}_n{neighbors_for_name}{exp_suffix}_"
        ]
        
        log_suffix = f"_{model_for_name}.txt"
        
        deleted_count = 0
        directories = [
            ("logs", logs_dir, log_prefixes[0]),
            ("prompt_logs", prompt_logs_dir, log_prefixes[1])
        ]
        
        for dir_name, directory, prefix in directories:
            if os.path.exists(directory):
                print(f"  🔍 Verificando diretório {dir_name}...")
                for file_name in os.listdir(directory):
                    # Verifica se o arquivo corresponde exatamente ao padrão esperado
                    if file_name.startswith(prefix) and file_name.endswith(log_suffix):
                        full_path = os.path.join(directory, file_name)
                        try:
                            os.remove(full_path)
                            deleted_count += 1
                            print(f"  🗑️  Log antigo deletado de {dir_name}: {file_name}")
                        except Exception as e:
                            print(f"  ⚠️  Erro ao deletar log de {dir_name} {file_name}: {e}")
            else:
                print(f"  📁 Diretório {dir_name} não existe ainda")
        
        if deleted_count > 0:
            print(f"  🧹 Total: {deleted_count} arquivo(s) de log antigo(s) deletado(s)")
        else:
            print(f"  ✅ Nenhum log antigo encontrado para limpeza")
        
    except Exception as e:
        print(f"  ⚠️  Erro durante limpeza de logs: {e}")

def clean_tmux_logs():
    """Limpa logs antigos da pasta ~/logs criados pelo tmux"""
    try:
        home_logs_dir = os.path.expanduser("~/logs")
        if not os.path.exists(home_logs_dir):
            return
            
        deleted_count = 0
        for file_name in os.listdir(home_logs_dir):
            if file_name.startswith("experimentos-") and file_name.endswith(".log"):
                file_path = os.path.join(home_logs_dir, file_name)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"  🗑️  Log tmux antigo deletado: {file_name}")
                except Exception as e:
                    print(f"  ⚠️  Erro ao deletar log tmux {file_name}: {e}")
        
        if deleted_count > 0:
            print(f"  🧹 {deleted_count} arquivo(s) de log tmux antigo(s) deletado(s)")
            
    except Exception as e:
        print(f"  ⚠️  Erro durante limpeza de logs tmux: {e}")

def setup_log_files(experiment_id=None, prompt_variant=None, num_neighbors=None, model_name=None):
    """Inicializa os arquivos de log com timestamp no nome do arquivo
    Parâmetros opcionais permitem evitar leitura de variáveis globais em cenários com múltiplas threads.
    """
    # Limpa logs antigos primeiro
    clean_old_logs(experiment_id, prompt_variant, num_neighbors, model_name)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Simplifica nome do modelo para estrutura de diretórios
    model_simplified = get_simplified_model_name(model_name if model_name else MODEL)
    neighbors_for_dir = num_neighbors if num_neighbors is not None else NUM_NEIGHBORS
    
    # Cria diretórios na estrutura correta organizados por modelo e n_vizinhos
    base_dir = os.getenv('OUTPUT_BASE_DIR', 'resultados')
    logs_dir = os.path.join(os.path.dirname(__file__), base_dir, model_simplified, f'n{neighbors_for_dir}', 'logs')
    prompt_logs_dir = os.path.join(os.path.dirname(__file__), base_dir, model_simplified, f'n{neighbors_for_dir}', 'prompt_logs')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(prompt_logs_dir, exist_ok=True)
    
    # Se temos ID do experimento, usa ele no nome para facilitar identificação
    if experiment_id:
        exp_suffix = f"_exp{experiment_id}"
    else:
        exp_suffix = ""
    
    # Determina atributos a partir dos parâmetros ou das variáveis globais (fallback)
    variant_for_name = prompt_variant if prompt_variant else PROMPT_VARIANT
    neighbors_for_name = num_neighbors if num_neighbors is not None else NUM_NEIGHBORS
    model_for_name = (model_name if model_name else MODEL or 'unknown').replace('-', '_').replace('/', '_').replace(':', '_')

    # Arquivo de log regular com variante e número de vizinhos no nome
    log_path = os.path.join(logs_dir, f"respostas_log_{variant_for_name}_n{neighbors_for_name}{exp_suffix}_{timestamp}_{model_for_name}.txt")
    
    # Arquivo de log de prompts com variante e número de vizinhos no nome
    prompt_log_path = os.path.join(prompt_logs_dir, f"prompts_log_{variant_for_name}_n{neighbors_for_name}{exp_suffix}_{timestamp}_{model_for_name}.txt")
    
    # Cria arquivo de log regular com cabeçalho (substitui se existir)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"=== LOG DE RESPOSTAS DO EXPERIMENTO DE VIZINHANÇA - {timestamp} ===\n")
        if experiment_id:
            f.write(f"Experimento ID: {experiment_id}\n")
        f.write(f"Modelo: {(model_name if model_name else MODEL)}\n")
        f.write(f"Número de vizinhos: {neighbors_for_name}\n")
        f.write(f"Data e hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Cria arquivo de log de prompts com cabeçalho (substitui se existir)
    with open(prompt_log_path, 'w', encoding='utf-8') as f:
        f.write(f"=== LOG DE PROMPTS DO EXPERIMENTO DE VIZINHANÇA - {timestamp} ===\n")
        if experiment_id:
            f.write(f"Experimento ID: {experiment_id}\n")
        f.write(f"Modelo: {(model_name if model_name else MODEL)}\n")
        f.write(f"Número de vizinhos: {neighbors_for_name}\n")
        f.write(f"Data e hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    print(f"Arquivos de log criados em: {log_path} e {prompt_log_path}")
    return log_path, prompt_log_path

def log_response(neighbors: str, temperature: float, response_text: str, predicted_choice: str, attempt_number: int = 1, is_retry: bool = False):
    """
    Registra a resposta em arquivo
    
    Args:
        neighbors: Representação em string da configuração dos vizinhos
        temperature: Configuração de temperatura usada
        response_text: Resposta do LLM
        predicted_choice: Escolha detectada (k ou z)
        attempt_number: Número da tentativa atual
        is_retry: Indica se é uma nova tentativa devido a erro
    """
    log_file = get_current_log_file()
    if log_file is None:
        return
    
    retry_str = " [RETRY]" if is_retry else ""
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{retry_str}\n")
        f.write(f"Temperatura: {temperature}\n")
        f.write(f"Vizinhos: {neighbors}\n")
        f.write(f"Tentativa: {attempt_number}\n")
        f.write(f"Resposta completa: {response_text}\n")
        f.write(f"Escolha detectada: {predicted_choice}\n")
        f.write("-" * 50 + "\n")

def log_prompt(neighbors: str, temperature: float, system_prompt: str, user_prompt: str, response_text: str = None, predicted_choice: str = None, binary_config: str = None, attempt_number: int = 1, is_retry: bool = False):
    """
    Registra prompt em arquivo separado para registro completo
    
    Args:
        neighbors: Representação em string da configuração dos vizinhos
        temperature: Configuração de temperatura usada
        system_prompt: System prompt enviado ao LLM
        user_prompt: User prompt enviado ao LLM
        response_text: Texto de resposta do LLM (opcional)
        predicted_choice: Escolha detectada - k ou z (opcional)
        binary_config: Configuração binária completa (opcional)
        attempt_number: Número da tentativa atual
        is_retry: Indica se é uma nova tentativa devido a erro
    """
    prompt_log_file = get_current_prompt_log_file()
    if prompt_log_file is None:
        return
        
    retry_str = " [RETRY]" if is_retry else ""
    
    with open(prompt_log_file, 'a', encoding='utf-8') as f:
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{retry_str}\n")
        f.write(f"Temperatura: {temperature}\n")
        
        # Informações detalhadas sobre a vizinhança
        if binary_config:
            # USA binary_to_letters() para respeitar a variante (_kz, _ab, _01)
            try:
                variant = get_task_prompt_variant()
                letter_config_list = binary_to_letters(binary_config, variant)
                letter_config = ''.join(letter_config_list)
            except:
                # Fallback se não conseguir obter variante
                letter_config = ''.join([REVERSE_OPINION_MAP[int(bit)] for bit in binary_config])
            f.write(f"Configuração completa: {binary_config} ({letter_config})\n")
            
        f.write(f"Vizinhos: {neighbors}\n")
        f.write(f"Tentativa: {attempt_number}\n")
        
        # Registra resposta se disponível
        if response_text is not None:
            f.write(f"=== RESPOSTA DO LLM ===\n{response_text}\n")
            if predicted_choice is not None:
                f.write(f"Escolha detectada: {predicted_choice}\n")
            f.write("\n")
        
        f.write(f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n")
        f.write(f"=== USER PROMPT ===\n{user_prompt}\n")
        f.write("-" * 80 + "\n")



def log_token_performance(input_tokens: int, output_tokens: int, tempo_exec: float, neighbors_str: str):
    """
    Registra estatísticas de performance de tokens no formato do teste_chamada.py.
    
    Args:
        input_tokens: Número de tokens de entrada (real ou estimado)
        output_tokens: Número de tokens de saída (real ou estimado)
        tempo_exec: Tempo de execução em segundos
        neighbors_str: String de identificação da configuração
    """
    global TOKEN_STATS
    
    total_tokens = input_tokens + output_tokens
    tps = total_tokens / tempo_exec if tempo_exec > 0 else 0
    
    # Atualiza estatísticas globais
    TOKEN_STATS['total_tokens'] += total_tokens
    TOKEN_STATS['total_time'] += tempo_exec
    TOKEN_STATS['request_count'] += 1
    
    # Calcula média móvel
    if TOKEN_STATS['total_time'] > 0:
        TOKEN_STATS['running_avg_tps'] = TOKEN_STATS['total_tokens'] / TOKEN_STATS['total_time']
    
    # Log detalhado no arquivo no formato simplificado
    log_file = get_current_log_file()
    if log_file is not None:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[PERF] Input tokens: {input_tokens} | Output tokens: {output_tokens} | Tempo: {tempo_exec:.2f}s | TPS: {tps:.1f}\n")
            f.write(f"STATS: Total tokens: {TOKEN_STATS['total_tokens']}, Avg TPS: {TOKEN_STATS['running_avg_tps']:.1f}, Requests: {TOKEN_STATS['request_count']}\n")

async def log_token_performance_async(input_tokens: int, output_tokens: int, tempo_exec: float, neighbors_str: str):
    """
    Registra estatísticas de performance de tokens no formato do teste_chamada.py (versão assíncrona).
    
    Args:
        input_tokens: Número de tokens de entrada (real ou estimado)
        output_tokens: Número de tokens de saída (real ou estimado)
        tempo_exec: Tempo de execução em segundos
        neighbors_str: String de identificação da configuração
    """
    global TOKEN_STATS
    
    total_tokens = input_tokens + output_tokens
    tps = total_tokens / tempo_exec if tempo_exec > 0 else 0
    
    # Atualiza estatísticas globais com lock para thread safety
    async with get_token_stats_lock():
        TOKEN_STATS['total_tokens'] += total_tokens
        TOKEN_STATS['total_time'] += tempo_exec
        TOKEN_STATS['request_count'] += 1
        
        # Calcula média móvel
        if TOKEN_STATS['total_time'] > 0:
            TOKEN_STATS['running_avg_tps'] = TOKEN_STATS['total_tokens'] / TOKEN_STATS['total_time']
    
    # Log detalhado no arquivo no formato simplificado
    log_file = get_current_log_file()
    if log_file is not None:
        def write_perf_log():
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[PERF] Input tokens: {input_tokens} | Output tokens: {output_tokens} | Tempo: {tempo_exec:.2f}s | TPS: {tps:.1f}\n")
                f.write(f"STATS: Total tokens: {TOKEN_STATS['total_tokens']}, Avg TPS: {TOKEN_STATS['running_avg_tps']:.1f}, Requests: {TOKEN_STATS['request_count']}\n")
        await asyncio.to_thread(write_perf_log)

def parse_llm_response(response_text: str) -> Optional[str]:
    """
    Extrai a opinião da resposta do LLM.
    Aceita todos os tokens: k/z, a/b, 0/1, p/q, α/β, △/○, ⊕/⊖, ł/þ, yes/no
    Aceita respostas com espaços: [k], [ k ], etc.
    
    Args:
        response_text: Texto de resposta do modelo
        
    Returns:
        String com a opinião ou None se não encontrado
    """
    # Verifica se a resposta é exatamente um dos formatos aceitos
    response_clean = response_text.lower().strip()
    
    # Verifica formatos exatos (todos os tokens possíveis)
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
    
    # Como fallback, tenta encontrar qualquer um dos padrões em qualquer lugar da resposta
    # Aceita espaços opcionais dentro dos colchetes: [k], [ k ], [  k  ], etc.
    match = re.search(r'\[\s*(k|z|a|b|0|1|p|q|α|β|△|○|⊕|⊖|ł|þ|yes|no)\s*\]', response_clean)
    
    if match:
        return match.group(1)
    
    # Não foi possível extrair a opinião no formato correto
    return None

# FUNÇÃO REMOVIDA: Código síncrono legado removido - use query_agent() que é assíncrona

# ===============================================================================
# FUNÇÕES ASSÍNCRONAS PARA EXECUÇÃO COM ASYNCIO
# ===============================================================================

async def _create_completion_async(system_prompt: str, user_prompt: str, temperature: float, model: str):
    """Wrapper para chamada assíncrona ao LLM com fallback para cliente síncrono"""
    if is_sglang_backend():
        effective_user_prompt = _append_no_think_if_needed(user_prompt, model)
        tokenizer = await asyncio.to_thread(_get_sglang_tokenizer, model)
        full_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": effective_user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        variant = get_task_prompt_variant()
        sampling_params: Dict[str, Any] = {
            "temperature": float(temperature),
            "max_new_tokens": _env_int("LLM_MAX_NEW_TOKENS", 3000),
            "sampling_seed": _env_int("LLM_SEED", 42),
            "repetition_penalty": _env_float("LLM_REPEAT_PENALTY", 1.0),
            "stop_regex": os.getenv("SGLANG_STOP_REGEX", "").strip() or _make_sglang_stop_regex(variant),
            "no_stop_trim": True,
        }

        top_k = _env_int("LLM_TOP_K")
        if top_k is not None:
            sampling_params["top_k"] = top_k
        top_p = _env_float("LLM_TOP_P")
        if top_p is not None:
            sampling_params["top_p"] = top_p
        min_p = _env_float("LLM_MIN_P")
        if min_p is not None:
            sampling_params["min_p"] = min_p

        url = f"{BASE_URL_ROOT}/generate"
        payload = {
            "text": full_prompt,
            "sampling_params": sampling_params,
        }

        client_http = _get_native_http_client()
        resp = await client_http.post(url, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()

        content_text = data.get("text", "") if isinstance(data, dict) else ""
        if isinstance(content_text, str) and content_text.startswith(full_prompt):
            content_text = content_text[len(full_prompt):]

        meta = data.get("meta_info", {}) if isinstance(data, dict) and isinstance(data.get("meta_info"), dict) else {}
        prompt_tokens = int(meta.get("prompt_tokens", meta.get("prompt_token_count", 0)) or 0)
        completion_tokens = int(meta.get("completion_tokens", meta.get("completion_token_count", 0)) or 0)
        total_tokens = int(meta.get("total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))

        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        choice = SimpleNamespace(message=SimpleNamespace(content=content_text))
        return SimpleNamespace(choices=[choice], usage=usage, _native={"backend": "sglang", "raw": data, "logprobs": None})

    # ===================================================================
    # CÓDIGO ANTERIOR (OpenAI-compat) - COMENTADO
    # if async_client is not None:
    #     return await async_client.chat.completions.create(
    #         model=model,
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": user_prompt},
    #         ],
    #         temperature=temperature,
    #         seed=42,
    #     )
    # return await asyncio.to_thread(
    #     client.chat.completions.create,
    #     model=model,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ],
    #     temperature=temperature,
    #     seed=42,
    # )
    # ===================================================================

    # ===================================================================
    # CÓDIGO NATIVO LEGADO (COMENTADO)
    # ===================================================================
    # # API NATIVA LM STUDIO (/api/v1/chat)
    # # Monta input como string única (único formato aceito pelo native no servidor).
    # combined_input = f"SYSTEM:\\n{system_prompt}\\n\\nUSER:\\n{user_prompt}"
    # payload: Dict[str, Any] = {
    #     "model": model,
    #     "input": combined_input,
    #     "temperature": temperature,
    # }
    # client_http = _get_native_http_client()
    # url = f"{BASE_URL_ROOT}/api/v1/chat"
    # resp = await client_http.post(url, json=payload, headers={"Content-Type": "application/json"})
    # resp.raise_for_status()
    # data = resp.json()
    #
    # # Extrai conteúdo do formato nativo
    # content_text = ""
    # if isinstance(data.get("output"), list):
    #     for item in data["output"]:
    #         if isinstance(item, dict) and isinstance(item.get("content"), str):
    #             content_text = item.get("content", "")
    #             break
    # if not content_text:
    #     content_text = data.get("content", "") if isinstance(data.get("content"), str) else ""
    #
    # # Mapeia stats -> usage (compatível com o restante do fluxo)
    # stats = data.get("stats", {}) if isinstance(data.get("stats"), dict) else {}
    # prompt_tokens = int(stats.get("input_tokens", 0) or 0)
    # completion_tokens = int(stats.get("total_output_tokens", 0) or 0)
    # total_tokens = prompt_tokens + completion_tokens
    #
    # usage = SimpleNamespace(
    #     prompt_tokens=prompt_tokens,
    #     completion_tokens=completion_tokens,
    #     total_tokens=total_tokens,
    # )
    # choice = SimpleNamespace(message=SimpleNamespace(content=content_text))
    # return SimpleNamespace(choices=[choice], usage=usage, _native=data)

    # ===================================================================
    # METODO PARA FORCAR O DETERMINISMO + LOGPROBS (HTTP RAW /v1/responses)
    # ===================================================================
    url = f"{BASE_URL_ROOT}/v1/responses"

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "seed": _env_int("LLM_SEED", 42),
        "max_output_tokens": _env_int("LLM_MAX_OUTPUT_TOKENS", 50),
        "stream": False,
        "top_logprobs": LOGPROBS_TOP,
        "include": ["message.output_text.logprobs"],
    }
    payload = _apply_sampling_overrides(payload)
    
    client_http = _get_native_http_client()
    try:
        resp = await client_http.post(url, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()

        # Extrai texto no formato Responses
        content_text = ""
        if isinstance(data.get("output"), list):
            for item in data["output"]:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for c in item.get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        content_text = c.get("text", "") or ""
                        break
                if content_text:
                    break
        
        # Extração de uso (best-effort)
        usage_data = data.get("usage", {}) if isinstance(data.get("usage"), dict) else {}
        prompt_tokens = int(usage_data.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage_data.get("completion_tokens", 0) or 0)
        total_tokens = int(usage_data.get("total_tokens", 0) or (prompt_tokens + completion_tokens))

        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        choice = SimpleNamespace(message=SimpleNamespace(content=content_text))
        return SimpleNamespace(choices=[choice], usage=usage, _native=data)
        
    except Exception as e:
        # Fallback de segurança ou re-raise
        print(f"❌ Erro HTTP RAW Async: {e}")
        raise e

async def query_agent(
    strategy: PromptStrategy,
    neighbor_data_kwargs: Dict,
    temperature: float,
    current_iteration: int = 1,
    config_info: Dict = None,
    model_name: str = None,
    *,
    llm_gate: asyncio.Semaphore = None,
) -> Dict:
    """Consulta um agente LLM usando a estratégia de prompt fornecida (versão assíncrona).
    
    Args:
        strategy: Instância da estratégia de prompt a ser usada
        neighbor_data_kwargs: Dados dos vizinhos no formato esperado pela estratégia
        temperature: Temperatura para geração do LLM
        current_iteration: Iteração atual para identificação clara nos logs
        config_info: Informações da configuração atual (opcional, para melhor logging)
        model_name: Nome do modelo (opcional, para validação)
        llm_gate: Semáforo para controlar concorrência de chamadas ao LLM
    """
    # A construção do prompt agora é delegada para a estratégia
    system_prompt, user_prompt = strategy.build_prompt(**neighbor_data_kwargs)
    if is_sglang_backend():
        try:
            prompt_model_for_backend = model_name or get_task_model()
        except RuntimeError:
            prompt_model_for_backend = model_name or MODEL
        user_prompt = _append_no_think_if_needed(user_prompt, prompt_model_for_backend)
    
    # Cria uma representação em string dos vizinhos para registro
    if 'left' in neighbor_data_kwargs and 'right' in neighbor_data_kwargs:
        neighbors_str = f"Left: {neighbor_data_kwargs['left']}, Right: {neighbor_data_kwargs['right']}"
    else:
        neighbors_str = f"Strategy: {strategy.__class__.__name__}, Data: {neighbor_data_kwargs}"
    
    # Registra o prompt que vamos enviar APENAS UMA VEZ por iteração
    await asyncio.to_thread(log_prompt, neighbors_str, temperature, system_prompt, user_prompt, attempt_number=current_iteration)
    
    # Track retries for errors
    max_retries = 3
    retry_count = 0
    
    # For reexecution to get a valid response format
    max_response_attempts = int(os.getenv('MAX_RESPONSE_ATTEMPTS', '5'))
    response_attempt = 0
    
    while retry_count < max_retries:
        try:
            valid_response = False
            response_attempt = 0
            
            while not valid_response and response_attempt < max_response_attempts:
                # Print retry message if needed
                if response_attempt > 0:
                    print(f"⚠️ Format retry #{response_attempt} for neighbors: {neighbors_str}")
                
                # Mede tempo de inferência e calcula TPS
                start_time = time.time()
                
                # BINDING RÍGIDO: Sempre usar o modelo da tarefa, nunca fallback
                task_model = get_task_model()
                
                # AUDITORIA DE SEGURANÇA: Verificar consistência
                if model_name and model_name != task_model:
                    error_msg = f"❌ VIOLAÇÃO DE BINDING: model_name={model_name} != task_model={task_model}"
                    print(error_msg)
                    exp_id = CURRENT_EXPERIMENT_ID.get()
                    await asyncio.to_thread(log_prompt, f"EXP_{exp_id}_BINDING_ERROR", temperature, "ERROR", error_msg, attempt_number=current_iteration)
                    raise RuntimeError(error_msg)
                
                # Log de telemetria de segurança
                exp_id = CURRENT_EXPERIMENT_ID.get()
                print(f"🔍 EXP_{exp_id}: Usando modelo={task_model} (binding OK)")
                
                # Gate de concorrência por modelo (1 request por vez, por deploy)
                model_lock = get_model_request_lock(task_model)
                if llm_gate is not None:
                    async with model_lock:
                        async with llm_gate:
                            response = await _create_completion_async(system_prompt, user_prompt, temperature, task_model)
                else:
                    async with model_lock:
                        response = await _create_completion_async(system_prompt, user_prompt, temperature, task_model)
                
                end_time = time.time()
                
                response_text = response.choices[0].message.content.strip().lower()
                
                # Calcula métricas de performance usando dados reais da API
                tempo_exec = end_time - start_time
                
                # Usar dados reais de tokens da resposta da API
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                else:
                    # Se não houver dados de usage, definir como 0
                    input_tokens = 0
                    output_tokens = 0
                    total_tokens = 0

                
                tps = total_tokens / tempo_exec if tempo_exec > 0 else 0
                
                # Log performance no formato simples (assíncrono)
                await log_token_performance_async(input_tokens, output_tokens, tempo_exec, neighbors_str)
                
                # Print simplificado de performance igual ao teste_chamada.py
                # Exibe informações da configuração junto com as métricas
                if config_info:
                    config_display = f"{config_info['binary_config']} ({''.join(config_info['letter_config'])})"
                    print(f"    ✅ [PERF] {config_display} | In:{input_tokens} Out:{output_tokens} Tot:{total_tokens} | {tempo_exec:.2f}s | {tps:.1f} TPS")
                else:
                    print(f"    ✅ [PERF] In:{input_tokens} Out:{output_tokens} Tot:{total_tokens} | {tempo_exec:.2f}s | {tps:.1f} TPS")
                
                predicted_choice = parse_llm_response(response_text)
                
                # Log all responses, including invalid ones
                await asyncio.to_thread(log_response, neighbors_str, temperature, response_text, predicted_choice, attempt_number=current_iteration)
                
                # Log apenas a resposta no arquivo de prompt (sem repetir o prompt)
                prompt_log_file = get_current_prompt_log_file()
                if prompt_log_file is not None:
                    await asyncio.to_thread(
                        lambda: open(prompt_log_file, 'a', encoding='utf-8').write(
                            f"=== RESPOSTA DO LLM (Tentativa {response_attempt+1}) ===\n{response_text}\n"
                            f"Escolha detectada: {predicted_choice}\n\n" if predicted_choice is not None
                            else "Escolha detectada: None\n\n"
                        )
                    )
                
                # Check if we have a valid response in correct format
                if predicted_choice in ['k', 'z', 'a', 'b', '0', '1', 'p', 'q', 'α', 'β', '△', '○', '⊕', '⊖', 'ł', 'þ', 'yes', 'no']:
                    valid_response = True
                    # Registro especial para respostas válidas
                    log_file = get_current_log_file()
                    if log_file is not None:
                        await asyncio.to_thread(
                            lambda: open(log_file, 'a', encoding='utf-8').write(
                                f"RESPOSTA VÁLIDA após {response_attempt+1} tentativas\n" +
                                "-" * 50 + "\n"
                            )
                        )
                else:
                    # Registra como tentativa inválida
                    log_file = get_current_log_file()
                    if log_file is not None:
                        await asyncio.to_thread(
                            lambda: open(log_file, 'a', encoding='utf-8').write(
                                f"FORMATO INVÁLIDO (tentativa {response_attempt+1}): Resposta não contém token válido entre colchetes\n" +
                                "-" * 50 + "\n"
                            )
                        )
                    response_attempt += 1
            
            # Se esgotamos todas as tentativas de resposta e ainda não temos uma resposta válida
            if not valid_response:
                print(f"❌ Falhou ao obter resposta válida após {max_response_attempts} tentativas para vizinhos: {neighbors_str}")
                log_file = get_current_log_file()
                if log_file is not None:
                    await asyncio.to_thread(
                        lambda: open(log_file, 'a', encoding='utf-8').write(
                            f"AVISO: Falha ao obter resposta válida após {max_response_attempts} tentativas\n" +
                            "-" * 50 + "\n"
                        )
                    )
            
            return {
                "neighbors": neighbors_str,
                "response": response_text,
                "predicted_choice": predicted_choice,
                "binary_choice": OPINION_MAP.get(predicted_choice, None) if predicted_choice else None,
                "full_prompt": f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}",
                "full_response": response_text,
                "logprobs_json": json.dumps(
                    _extract_output_text_logprobs_from_responses_native(getattr(response, "_native", {})),
                    ensure_ascii=False,
                ),
            }
            
        except Exception as e:
            retry_count += 1
            wait_time = 2 ** retry_count
            print(f"🔄 API ERROR RETRY #{retry_count}/{max_retries} for neighbors: {neighbors_str}")
            print(f"   Error: {str(e)}. Retrying in {wait_time:.2f}s...")
            
            # Log prompt and error for this retry attempt
            prompt_log_file = get_current_prompt_log_file()
            if prompt_log_file is not None:
                await asyncio.to_thread(
                    lambda: open(prompt_log_file, 'a', encoding='utf-8').write(
                        f"=== ERRO [RETRY] ===\n{str(e)}\n" +
                        f"Tentativa: {current_iteration}\n\n"
                    )
                )
            
            await asyncio.to_thread(log_response, neighbors_str, temperature, f"ERROR: {str(e)}", None, 
                        attempt_number=current_iteration, is_retry=True)
            
            await asyncio.sleep(wait_time)
    
    # If all retries fail
    print(f"❌❌ TODAS AS TENTATIVAS FALHARAM para vizinhos: {neighbors_str}")
    return {
        "neighbors": neighbors_str,
        "response": "ERRO",
        "predicted_choice": None,
        "binary_choice": None,
        "full_prompt": f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}",
        "full_response": "ERRO"
    }

def binary_to_letters(binary_config: str, variant: str = None) -> List[str]:
    """Converte configuração binária para opiniões baseado na variante do prompt"""
    if variant is None:
        # Tenta primeiro do contexto async, depois fallback para global
        try:
            variant = get_task_prompt_variant()
        except:
            variant = PROMPT_VARIANT
    
    # Mapeia variantes para seus símbolos correspondentes
    variant_symbols = {
        # Variantes k/z (padrão) - mantendo compatibilidade com versões sem sufixo
        'v20_lista_completa_meio_raciocinio_primeiro': ('k', 'z'),
        'v9_lista_completa_meio': ('k', 'z'),
        'v21_zero_shot_cot': ('k', 'z'),
        'v22_plan_solve': ('k', 'z'),
        'v23_least_to_most': ('k', 'z'),
        'v24_self_consistency': ('k', 'z'),
        'v25_self_ask': ('k', 'z'),
        # Variantes k/z (com sufixo _kz para consistência)
        'v20_lista_completa_meio_raciocinio_primeiro_kz': ('k', 'z'),
        'v9_lista_completa_meio_kz': ('k', 'z'),
        'v21_zero_shot_cot_kz': ('k', 'z'),
        # Variantes A/B
        'v20_lista_completa_meio_raciocinio_primeiro_ab': ('a', 'b'),
        'v9_lista_completa_meio_ab': ('a', 'b'),
        'v21_zero_shot_cot_ab': ('a', 'b'),
        'v22_plan_solve_ab': ('a', 'b'),
        'v23_least_to_most_ab': ('a', 'b'),
        'v24_self_consistency_ab': ('a', 'b'),
        'v25_self_ask_ab': ('a', 'b'),
        # Variantes 0/1
        'v20_lista_completa_meio_raciocinio_primeiro_01': ('0', '1'),
        'v9_lista_completa_meio_01': ('0', '1'),
        'v21_zero_shot_cot_01': ('0', '1'),
        'v22_plan_solve_01': ('0', '1'),
        'v23_least_to_most_01': ('0', '1'),
        'v24_self_consistency_01': ('0', '1'),
        'v25_self_ask_01': ('0', '1'),
        # TIER 1 - Grego (α/β)
        'v9_lista_completa_meio_αβ': ('α', 'β'),
        'v20_lista_completa_meio_raciocinio_primeiro_αβ': ('α', 'β'),
        'v21_zero_shot_cot_αβ': ('α', 'β'),
        # TIER 1 - Geométrico (△/○)
        'v9_lista_completa_meio_△○': ('△', '○'),
        'v20_lista_completa_meio_raciocinio_primeiro_△○': ('△', '○'),
        'v21_zero_shot_cot_△○': ('△', '○'),
        # TIER 2 - Matemático (⊕/⊖)
        'v9_lista_completa_meio_⊕⊖': ('⊕', '⊖'),
        'v20_lista_completa_meio_raciocinio_primeiro_⊕⊖': ('⊕', '⊖'),
        'v21_zero_shot_cot_⊕⊖': ('⊕', '⊖'),
        # TIER 2 - Latino médio (p/q)
        'v9_lista_completa_meio_pq': ('p', 'q'),
        'v20_lista_completa_meio_raciocinio_primeiro_pq': ('p', 'q'),
        'v21_zero_shot_cot_pq': ('p', 'q'),
        # TIER 3 - Latino estendido (ł/þ)
        'v9_lista_completa_meio_łþ': ('ł', 'þ'),
        'v20_lista_completa_meio_raciocinio_primeiro_łþ': ('ł', 'þ'),
        'v21_zero_shot_cot_łþ': ('ł', 'þ'),
        # YES/NO
        'v9_lista_completa_meio_yesno': ('no', 'yes'),
        'v21_zero_shot_cot_yesno': ('no', 'yes'),
    }
    
    # Pega os símbolos para a variante atual, fallback para k/z
    symbols = variant_symbols.get(variant, ('k', 'z'))
    
    return [symbols[int(bit)] for bit in binary_config]


def _extract_output_text_logprobs_from_responses_native(native: Dict[str, Any]) -> Any:
    """
    Extract `output_text.logprobs` from a LM Studio /v1/responses response.
    Returns the raw structure (usually a list[dict]) or None.
    """
    if not isinstance(native, dict):
        return None
    out = native.get("output")
    if not isinstance(out, list):
        return None
    for item in out:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "output_text":
                return c.get("logprobs")
    return None

def count_ones(binary_string: str) -> int:
    """Conta o número de '1's em uma string binária"""
    return binary_string.count('1')

def generate_neighbor_configs(total_neighbors: int):
    """
    Gera todas as possíveis configurações de vizinhos para o número total de vizinhos.
    
    Args:
        total_neighbors: Número total de vizinhos (serão divididos entre 'before' e 'after')
        
    Returns:
        Lista de configurações binárias
    """
    return [format(i, f'0{total_neighbors}b') for i in range(2**total_neighbors)]

# FUNÇÃO REMOVIDA: Código síncrono legado removido - use run_experiment() que é assíncrona

async def run_experiment(temperature: float, model_name: str = None, num_neighbors: int = None):
    """
    Executa o experimento para uma temperatura específica (versão assíncrona).
    
    Usa as constantes globais:
    - NUM_NEIGHBORS: Número TOTAL de vizinhos no experimento (sempre ímpar)
    - TOTAL_PARTICIPANTS: Número total de participantes no experimento completo
    
    A configuração usa uma abordagem de anel onde o agente LLM é o centro da vizinhança,
    com vizinhos distribuídos igualmente antes e depois dele.
    
    IMPORTANTE: O agente central (LLM) não conhece sua própria opinião e decide apenas
    com base nas opiniões dos vizinhos ao seu redor.
    
    Returns:
        tuple: (averages, raw_data, extended_raw_data) onde:
            - averages: Um dicionário com a média de escolhas para cada número de 1's
            - raw_data: Uma lista de dicionários com os dados brutos de cada iteração
            - extended_raw_data: Uma lista de dicionários com dados estendidos (incluindo prompts)
    """
    # Usa parâmetro ou fallback para global
    effective_neighbors = num_neighbors if num_neighbors is not None else NUM_NEIGHBORS
    
    print(f"\nRunning async experiment with temp: {temperature}, prompt variant: {get_task_prompt_variant()}")
    print(f"Using {effective_neighbors} neighbors for this experiment")

    # Gate global: evita requests paralelos entre workers de modelos diferentes.
    llm_gate = get_global_llm_gate()

    # Pega a estratégia UMA VEZ no início do experimento
    try:
        strategy = get_prompt_strategy(get_task_prompt_variant())
    except (ValueError, NotImplementedError) as e:
        print(f"❌ Erro ao carregar estratégia de prompt: {e}")
        return {}, [], []  # Retorna vazio para não quebrar a execução
    
    # Calculamos o número de vizinhos para cada lado (floor division para garantir inteiros)
    neighbors_per_side = effective_neighbors // 2
    
    # Geramos configurações para o total de vizinhos
    binary_configs = generate_neighbor_configs(effective_neighbors)
    max_ones = effective_neighbors  # Número máximo de uns possível
    
    # Resultados serão organizados por número de 1's na configuração
    results_by_ones = {i: [] for i in range(max_ones + 1)}  # 0 até max_ones
    
    # Armazenar a configuração específica usada para cada escolha
    # A chave é (num_ones, índice da escolha), o valor é a configuração binária
    config_mapping = {}
    
    # Armazenar resultados por configuração para calcular maioria
    # A chave é binary_config, o valor é uma lista de escolhas para essa configuração
    results_by_config = {}
    
    # Armazenar dados completos das respostas (incluindo prompts)
    # A chave é binary_config, o valor é uma lista de dados completos das respostas
    config_responses = {}
    
    # Variáveis para estimativa de tempo
    experiment_start_time = time.time()
    total_configs = len(binary_configs)
    total_calls = total_configs * NUM_ITERATIONS
    
    # Pré-processa todas as configurações para evitar recálculos
    config_data = []
    for config_index, binary_config in enumerate(binary_configs):
        letter_config = binary_to_letters(binary_config, get_task_prompt_variant())
        num_ones = count_ones(binary_config)
        
        # Lógica "Agente no Centro": o LLM está no meio da fila de vizinhos.
        # O índice do elemento central é ignorado ao construir o prompt.
        middle_index = effective_neighbors // 2

        # Os vizinhos 'left' são todos os elementos ANTES do índice do meio.
        left_neighbors = letter_config[:middle_index]

        # Os vizinhos 'right' são todos os elementos DEPOIS do índice do meio.
        right_neighbors = letter_config[middle_index + 1:]
        
        # Determina a opinião ATUAL REAL do agente a partir da configuração
        agent_opinion_real = letter_config[middle_index]

        # A contagem de vizinhos é atualizada para refletir a nova divisão.
        left_count = len(left_neighbors)
        right_count = len(right_neighbors)
        
        # Armazena os dados da configuração
        config_data.append({
            'config_index': config_index,
            'binary_config': binary_config,
            'letter_config': letter_config,
            'num_ones': num_ones,
            'left_neighbors': left_neighbors,
            'right_neighbors': right_neighbors,
            'agent_opinion_real': agent_opinion_real,
            'left_count': left_count,
            'right_count': right_count,
            'middle_index': middle_index
        })
    
    print(f"\n🚀 PREPARAÇÃO DO EXPERIMENTO ASSÍNCRONO:")
    print(f"   🎯 Configurações geradas: {total_configs}")
    print(f"   🔄 Iterações por config: {NUM_ITERATIONS}")
    print(f"   📞 Total de calls ao LLM: {total_calls}")
    print(f"   🔒 Estratégia de concorrência: Semáforo(1) por experimento")
    
    print(f"\n🔄 EXECUTANDO {NUM_ITERATIONS} ITERAÇÕES ASSÍNCRONAS PARA EVITAR CACHE")
    
    # Executa por iteração para evitar cache (mudança principal)
    call_count = 0
    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"\n=== ITERAÇÃO ASSÍNCRONA {iteration}/{NUM_ITERATIONS} ===")
        iteration_start_time = time.time()
        
        for config_info in config_data:
            call_count += 1
            
            # Extrai os dados pré-processados
            config_index = config_info['config_index']
            binary_config = config_info['binary_config']
            
            letter_config = config_info['letter_config']
            num_ones = config_info['num_ones']
            left_neighbors = config_info['left_neighbors']
            right_neighbors = config_info['right_neighbors']
            agent_opinion_real = config_info['agent_opinion_real']
            left_count = config_info['left_count']
            right_count = config_info['right_count']
            middle_index = config_info['middle_index']
            
            # Mostra qual configuração está sendo processada
            print(f"  🔧 [Async Iter {iteration}] Config {config_index+1}/{total_configs}: {binary_config} ({''.join(letter_config)}) | Ones: {num_ones} | Left: {''.join(left_neighbors)} | Right: {''.join(right_neighbors)}")
            
            # A cada 10 chamadas dentro da iteração, mostra estimativa de tempo restante total
            calls_in_current_iteration = config_index + 1
            if calls_in_current_iteration % 10 == 0:
                elapsed_so_far = time.time() - experiment_start_time
                avg_time_per_call = elapsed_so_far / call_count if call_count > 0 else 0
                
                # Calcula quantas chamadas restam no experimento total
                remaining_calls_total = total_calls - call_count
                estimated_remaining_total = remaining_calls_total * avg_time_per_call
                
                print(f"    ⏰ [Async Checkpoint] {calls_in_current_iteration}/{total_configs} configs nesta iteração | {call_count}/{total_calls} calls totais")
                print(f"       Tempo restante estimado: {estimated_remaining_total/60:.1f}min | Velocidade: {avg_time_per_call:.2f}s/call")
            
            # Progresso geral a cada 50 calls (mantido para iterações longas)
            elif call_count % 50 == 0:
                elapsed_so_far = time.time() - experiment_start_time
                avg_time_per_call = elapsed_so_far / call_count if call_count > 0 else 0
                estimated_total_time = avg_time_per_call * total_calls
                estimated_remaining = estimated_total_time - elapsed_so_far
                
                print(f"  📈 Progresso Geral Assíncrono: {call_count}/{total_calls} calls ({call_count/total_calls*100:.1f}%) | Tempo médio: {avg_time_per_call:.2f}s/call | Restante: {estimated_remaining/60:.1f}min")
            
            # --- Ponto chave: Preparar os dados para a estratégia atual ---
            # A lógica de como montar os dados agora fica aqui,
            # mantendo a estratégia agnóstica.
            
            data_for_strategy = {}
            current_variant = get_task_prompt_variant()
            if current_variant == 'v5_original':
                data_for_strategy = {
                    "left": left_neighbors,
                    "right": right_neighbors,
                    "current_opinion": agent_opinion_real
                }
            elif current_variant == 'v6_lista_indices':
                # Para v6, precisamos criar uma lista completa de vizinhos incluindo o agente no meio
                # Usa a opinião real do agente a partir da configuração
                
                # Constrói a lista completa com o agente no meio
                full_neighborhood = left_neighbors.copy()
                full_neighborhood.append(agent_opinion_real)  # Insere o agente no meio
                full_neighborhood.extend(right_neighbors)
                
                data_for_strategy = {
                    "neighborhood": full_neighborhood,
                    "position": middle_index
                }
            elif current_variant == 'v7_offsets':
                # Para v7, usamos o mesmo formato que v5 (left/right)
                data_for_strategy = {
                    "left": left_neighbors,
                    "right": right_neighbors,
                    "current_opinion": agent_opinion_real
                }
            elif current_variant == 'v8_visual':
                # Para v8, usamos o mesmo formato que v5 (left/right)
                data_for_strategy = {
                    "left": left_neighbors,
                    "right": right_neighbors,
                    "current_opinion": agent_opinion_real
                }
            elif current_variant == 'v9_lista_completa_meio':
                # Para v9, usamos left/right e passamos current_opinion real
                data_for_strategy = {
                    "left": left_neighbors,
                    "right": right_neighbors,
                    "current_opinion": agent_opinion_real
                }
            elif current_variant == 'v10_lista_indice_especifico':
                # Para v10, usamos left/right e passamos current_opinion real
                data_for_strategy = {
                    "left": left_neighbors,
                    "right": right_neighbors,
                    "current_opinion": agent_opinion_real
                }
            elif current_variant == 'v12_python':
                # Para v12_python, usamos o mesmo formato que v6_lista_indices 
                # (lista completa com agente no meio + posição)
                
                # Constrói a lista completa com o agente no meio
                full_neighborhood = left_neighbors.copy()
                full_neighborhood.append(agent_opinion_real)  # Insere o agente no meio
                full_neighborhood.extend(right_neighbors)
                
                data_for_strategy = {
                    "neighborhood": full_neighborhood,
                    "position": middle_index
                }
            elif current_variant == 'v20_lista_completa_meio_raciocinio_primeiro':
                # Para v20, usamos o mesmo formato que v19 (left/right e current_opinion)
                data_for_strategy = {
                    "left": left_neighbors,
                    "right": right_neighbors,
                    "current_opinion": agent_opinion_real
                }
            else:
                # Fallback para formato v5_original
                data_for_strategy = {
                    "left": left_neighbors,
                    "right": right_neighbors,
                    "current_opinion": agent_opinion_real
                }
            
            # Registra a configuração binária completa para referência no prompt log (só na primeira iteração)
            prompt_log_file = get_current_prompt_log_file()
            if iteration == 1 and prompt_log_file is not None:
                await asyncio.to_thread(
                    lambda: open(prompt_log_file, 'a', encoding='utf-8').write(
                        f"\n=== NOVA CONFIGURAÇÃO ===\n" +
                        f"Configuração binária completa: {binary_config} ({''.join(letter_config)})\n" +
                        f"Posição central (agente LLM): {letter_config[middle_index]} na posição {middle_index}\n" +
                        f"Vizinhos à esquerda: {''.join(left_neighbors)}, Vizinhos à direita: {''.join(right_neighbors)}\n" +
                        "-" * 80 + "\n"
                    )
                )
            
            result = await query_agent(strategy, data_for_strategy, temperature, iteration, config_info, model_name, llm_gate=llm_gate)
            
            # Store the binary choice (0 for 'k', 1 for 'z')
            if result["binary_choice"] is not None:
                # Inicializa a lista se não existir
                if binary_config not in results_by_config:
                    results_by_config[binary_config] = []
                
                # Adiciona à lista de escolhas desta configuração específica
                results_by_config[binary_config].append(result["binary_choice"])
                
                # Armazenar o índice desta escolha (mantido para compatibilidade com dados brutos)
                if num_ones not in config_mapping:
                    config_mapping[num_ones] = []
                config_mapping[num_ones].append(binary_config)
            
            # Armazenar dados completos da resposta (incluindo prompts)
            if binary_config not in config_responses:
                config_responses[binary_config] = []
            config_responses[binary_config].append(result)
            
            if result["binary_choice"] is None:
                # Log quando uma escolha não for válida
                log_file = get_current_log_file()
                if log_file is not None:
                    await asyncio.to_thread(
                        lambda: open(log_file, 'a', encoding='utf-8').write(
                            f"WARNING: Async Iteration {iteration} config {binary_config} produced invalid choice\n" +
                            "-" * 50 + "\n"
                        )
                    )
        
        # Relatório de progresso da iteração
        iteration_elapsed = time.time() - iteration_start_time
        iteration_avg_per_call = iteration_elapsed / total_configs
        calls_this_iteration = total_configs
        
        print(f"✅ Iteração Assíncrona {iteration} completada:")
        print(f"   ⏱️  Tempo: {iteration_elapsed:.2f}s ({iteration_elapsed/60:.1f}min)")
        print(f"   Calls: {calls_this_iteration} configs")
        print(f"   🚀 Velocidade: {iteration_avg_per_call:.2f}s/config")
        
        # Estatísticas globais atualizadas
        total_elapsed_so_far = time.time() - experiment_start_time
        global_avg_per_call = total_elapsed_so_far / call_count if call_count > 0 else 0
        
        # Estimativa de tempo restante baseada na performance real
        if iteration < NUM_ITERATIONS:
            remaining_iterations = NUM_ITERATIONS - iteration
            remaining_calls = remaining_iterations * total_configs
            estimated_remaining = remaining_calls * global_avg_per_call
            
            print(f"   📈 Global: {call_count}/{total_calls} calls totais ({call_count/total_calls*100:.1f}%)")
            print(f"   ⏳ Estimativa restante: {estimated_remaining/60:.1f}min ({remaining_iterations} iterações)")
        else:
            print(f"   🎉 EXPERIMENTO ASSÍNCRONO CONCLUÍDO! Total: {call_count} calls em {total_elapsed_so_far/60:.1f}min")
    
    # Após todas as iterações, processa os resultados
    print(f"\nPROCESSANDO RESULTADOS ASSÍNCRONOS...")
    
    for binary_config, config_choices in results_by_config.items():
        num_ones = count_ones(binary_config)
        
        if config_choices:
            # Calcula a escolha majoritária para esta configuração específica
            majority_choice = 1 if sum(config_choices) > len(config_choices) / 2 else 0
            
            # Adiciona a escolha majoritária aos resultados por número de 1's
            results_by_ones[num_ones].append(majority_choice)
            
            print(f"    Config {binary_config}: choices={config_choices}, majority={majority_choice}")
        else:
            print(f"    Config {binary_config}: No valid choices")
    
    # Log final com tempo total
    total_elapsed = time.time() - experiment_start_time
    log_file = get_current_log_file()
    if log_file is not None:
        await asyncio.to_thread(
            lambda: open(log_file, 'a', encoding='utf-8').write(
                f"\n=== EXPERIMENTO ASSÍNCRONO CONCLUÍDO ===\n" +
                f"Total de configurações: {total_configs}\n" +
                f"Total de iterações: {NUM_ITERATIONS}\n" +
                f"Total de chamadas: {total_calls}\n" +
                f"Tempo total: {total_elapsed:.2f} segundos ({total_elapsed/60:.1f} minutos)\n" +
                f"Tempo médio por chamada: {total_elapsed/total_calls:.2f} segundos\n" +
                "-" * 50 + "\n"
            )
        )
    
    # Preparar dados brutos para retornar para a função main
    raw_data = []
    extended_raw_data = []  # Dados estendidos com prompts e respostas
    
    # Organiza os dados brutos em formato tabular usando as escolhas individuais por configuração
    # Use config_responses as source of truth to include instrumentation fields (e.g., logprobs).
    for binary_config, responses_list in config_responses.items():
        num_ones = count_ones(binary_config)
        # USA binary_to_letters() para respeitar a variante (_kz, _ab, _01)
        config_letras_list = binary_to_letters(binary_config, get_task_prompt_variant())
        config_letras = ''.join(config_letras_list)
        
        for response_data in responses_list:
            choice = response_data.get("binary_choice", None)
            if choice is None:
                continue
            raw_data.append({
                'temperatura': temperature,
                'configuracao_binaria': binary_config,
                'configuracao_letras': config_letras,
                'num_ones': num_ones,
                'escolha': choice,
                'logprobs_json': response_data.get('logprobs_json', '')
            })
    
    # Organiza os dados estendidos com prompts e respostas
    for binary_config, config_data in config_responses.items():
        num_ones = count_ones(binary_config)
        # USA binary_to_letters() para respeitar a variante (_kz, _ab, _01)
        config_letras_list = binary_to_letters(binary_config, get_task_prompt_variant())
        config_letras = ''.join(config_letras_list)
        
        for response_data in config_data:
            extended_raw_data.append({
                'temperatura': temperature,
                'configuracao_binaria': binary_config,
                'configuracao_letras': config_letras,
                'num_ones': num_ones,
                'escolha': response_data.get('binary_choice', None),
                'prompt_input': response_data.get('full_prompt', ''),
                'llm_response': response_data.get('full_response', '')
            })
    
    # Calcula média para cada número de 1's baseada nas escolhas majoritárias
    averages = {}
    for num_ones, majority_choices in results_by_ones.items():
        if majority_choices:
            avg = np.mean(majority_choices)
            averages[num_ones] = avg
            print(f"  {num_ones} 1's: {len(majority_choices)} configs, majority-based average = {avg:.4f}")
        else:
            print(f"AVISO: Nenhuma resposta válida para configuração com {num_ones} 1's")
            averages[num_ones] = None
    
    return averages, raw_data, extended_raw_data

def get_simplified_model_name(model_name: str) -> str:
    """
    Simplifica o nome do modelo para uso em diretórios.
    Remove sufixos de deploy (:2, :3, etc) para agrupar resultados do mesmo modelo.
    
    Exemplos:
        google/gemma-3-4b -> gemma4b
        google/gemma-3-4b:2 -> gemma4b
        meta-llama-3.1-8b-instruct -> llama8b
        meta-llama-3.1-8b-instruct:3 -> llama8b
        gemma-3-27b-it -> gemma27b
    """
    if not model_name:
        return 'unknown'
    
    # Remove sufixos de deploy (:2, :3, etc) antes de processar
    model_base = model_name.split(':')[0]
    model_lower = model_base.lower()
    
    # Mapeamento de modelos conhecidos
    if 'gemma' in model_lower and '4b' in model_lower:
        return 'gemma4b'
    elif 'gemma' in model_lower and '12b' in model_lower:
        return 'gemma12b'
    elif 'gemma' in model_lower and '27b' in model_lower:
        return 'gemma27b'
    elif 'llama' in model_lower and '8b' in model_lower:
        return 'llama8b'
    elif 'llama' in model_lower and '70b' in model_lower:
        return 'llama70b'
    elif 'qwen' in model_lower and '4b' in model_lower:
        return 'qwen4b'
    elif 'qwen' in model_lower and '14b' in model_lower:
        return 'qwen14b'
    elif 'qwen' in model_lower and '8b' in model_lower:
        return 'qwen8b'
    else:
        # Fallback: remove caracteres especiais e mantém nome base
        return model_base.replace('/', '_').replace(':', '_').replace('-', '_').replace('.', '_')

def save_combined_csv(all_raw_data: List[Dict], num_neighbors: int, variant: str = None, experiment_id: int = None, model_name: str = None):
    """
    Salva todos os dados brutos combinados em um único arquivo CSV.
    
    Args:
        all_raw_data: Lista de dicionários com todos os dados brutos de todas as temperaturas
        num_neighbors: Número de vizinhos usado no experimento
        variant: Variante específica do prompt (opcional)
        experiment_id: ID do experimento (opcional, para incluir no nome)
        model_name: Nome do modelo executado (opcional)
    
    Returns:
        str: Caminho do arquivo CSV salvo
    """
    # Simplifica nome do modelo para estrutura de diretórios
    model_simplified = get_simplified_model_name(model_name if model_name else MODEL)
    
    # Cria diretório para resultados organizados por modelo e n_vizinhos
    base_dir = os.getenv('OUTPUT_BASE_DIR', 'resultados')
    results_dir = os.path.join(os.path.dirname(__file__), base_dir, model_simplified, f'n{num_neighbors}', 'csv', 'basic')
    os.makedirs(results_dir, exist_ok=True)
    
    # Cria nome do arquivo com timestamp e variante
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Usa PROMPT_VARIANT se variant não foi fornecido
    variant_name = variant if variant is not None else PROMPT_VARIANT
    
    # ID do experimento
    exp_suffix = f"_exp{experiment_id}" if experiment_id else ""
    
    # Inclui o modelo no nome do arquivo para diferenciação
    model_safe = (model_name if model_name else MODEL or 'unknown').replace('/', '_').replace(':', '_')
    
    # PADRÃO: {tipo}_{variant}_n{neighbors}_exp{id}_iter{iterations}_{timestamp}_{model}.csv
    csv_filename = os.path.join(results_dir, f"dados_combinados_{variant_name}_n{num_neighbors}{exp_suffix}_iter{NUM_ITERATIONS}_{timestamp}_{model_safe}.csv")
    
    # Define o cabeçalho do CSV
    fieldnames = [
        'temperatura', 
        'configuracao_binaria', 
        'configuracao_letras',
        'num_ones', 
        'escolha',
        'logprobs_json',
    ]
    
    # Adiciona coluna de variante se existir nos dados
    if all_raw_data and 'variante' in all_raw_data[0]:
        fieldnames.insert(0, 'variante')
    
    # Escreve os dados no arquivo CSV
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_raw_data:
            writer.writerow(row)
    
    return csv_filename

def save_extended_csv(all_extended_data: List[Dict], num_neighbors: int, variant: str = None, experiment_id: int = None, model_name: str = None):
    """
    Salva dados estendidos com prompts e respostas em um arquivo CSV.
    
    Args:
        all_extended_data: Lista de dicionários com dados estendidos (incluindo prompts e respostas)
        num_neighbors: Número de vizinhos usado no experimento
        variant: Variante específica do prompt (opcional)
        experiment_id: ID do experimento (opcional, para incluir no nome)
        model_name: Nome do modelo executado (opcional)
    
    Returns:
        str: Caminho do arquivo CSV salvo
    """
    # Simplifica nome do modelo para estrutura de diretórios
    model_simplified = get_simplified_model_name(model_name if model_name else MODEL)
    
    # Cria diretório para resultados estendidos organizados por modelo e n_vizinhos
    base_dir = os.getenv('OUTPUT_BASE_DIR', 'resultados')
    results_dir = os.path.join(os.path.dirname(__file__), base_dir, model_simplified, f'n{num_neighbors}', 'csv', 'extended')
    os.makedirs(results_dir, exist_ok=True)
    
    # Cria nome do arquivo com timestamp e variante
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Usa PROMPT_VARIANT se variant não foi fornecido
    variant_name = variant if variant is not None else PROMPT_VARIANT
    
    # ID do experimento
    exp_suffix = f"_exp{experiment_id}" if experiment_id else ""
    
    # Inclui o modelo no nome do arquivo para diferenciação
    model_safe = (model_name if model_name else MODEL or 'unknown').replace('/', '_').replace(':', '_')
    
    # PADRÃO: {tipo}_{variant}_n{neighbors}_exp{id}_iter{iterations}_{timestamp}_{model}.csv
    csv_filename = os.path.join(results_dir, f"dados_estendidos_{variant_name}_n{num_neighbors}{exp_suffix}_iter{NUM_ITERATIONS}_{timestamp}_{model_safe}.csv")
    
    # Define o cabeçalho do CSV estendido
    fieldnames = [
        'temperatura', 
        'configuracao_binaria', 
        'configuracao_letras',
        'num_ones', 
        'escolha',
        'prompt_input',
        'llm_response'
    ]
    
    # Adiciona coluna de variante se existir nos dados
    if all_extended_data and 'variante' in all_extended_data[0]:
        fieldnames.insert(0, 'variante')
    
    # Escreve os dados no arquivo CSV
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_extended_data:
            writer.writerow(row)
    
    return csv_filename

def plot_results(all_results: Dict[float, Dict[int, float]], num_neighbors: int = None, variant: str = None, model_name: str = None, experiment_id: int = None):
    """Plota os resultados para todas as temperaturas
    
    Args:
        all_results: Dicionário com resultados por temperatura
        num_neighbors: Número de vizinhos (específico do experimento)
        variant: Variante do prompt (específica do experimento)
        model_name: Nome do modelo (específico do experimento)
        experiment_id: ID do experimento (para incluir no nome)
    """
    try:
        # Usa valores específicos ou fallback para globais
        effective_neighbors = num_neighbors if num_neighbors is not None else NUM_NEIGHBORS
        effective_variant = variant if variant is not None else PROMPT_VARIANT
        effective_model = model_name if model_name is not None else MODEL
        
        print(f"🎨 Iniciando geração de plot: neighbors={effective_neighbors}, variant={effective_variant}")
        
        plt.figure(figsize=(10, 6))
        
        # Cores para diferentes temperaturas
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        plot_data_found = False
        
        for i, (temperature, results) in enumerate(all_results.items()):
            print(f"  Processando temperatura {temperature}: {len(results)} pontos de dados")
            
            # Filtra valores None para plotagem
            valid_results = {k: v for k, v in results.items() if v is not None}
            print(f"  Dados válidos para temp {temperature}: {len(valid_results)} pontos")
            
            if valid_results:
                plot_data_found = True
                x_values = list(valid_results.keys())
                y_values = list(valid_results.values())

                # Normaliza o eixo X: divide o número de 1's pelo número total de vizinhos
                # Isso transforma valores absolutos (0, 1, 2, 3...) em proporções (0.0, 0.33, 0.66, 1.0...)
                x_values_normalized = [x / effective_neighbors for x in x_values]
                
                print(f"    X normalizado: {x_values_normalized[:5]}... (primeiros 5)")
                print(f"    Y values: {y_values[:5]}... (primeiros 5)")
                
                # Plota os dados desta temperatura com sua própria cor
                color_index = i % len(colors)
                # Usa marcadores maiores e mais visíveis para garantir que todos os pontos sejam exibidos
                plt.plot(x_values_normalized, y_values, '-', label=f'Temperatura: {temperature}', color=colors[color_index])
                # Adiciona pontos separadamente com marcadores maiores e mais visíveis
                plt.scatter(x_values_normalized, y_values, s=50, color=colors[color_index], zorder=5)
            else:
                print(f"⚠️  Nenhum resultado válido para Temperatura {temperature}")
        
        if not plot_data_found:
            print("❌ Nenhum dado válido encontrado para plotagem!")
            plt.close()  # Limpa a figura vazia
            return None
        
        plt.xlabel('Proporção de vizinhos com opinião \'z\' (Número de 1\'s / Total de Vizinhos)')
        plt.ylabel('Média do output (0=K, 1=Z)')
        plt.title(f'Effect of Neighborhood Configuration ({effective_neighbors} neighbors)')
        plt.grid(True)
        plt.legend()
        
        # Simplifica nome do modelo para estrutura de diretórios
        model_simplified = get_simplified_model_name(effective_model)
        
        # Cria diretório para plots organizados por modelo e n_vizinhos
        base_dir = os.getenv('OUTPUT_BASE_DIR', 'resultados')
        results_dir = os.path.join(os.path.dirname(__file__), base_dir, model_simplified, f'n{effective_neighbors}', 'plots')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # ID do experimento
        exp_suffix = f"_exp{experiment_id}" if experiment_id else ""
        
        # Inclui o modelo e número de iterações no nome do arquivo para diferenciação
        model_safe = (effective_model or 'unknown').replace('/', '_').replace(':', '_')
        
        # PADRÃO: vizinhanca_{variant}_n{neighbors}_exp{id}_iter{iterations}_{timestamp}_{model}.png
        fig_filename = os.path.join(results_dir, f"vizinhanca_{effective_variant}_n{effective_neighbors}{exp_suffix}_iter{NUM_ITERATIONS}_{timestamp}_{model_safe}.png")
        
        plt.savefig(fig_filename, dpi=150, bbox_inches='tight')
        print(f"📈 Gráfico salvo em: {fig_filename}")
        
        # Remove plt.show() para evitar problemas no ambiente remoto
        plt.close()  # Limpa a memória
        
        return fig_filename
        
    except Exception as e:
        print(f"❌ ERRO na geração do plot: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        try:
            plt.close()  # Tenta limpar em caso de erro
        except:
            pass
        return None

def generate_results_report(all_raw_data: List[Dict]):
    """
    Gera um relatório dos resultados por temperatura e configuração binária.
    
    Args:
        all_raw_data: Lista de dicionários com todos os dados brutos
    """
    print("\n=== RELATÓRIO DETALHADO POR TEMPERATURA E CONFIGURAÇÃO ===")
    
    # Agrupa dados por temperatura
    data_by_temp = {}
    for row in all_raw_data:
        temp = row['temperatura']
        config = row['configuracao_binaria']
        escolha = row['escolha']
        
        if temp not in data_by_temp:
            data_by_temp[temp] = {}
        
        if config not in data_by_temp[temp]:
            data_by_temp[temp][config] = {'total': 0, 'escolhas_z': 0, 'num_ones': config.count('1')}
        
        data_by_temp[temp][config]['total'] += 1
        if escolha == 1:  # 1 representa escolha 'Z'
            data_by_temp[temp][config]['escolhas_z'] += 1
    
    # Imprime o relatório para cada temperatura
    for temp in sorted(data_by_temp.keys()):
        print(f"\nTemperatura: {temp}")
        print("Configuração Binária | Nº de 1's | Total de Iterações | Nº de vezes que escolheu \"Z\"")
        print("-" * 80)
        
        # Ordena configurações por ordem binária
        for config in sorted(data_by_temp[temp].keys()):
            stats = data_by_temp[temp][config]
            num_ones = stats['num_ones']
            total = stats['total']
            escolhas_z = stats['escolhas_z']
            print(f"{config:22} | {num_ones:9} | {total:20} | {escolhas_z:31}")

def print_final_token_stats():
    """Imprime estatísticas finais de performance baseadas no formato do teste_chamada.py."""
    global TOKEN_STATS
    
    print(f"\n[PERF] === ESTATÍSTICAS FINAIS ===")
    print(f"Total tokens processados: {TOKEN_STATS['total_tokens']:,}")
    print(f"Total requests: {TOKEN_STATS['request_count']:,}")
    print(f"Tempo total: {TOKEN_STATS['total_time']:.2f} segundos")
    
    if TOKEN_STATS['total_time'] > 0:
        avg_tps = TOKEN_STATS['total_tokens'] / TOKEN_STATS['total_time']
        avg_time_per_request = TOKEN_STATS['total_time'] / TOKEN_STATS['request_count'] if TOKEN_STATS['request_count'] > 0 else 0
        avg_tokens_per_request = TOKEN_STATS['total_tokens'] / TOKEN_STATS['request_count'] if TOKEN_STATS['request_count'] > 0 else 0
        
        print(f"TPS médio: {avg_tps:.1f}")
        print(f"Tempo médio por request: {avg_time_per_request:.2f}s")
        print(f"Tokens médios por request: {avg_tokens_per_request:.1f}")
        
        # Log também no arquivo
        log_file = get_current_log_file()
        if log_file is not None:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n[PERF] === ESTATÍSTICAS FINAIS ===\n")
                f.write(f"Total tokens: {TOKEN_STATS['total_tokens']:,}\n")
                f.write(f"Total requests: {TOKEN_STATS['request_count']:,}\n")
                f.write(f"Tempo total: {TOKEN_STATS['total_time']:.2f}s\n")
                f.write(f"TPS médio: {avg_tps:.1f}\n")
                f.write(f"Tempo médio por request: {avg_time_per_request:.2f}s\n")
                f.write(f"Tokens médios por request: {avg_tokens_per_request:.1f}\n")
    else:
        print("❌ Nenhum token foi processado.")

# FUNÇÃO REMOVIDA: Código síncrono legado removido - use run_orchestrated_experiment() que é assíncrona

async def run_orchestrated_experiment(experiment_data):
    """Executa um experimento específico baseado nos dados da planilha mestre (versão assíncrona)"""
    global MODEL, NUM_ITERATIONS, NUM_NEIGHBORS, PROMPT_VARIANT, LOG_FILE, PROMPT_LOG_FILE
    
    experiment_id = experiment_data['id_experimento']
    model_executado = experiment_data.get('modelo_executado', AVAILABLE_MODELS[0])
    
    print(f"\n=== EXECUTANDO EXPERIMENTO ASSÍNCRONO {experiment_id} ===")
    print(f"Conjunto: {experiment_data['conjunto_experimento']}")
    print(f"Notas: {experiment_data['notas']}")
    
    # BINDING TASK-LOCAL RÍGIDO: Define modelo e experimento para esta tarefa
    set_task_model(model_executado)
    set_task_experiment_id(experiment_id)
    
    # VALIDAÇÃO DE MODELO: Verifica se é válido antes de prosseguir
    if model_executado not in AVAILABLE_MODELS:
        error_msg = f"❌ MODELO INVÁLIDO: {model_executado} não está em AVAILABLE_MODELS: {AVAILABLE_MODELS}"
        print(error_msg)
        raise ValueError(error_msg)
    
    # Configura parâmetros globais baseado no experimento
    MODEL = experiment_data['modelo']
    NUM_ITERATIONS = int(experiment_data['num_iteracoes'])
    NUM_NEIGHBORS = int(experiment_data['num_vizinhos'])
    PROMPT_VARIANT = experiment_data['variante_prompt']
    
    # Define a variante no contexto da tarefa assíncrona para evitar race conditions
    set_task_prompt_variant(experiment_data['variante_prompt'])
    
    # Processa temperaturas (usa padrão se não definido)
    temp_raw = experiment_data.get('temperaturas')
    if temp_raw is None or str(temp_raw).strip() == '' or str(temp_raw).strip().lower() == 'none':
        # Temperatura padrão para experimentos n=3,7,9,11 (0 = determinístico)
        temperatures = [0.0]
    else:
        temp_str = str(temp_raw).strip()
        if ',' in temp_str:
            temperatures = [float(t.strip()) for t in temp_str.split(',')]
        else:
            temperatures = [float(temp_str)]
    
    print(f"Parâmetros: Modelo={MODEL}, Modelo Executado={model_executado}, Vizinhos={NUM_NEIGHBORS}, Iter={NUM_ITERATIONS}")
    print(f"Temperaturas: {temperatures}")
    print(f"Prompt: {PROMPT_VARIANT}")
    
    # Validação: O número de vizinhos deve ser ímpar
    if NUM_NEIGHBORS % 2 == 0:
        raise ValueError(f"O número de vizinhos ({NUM_NEIGHBORS}) deve ser ímpar")
    
    # Obtém o ID do experimento
    experiment_id = experiment_data['id_experimento']
    
    # Setup log files com ID do experimento (evita dependência de globais entre tarefas)
    LOG_FILE, PROMPT_LOG_FILE = setup_log_files(
        experiment_id=experiment_id,
        prompt_variant=PROMPT_VARIANT,
        num_neighbors=NUM_NEIGHBORS,
        model_name=model_executado
    )
    # Isola arquivos de log por tarefa para evitar mistura entre experimentos paralelos
    set_task_log_files(LOG_FILE, PROMPT_LOG_FILE)
    
    # Executa o experimento
    all_results = {}
    all_raw_data = []
    all_extended_data = []
    
    variant_results = {}
    
    for temp in temperatures:
        print(f"\nExecutando temperatura {temp} assincronamente...")
        # Passa o número específico de vizinhos deste experimento
        experiment_neighbors = int(experiment_data['num_vizinhos'])
        result_tuple = await run_experiment(temp, model_executado, experiment_neighbors)
        
        # Desempacota o resultado
        averages, raw_data, extended_data = result_tuple
        
        variant_results[temp] = averages
        
        # Adiciona informação da variante aos dados brutos (usa contexto da tarefa)
        task_variant = get_task_prompt_variant()
        for row in raw_data:
            row['variante'] = task_variant
            all_raw_data.append(row)
            
        # Adiciona informação da variante aos dados estendidos
        for row in extended_data:
            row['variante'] = task_variant
            all_extended_data.append(row)
    
    all_results[task_variant] = variant_results
    
    # Salva CSV combinado com ID do experimento (usando asyncio.to_thread para não bloquear)
    combined_csv_filename = None
    extended_csv_filename = None
    if all_raw_data:
        # Usa o número específico de vizinhos deste experimento
        experiment_neighbors = int(experiment_data['num_vizinhos'])
        combined_csv_filename = await asyncio.to_thread(
            save_combined_csv, all_raw_data, experiment_neighbors, task_variant, experiment_id, model_executado
        )
        print(f"CSV básico salvo em: {os.path.abspath(combined_csv_filename)}")
        
    if all_extended_data:
        # Salva CSV estendido com prompts e respostas
        experiment_neighbors = int(experiment_data['num_vizinhos'])
        extended_csv_filename = await asyncio.to_thread(
            save_extended_csv, all_extended_data, experiment_neighbors, task_variant, experiment_id, model_executado
        )
        print(f"CSV estendido salvo em: {os.path.abspath(extended_csv_filename)}")
    
    # Gera gráfico dos resultados APENAS se há dados suficientes para CSV
    # Isso garante consistência entre plots e CSVs gerados
    if all_results and task_variant in all_results and all_raw_data:
        experiment_neighbors = int(experiment_data['num_vizinhos'])
        print(f"Gerando gráfico para experimento {experiment_id} assincronamente...")
        print(f"  Dados para plot: {len(all_results[task_variant])} temperaturas")
        for temp, temp_results in all_results[task_variant].items():
            valid_count = len([v for v in temp_results.values() if v is not None])
            print(f"    Temp {temp}: {valid_count}/{len(temp_results)} resultados válidos")
        
        try:
            plot_filename = await asyncio.to_thread(
                plot_results, 
                all_results[task_variant],
                experiment_neighbors,  # Passa o número correto de vizinhos
                task_variant,          # Passa a variante específica
                model_executado,       # Passa o modelo específico
                experiment_id          # Passa o ID do experimento
            )
            if plot_filename:
                print(f"📈 Gráfico salvo com sucesso em: {plot_filename}")
            else:
                print(f"⚠️  Plot não foi gerado (dados insuficientes ou erro)")
        except Exception as e:
            print(f"❌ Erro ao gerar gráfico: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    elif all_results and task_variant in all_results and not all_raw_data:
        print(f"⚠️  Dados insuficientes para CSV - pulando geração de gráfico para manter consistência")
    else:
        print(f"⚠️  Condições não atendidas para geração de plot:")
        print(f"    all_results exists: {bool(all_results)}")
        print(f"    task_variant in all_results: {task_variant in all_results if all_results else False}")
        print(f"    all_raw_data exists: {bool(all_raw_data)}")
    
    # Imprime estatísticas finais de performance
    await asyncio.to_thread(print_final_token_stats)
    
    # Retorna caminhos dos arquivos gerados
    return {
        'caminho_csv_saida': os.path.abspath(combined_csv_filename) if combined_csv_filename else '',
        'caminho_log_saida': os.path.abspath(get_current_log_file()) if get_current_log_file() else '',
        'caminho_log_prompt_saida': os.path.abspath(get_current_prompt_log_file()) if get_current_prompt_log_file() else ''
    }

async def run_worker_for_model(model: str, start_id: int = None, end_id: int = None):
    """Worker assíncrono que executa experimentos para um modelo específico"""
    manager = ExperimentManager()
    
    while True:
        # Reservar próximo experimento no SQLite diretamente (sem to_thread para melhor debug)
        print(f"🔍 [WORKER_DEBUG] Modelo {model.split('/')[-1]}: buscando próximo experimento...")
        experiment_data = manager.get_next_experiment(model, start_id, end_id)
        if experiment_data is None:
            print(f"🏁 Worker para modelo {model.split('/')[-1]}: Não há mais experimentos pendentes")
            break

        exp_id = experiment_data['id_experimento']
        try:
            print(f"🚀 [Modelo {model.split('/')[-1]}] Iniciando experimento {exp_id}")
            
            file_paths = await run_orchestrated_experiment(experiment_data)
            
            # Atualizar status diretamente (sem to_thread para eliminar race condition)
            print(f"🔍 [WORKER_DEBUG] Modelo {model.split('/')[-1]}: atualizando status do experimento {exp_id}")
            success = manager.update_experiment_status(exp_id, 'concluido', **file_paths)
            if not success:
                print(f"❌ [WORKER_DEBUG] FALHA ao atualizar status do experimento {exp_id}!")
            print(f"✅ [Modelo {model.split('/')[-1]}] Experimento {exp_id} concluído!")
            
        except Exception as e:
            error_message = str(e)
            print(f"❌ [WORKER_DEBUG] Modelo {model.split('/')[-1]}: experimento {exp_id} falhou: {error_message}")
            success = manager.update_experiment_status(exp_id, 'erro', notas=f"ERRO: {error_message}")
            if not success:
                print(f"❌ [WORKER_DEBUG] FALHA ao atualizar status de erro do experimento {exp_id}!")
            print(f"❌ [Modelo {model.split('/')[-1]}] Experimento {exp_id} falhou: {error_message}")

async def run_auto_orchestrator(single_experiment=False, start_id=None, end_id=None):
    """Modo orquestrador automático assíncrono - executa experimentos da planilha mestre"""
    
    if single_experiment:
        print("🎯 === MODO EXPERIMENTO ÚNICO ASSÍNCRONO ===")
    else:
        print("🤖 === MODO ORQUESTRADOR AUTOMÁTICO ASSÍNCRONO COM PARALELIZAÇÃO ===")
    
    # Mostra range de IDs se especificado
    if start_id is not None or end_id is not None:
        range_str = f"IDs {start_id or 'início'} até {end_id or 'fim'}"
        print(f"🎯 Executando apenas experimentos no range: {range_str}")
    
    # Sincroniza CSV para SQLite APENAS se solicitado explicitamente
    if os.getenv('SYNC_DB_FROM_CSV', 'false').lower() == 'true':
        print("📊 Sincronizando CSV → SQLite (SYNC_DB_FROM_CSV=true)...")
        import csv_to_sqlite
        sync_success = await asyncio.to_thread(csv_to_sqlite.csv_to_sqlite)
        if not sync_success:
            print("❌ Falha na sincronização CSV → SQLite. Abortando.")
            return
        print("✅ Sincronização CSV → SQLite concluída")
    else:
        print("⏭️  Pulando sincronização CSV → SQLite (usando estado atual do DB).")
    
    # Limpa logs antigos do tmux antes de começar
    print("🧹 Limpando logs antigos...")
    await asyncio.to_thread(clean_tmux_logs)
    
    manager = ExperimentManager()
    
    # Recupera experimentos órfãos
    print("🔄 Verificando experimentos órfãos...")
    recovered = await asyncio.to_thread(manager.recover_orphaned_experiments)
    
    if single_experiment:
        # Modo single experiment - executa apenas 1
        print(f"🔍 [SINGLE_DEBUG] Buscando experimento único...")
        experiment_data = manager.get_next_experiment(None, start_id, end_id)
        
        if experiment_data is None:
            if start_id is not None or end_id is not None:
                print(f"\n❌ Não há experimentos pendentes no range {start_id or 'início'} até {end_id or 'fim'}.")
            else:
                print("\n❌ Não há experimentos pendentes.")
            return
        
        experiment_id = experiment_data['id_experimento']
        print(f"\n🎯 Executando experimento único assíncrono {experiment_id}")
        
        try:
            file_paths = await run_orchestrated_experiment(experiment_data)
            print(f"🔍 [SINGLE_DEBUG] Atualizando status do experimento {experiment_id}")
            success = manager.update_experiment_status(experiment_id, 'concluido', **file_paths)
            if not success:
                print(f"❌ [SINGLE_DEBUG] FALHA ao atualizar status do experimento {experiment_id}!")
            print(f"✅ Experimento assíncrono {experiment_id} concluído com sucesso!")
        except Exception as e:
            error_message = str(e)
            print(f"❌ [SINGLE_DEBUG] Experimento {experiment_id} falhou: {error_message}")
            success = manager.update_experiment_status(experiment_id, 'erro', notas=f"ERRO: {error_message}")
            if not success:
                print(f"❌ [SINGLE_DEBUG] FALHA ao atualizar status de erro do experimento {experiment_id}!")
            print(f"❌ Experimento assíncrono {experiment_id} falhou: {error_message}")
        
        return
    
    # Modo paralelo - executa até N experimentos simultaneamente (1 por modelo)
    print(f"Iniciando orquestração assíncrona paralela com {len(AVAILABLE_MODELS)} modelos disponíveis")
    
    # Cria tarefas para cada modelo
    tasks = [
        asyncio.create_task(run_worker_for_model(model, start_id, end_id), name=f"Worker-{model}")
        for model in AVAILABLE_MODELS
    ]
    
    # Aguarda todas as tarefas terminarem
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Mostra status final
    status_info = await asyncio.to_thread(manager.get_experiments_status)
    if status_info:
        print(f"\n=== STATUS FINAL ASSÍNCRONO ===")
        print(f"Total de experimentos: {status_info['total']}")
        for status, count in status_info['status_counts'].items():
            print(f"  {status}: {count}")

async def run_manual(temperatures: list, variantes: list):
    """Executa experimentos manuais de forma assíncrona"""
    global PROMPT_VARIANT
    
    set_task_model(AVAILABLE_MODELS[0])  # Usa o primeiro modelo disponível
    log_file, prompt_log_file = setup_log_files(
        prompt_variant=PROMPT_VARIANT, num_neighbors=NUM_NEIGHBORS, model_name=get_task_model()
    )
    set_task_log_files(log_file, prompt_log_file)

    all_results = {}
    all_raw_data = []
    
    for variant in variantes:
        print(f"\n🎯 === EXECUTANDO VARIANTE ASSÍNCRONA: {variant} ===")
        
        # Atualiza a variante global para ser usada nas funções
        PROMPT_VARIANT = variant
        
        variant_results = {}
        variant_raw_data = []
        
        for temp in temperatures:
            print(f"\nExecutando temperatura {temp} com variante {variant} assincronamente...")
            # Agora run_experiment_async retorna médias, dados brutos e dados estendidos
            result_tuple = await run_experiment(temp, AVAILABLE_MODELS[0], NUM_NEIGHBORS)
            
            # Desempacota o resultado
            averages, raw_data, extended_data = result_tuple
            
            variant_results[temp] = averages
            
            # Adiciona informação da variante aos dados brutos
            for row in raw_data:
                row['variante'] = variant
                variant_raw_data.append(row)
                all_raw_data.append(row)
        
        # Salva resultados específicos desta variante
        all_results[variant] = variant_results
    
    # Print summary
    print("\n=== RESUMO DOS RESULTADOS ASSÍNCRONOS ===")
    print(f"Total de vizinhos: {NUM_NEIGHBORS}, Total de participantes: {TOTAL_PARTICIPANTS}")
    print(f"Configuração: before: {NUM_NEIGHBORS // 2}, after: {(NUM_NEIGHBORS + 1) // 2}")
    
    for variant in variantes:
        print(f"\n🎯 Variante: {variant}")
        for temp, results in all_results[variant].items():
            print(f"  Temperatura {temp}:")
            for num_ones, avg in results.items():
                if avg is not None:
                    print(f"    {num_ones} vizinhos 'z': {avg:.4f}")
            else:
                    print(f"    {num_ones} vizinhos 'z': N/A (None)")
    
    # Plota resultados APENAS se há dados suficientes para CSV (mantém consistência)
    if len(variantes) == 1 and all_raw_data:
        try:
            plot_filename = await asyncio.to_thread(
                plot_results, 
                all_results[variantes[0]],
                NUM_NEIGHBORS,
                variantes[0],
                get_task_model(),
                None  # Modo manual não tem experiment_id específico
            )
            if plot_filename:
                print(f"📈 Gráfico manual salvo em: {plot_filename}")
            else:
                print(f"⚠️  Plot manual não foi gerado")
        except Exception as e:
            print(f"❌ Erro ao gerar gráfico manual: {e}")
    elif len(variantes) == 1 and not all_raw_data:
        print(f"⚠️  Dados insuficientes para CSV - pulando geração de gráfico para manter consistência")
    elif len(variantes) > 1:
        print(f"\nPara visualizar gráficos de múltiplas variantes, execute cada uma individualmente.")
    
    # Salva CSV combinado com todas as variantes
    if all_raw_data:
        # Para CSV combinado, usa a primeira variante se houver apenas uma, senão usa "combinados"
        variant_for_combined = variantes[0] if len(variantes) == 1 else "combinados"
        # Modo manual não tem experiment_id específico, então passa None
        combined_csv_filename = await asyncio.to_thread(
            save_combined_csv, all_raw_data, NUM_NEIGHBORS, variant_for_combined, None, get_task_model()
        )
        
        # Gera relatório detalhado
        await asyncio.to_thread(generate_results_report, all_raw_data)
        
        # Print do caminho do CSV como último output
        print(f"\n📁 CSV combinado assíncrono com todas as variantes salvo em:")
        print(f"   {os.path.abspath(combined_csv_filename)}")
    else:
        print("Nenhum dado bruto disponível para salvar em CSV.")
    
    # Imprime estatísticas finais de performance
    await asyncio.to_thread(print_final_token_stats)

# FUNÇÃO REMOVIDA: Código síncrono legado removido - use run_auto_orchestrator() que é assíncrona

def show_experiments_status():
    """Mostra o status atual de todos os experimentos"""
    print("=== STATUS DOS EXPERIMENTOS ===")
    
    manager = ExperimentManager()
    status_info = manager.get_experiments_status()
    
    if not status_info:
        print("❌ Erro ao ler status dos experimentos.")
        return
    
    print(f"\nTotal de experimentos: {status_info['total']}")
    print(f"\nResumo por status:")
    for status, count in status_info['status_counts'].items():
        print(f"  {status}: {count}")
    
    # Mostra detalhes dos experimentos em execução
    executando = [exp for exp in status_info['experiments'] if exp['status'] == 'executando']
    if executando:
        print(f"\n🔄 Experimentos em execução ({len(executando)}):")
        for exp in executando:
            modelo_exec = exp.get('modelo_executado', 'N/A')
            print(f"  ID {exp['id_experimento']}: {exp['conjunto_experimento']} - "
                  f"Modelo: {modelo_exec} - PID {exp['process_id']} desde {exp['hora_inicio']}")
    
    # Mostra próximos experimentos pendentes (primeiros 5)
    pendentes = [exp for exp in status_info['experiments'] if exp['status'] == 'pendente']
    if pendentes:
        print(f"\n⏳ Próximos experimentos pendentes ({len(pendentes)} total, mostrando primeiros 5):")
        for exp in pendentes[:5]:
            print(f"  ID {exp['id_experimento']}: {exp['conjunto_experimento']} - {exp['variante_prompt']}")
    
    # Mostra experimentos com erro
    erro = [exp for exp in status_info['experiments'] if exp['status'] == 'erro']
    if erro:
        print(f"\n❌ Experimentos com erro ({len(erro)}):")
        for exp in erro:
            print(f"  ID {exp['id_experimento']}: {exp['conjunto_experimento']} - {exp['notas']}")

# ...existing code...

def main():
    global MODEL, NUM_ITERATIONS, LOG_FILE, NUM_NEIGHBORS, PROMPT_LOG_FILE, TOTAL_PARTICIPANTS, PROMPT_VARIANT
    
    # ===============================================================================
    # CONFIGURAÇÕES HARDCODED PARA EXECUÇÃO DIRETA (python execucao_simultanea.py)
    # ===============================================================================
    
    # Verifica se não foram passados argumentos (execução direta)
    import sys
    if len(sys.argv) == 1:
        print("🔧 === MODO EXECUÇÃO DIRETA COM CONFIGURAÇÕES HARDCODED ===")
        print("🚀 Usando modo assíncrono (padrão)")
        
        # Configurações hardcoded - MODIFIQUE AQUI CONFORME NECESSÁRIO
        HARDCODED_CONFIG = {
            'temperatures': [0.0],
            'model': 'gemma-12b',
            'iterations': 1,
            'neighbors': 3,  # Sempre ímpar
            'participants': 10,
            'prompt_variant': 'v20_lista_completa_meio_raciocinio_primeiro',
            'variantes_para_testar': ['v20_lista_completa_meio_raciocinio_primeiro'],
            'mode': 'manual'  # 'manual' ou 'orchestrator'
        }
        
        print(f"📋 Configurações:")
        print(f"   Modelo: {HARDCODED_CONFIG['model']}")
        print(f"   Vizinhos: {HARDCODED_CONFIG['neighbors']}")
        print(f"   Iterações: {HARDCODED_CONFIG['iterations']}")
        print(f"   Temperaturas: {HARDCODED_CONFIG['temperatures']}")
        print(f"   Variante: {HARDCODED_CONFIG['prompt_variant']}")
        print(f"   Modo: {HARDCODED_CONFIG['mode']}")
        
        # Aplica configurações globais
        MODEL = HARDCODED_CONFIG['model']
        NUM_ITERATIONS = HARDCODED_CONFIG['iterations']
        NUM_NEIGHBORS = HARDCODED_CONFIG['neighbors']
        TOTAL_PARTICIPANTS = HARDCODED_CONFIG['participants']
        PROMPT_VARIANT = HARDCODED_CONFIG['prompt_variant']
        
        # Validação: O número de vizinhos deve ser ímpar
        if NUM_NEIGHBORS % 2 == 0:
            print(f"❌ ERRO: O número de vizinhos ({NUM_NEIGHBORS}) deve ser ímpar para este experimento.")
            return
        
        # Executa baseado no modo configurado
        if HARDCODED_CONFIG['mode'] == 'orchestrator':
            print("🤖 Executando modo orquestrador...")
            asyncio.run(run_auto_orchestrator(single_experiment=False))
        elif HARDCODED_CONFIG['mode'] == 'single_experiment':
            print("🎯 Executando experimento único...")
            asyncio.run(run_auto_orchestrator(single_experiment=True))
        else:  # manual
            print("🧪 Executando modo manual...")
            asyncio.run(run_manual(HARDCODED_CONFIG['temperatures'], HARDCODED_CONFIG['variantes_para_testar']))
        
        return
    
    # ===============================================================================
    # MODO CLI NORMAL (com argumentos)
    # ===============================================================================
    
    parser = argparse.ArgumentParser(description='Executa experimentos de configuração de vizinhança com diferentes temperaturas')
    parser.add_argument('--temperatures', type=float, nargs='+', default=TEMPERATURES,
                      help=f'Temperaturas para executar experimentos (padrão: {TEMPERATURES})')
    parser.add_argument('--model', type=str, default=MODEL,
                      help=f'Modelo a ser usado para o experimento (padrão: {MODEL})')
    parser.add_argument('--iterations', type=int, default=NUM_ITERATIONS,
                      help=f'Número de iterações por configuração (padrão: {NUM_ITERATIONS})')
    parser.add_argument('--neighbors', type=int, default=NUM_NEIGHBORS,
                      help=f'Número TOTAL de vizinhos no experimento (sempre ímpar) (padrão: {NUM_NEIGHBORS})')
    parser.add_argument('--participants', type=int, default=TOTAL_PARTICIPANTS,
                      help=f'Número total de participantes no experimento (padrão: {TOTAL_PARTICIPANTS})')
    parser.add_argument('--prompt-variant', type=str, default=PROMPT_VARIANT,
                      help=f'Variante de prompt a ser usada (padrão: {PROMPT_VARIANT}, opções disponíveis: {", ".join(VARIANTES_TESTE)})')
    parser.add_argument('--test-all-variants', action='store_true',
                      help='Executar experimento com todas as variantes de prompt disponíveis')
    parser.add_argument('--auto-orchestrate', action='store_true',
                      help='Modo orquestrador: executa experimentos da planilha mestre automaticamente')
    parser.add_argument('--status', action='store_true',
                      help='Mostra o status atual de todos os experimentos')
    parser.add_argument('--single-experiment', action='store_true',
                      help='Executa apenas o próximo experimento pendente e para (não continua com outros)')
    parser.add_argument('--run-experiment-id', type=int, metavar='ID',
                      help='Executa um experimento específico por ID (modo manual)')
    parser.add_argument('--force-model', type=str, choices=AVAILABLE_MODELS, metavar='MODEL',
                      help=f'Força o uso de um modelo específico (opções: {", ".join(AVAILABLE_MODELS)})')
    parser.add_argument('--start-id', type=int, metavar='ID',
                      help='ID mínimo dos experimentos a executar (inclusive)')
    parser.add_argument('--end-id', type=int, metavar='ID',
                      help='ID máximo dos experimentos a executar (inclusive)')
    parser.add_argument('--force-cleanup', action='store_true',
                      help='Força limpeza de todos os experimentos órfãos')
    args = parser.parse_args()
    
    # Modo especial: mostrar status
    if args.status:
        show_experiments_status()
        return
    
    # Modo especial: forçar limpeza de órfãos
    if args.force_cleanup:
        print("🧹 === LIMPEZA FORÇADA DE EXPERIMENTOS ÓRFÃOS ===")
        manager = ExperimentManager()
        recovered = manager.recover_orphaned_experiments()
        if recovered and recovered > 0:
            print(f"✅ Limpeza concluída: {recovered} experimento(s) recuperado(s)")
        else:
            print("✅ Nenhum experimento órfão encontrado")
        return
    
    # Modo especial: executar experimento específico por ID
    if args.run_experiment_id:
        async def run_single_experiment_by_id():
            print(f"🎯 === MODO EXECUÇÃO MANUAL - EXPERIMENTO {args.run_experiment_id} ===")

            manager = ExperimentManager()

            # Tenta obter o experimento específico
            experiment_data = manager.get_experiment_by_id(args.run_experiment_id, args.force_model)

            if experiment_data is None:
                print(f"❌ Experimento {args.run_experiment_id} não encontrado, não está pendente, ou modelo não disponível.")
                if args.force_model:
                    print(f"   Modelo solicitado: {args.force_model}")
                return

            experiment_id = experiment_data['id_experimento']
            model_name = experiment_data['modelo_executado']

            print(f"✅ Experimento {experiment_id} encontrado e reservado com modelo {model_name}")

            try:
                file_paths = await run_orchestrated_experiment(experiment_data)
                manager.update_experiment_status(experiment_id, 'concluido', **file_paths)
                print(f"🎉 Experimento {experiment_id} concluído com sucesso!")
            except Exception as e:
                error_message = str(e)
                manager.update_experiment_status(experiment_id, 'erro', notas=f"ERRO: {error_message}")
                print(f"❌ Experimento {experiment_id} falhou: {error_message}")
        
        asyncio.run(run_single_experiment_by_id())
        return
    
    # Modo especial: orquestrador automático
    if args.auto_orchestrate or args.single_experiment:
        print("🚀 Executando modo orquestrador assíncrono")
        asyncio.run(run_auto_orchestrator(single_experiment=args.single_experiment, start_id=args.start_id, end_id=args.end_id))
        return
    
    # Validação: O número de vizinhos deve ser ímpar para que o agente possa estar no centro
    if args.neighbors % 2 == 0:
        print(f"❌ ERRO: O número de vizinhos (--neighbors) deve ser ímpar para este experimento. Você forneceu: {args.neighbors}.")
        return  # Encerra o script se o número for par
    
    # Validação: A variante de prompt deve estar na lista de variantes disponíveis
    if not args.test_all_variants and args.prompt_variant not in VARIANTES_TESTE:
        print(f"❌ ERRO: Variante de prompt '{args.prompt_variant}' não é válida.")
        print(f"Variantes disponíveis: {', '.join(VARIANTES_TESTE)}")
        return
    
    MODEL = args.model
    NUM_ITERATIONS = args.iterations
    NUM_NEIGHBORS = args.neighbors 
    TOTAL_PARTICIPANTS = args.participants
    
    # Determina quais variantes executar
    if args.test_all_variants:
        variantes_para_testar = VARIANTES_TESTE
        print(f"🧪 Executando experimento com TODAS as {len(VARIANTES_TESTE)} variantes de prompt.")
    else:
        variantes_para_testar = [args.prompt_variant]
        PROMPT_VARIANT = args.prompt_variant
        print(f"🧪 Executando experimento com a variante: {args.prompt_variant}")
    
    temperatures = args.temperatures

    print("\n=== EXPERIMENTO DE CONFIGURAÇÃO DE VIZINHANÇA COM TEMPERATURAS VARIÁVEIS ===\n")
    print(f"Modelo: {MODEL}")
    print(f"Iterações por configuração: {NUM_ITERATIONS}")
    print(f"Variantes a testar: {variantes_para_testar}")
    print(f"Temperaturas: {temperatures}")
    
    print("🚀 Executando modo manual assíncrono")
    asyncio.run(run_manual(temperatures, variantes_para_testar))

if __name__ == "__main__":
    main()
