#!/usr/bin/env python3
"""
Script ÚNICO para simulações de autômato com Numba JIT
Aceita CSV como entrada, extrai regras automaticamente se necessário, e roda simulações.

USO:
    python run_automaton_numba.py --csv dados.csv
    python run_automaton_numba.py --csv dados.csv --n_simulations 50 --agents 200 --no-plots
    python run_automaton_numba.py --csv dados.csv --force-extract
    python run_automaton_numba.py --all-configs
"""

import csv
import json
import sys
import os
import time
import glob
import re
from typing import List, Dict, Tuple, Optional

from metrics_utils import calcular_ici_token_bias_from_csv

try:
    from numba import jit
    import numpy as np
    NUMBA_AVAILABLE = True
except ImportError:
    print("❌ ERRO: Numba não está instalado!")
    print("   Instale com: pip install numba")
    sys.exit(1)

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projeto_final.utils.initial_distribution import (
    compute_majority_counts as shared_compute_majority_counts,
)
from projeto_final.utils.utils import generate_initial_distribution_shared

# Constantes
OPINION_K = 0
OPINION_Z = 1
OPINION_LABELS = {OPINION_K: 'k', OPINION_Z: 'z'}

# ═══════════════════════════════════════════════════════════════════════════
# EXTRAÇÃO DE REGRAS (integrada)
# ═══════════════════════════════════════════════════════════════════════════

def get_rules_json_path(csv_path: str) -> str:
    """Retorna caminho esperado do JSON baseado no CSV"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_basename = os.path.basename(csv_path).replace('.csv', '')
    output_dir = os.path.join(script_dir, "extracted_rules")
    return os.path.join(output_dir, f"regras_FULL_{csv_basename}.json")

def check_rules_exist(csv_path: str) -> bool:
    """Verifica se regras já foram extraídas"""
    json_path = get_rules_json_path(csv_path)
    return os.path.exists(json_path)

def extract_rules_from_csv(csv_path: str) -> str:
    """
    Extrai regras de um CSV e salva JSON.
    Retorna: caminho do JSON criado
    
    Verifica primeiro se JSON já existe:
    - Se existe: retorna caminho e pula extração
    - Se não existe: extrai e salva
    """
    json_path = get_rules_json_path(csv_path)
    
    # Verifica se já existe
    if os.path.exists(json_path):
        print(f"✅ Regras já extraídas: {os.path.basename(json_path)}")
        return json_path
    
    # Extrai regras
    print(f"📄 Extraindo regras de: {os.path.basename(csv_path)}")
    rules = {}
    n = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = row['configuracao_binaria']
            escolha = int(row['escolha'])
            n = len(config)
            rules[config] = escolha
    
    print(f"✅ Extraídas {len(rules)} regras para n={n}")
    print(f"   Esperado: {2**n} regras")
    print(f"   K(1): {sum(1 for v in rules.values() if v==1)}")
    print(f"   Z(0): {sum(1 for v in rules.values() if v==0)}")
    
    # Salva JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "extracted_rules")
    os.makedirs(output_dir, exist_ok=True)
    
    data = {
        'source_csv': os.path.basename(csv_path),
        'n_agents': n,
        'total_rules': len(rules),
        'rules': rules
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Salvo em: {os.path.basename(json_path)}")
    return json_path

# ═══════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════

def ensure_directory_exists(directory: str):
    """Cria diretório se não existir"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def _compute_majority_counts(n_agents: int, majority_ratio: float) -> Tuple[int, int]:
    """Wrapper para manter compatibilidade local com gerador compartilhado."""
    return shared_compute_majority_counts(n_agents, majority_ratio)


def generate_initial_configuration_shared(
    n_agents: int,
    seed_distribution: int,
    majority_ratio: float = 0.51,
    distribution_mode: str = "auto",
) -> Tuple[np.ndarray, int, int, int, float, str]:
    """
    Gera configuração inicial usando o módulo compartilhado.
    Retorna: opinions, majority, count_0, count_1, used_ratio, used_mode.
    """
    dist = generate_initial_distribution_shared(
        n_agents=n_agents,
        seed_distribution=seed_distribution,
        majority_ratio=majority_ratio,
        requested_mode=distribution_mode,
    )
    return (
        dist.opinions,
        int(dist.initial_majority_opinion),
        int(dist.count_0),
        int(dist.count_1),
        float(dist.majority_ratio),
        str(dist.mode),
    )

def build_rules_array(rules_dict: Dict[str, int], window_size: int) -> np.ndarray:
    """Converte dicionário de regras em array NumPy indexado"""
    max_index = 2 ** window_size
    rules_array = np.full(max_index, -1, dtype=np.int8)
    
    for binary_str, decision in rules_dict.items():
        index = int(binary_str, 2)
        rules_array[index] = decision
    
    return rules_array

# ═══════════════════════════════════════════════════════════════════════════
# FUNÇÕES COMPILADAS COM NUMBA JIT
# ═══════════════════════════════════════════════════════════════════════════

@jit(nopython=True)
def extract_windows_numba(state: np.ndarray, window_size: int) -> np.ndarray:
    """
    Extrai todas as janelas de uma vez - COMPILADO COM NUMBA.
    ~5-10x mais rápido que versão Python pura.
    """
    n_agents = len(state)
    half_window = window_size // 2
    window_indices = np.zeros(n_agents, dtype=np.int32)
    
    # Pré-calcula potências de 2
    powers = np.zeros(window_size, dtype=np.int32)
    for i in range(window_size):
        powers[i] = 2 ** (window_size - 1 - i)
    
    # Extrai todas as janelas
    for i in range(n_agents):
        window_val = 0
        for offset in range(-half_window, half_window + 1):
            pos = (i + offset) % n_agents  # Wrap-around circular
            idx = offset + half_window
            window_val += state[pos] * powers[idx]
        window_indices[i] = window_val
    
    return window_indices

@jit(nopython=True)
def check_consensus_numba(state: np.ndarray) -> bool:
    """Verifica consenso - COMPILADO COM NUMBA"""
    if len(state) == 0:
        return False
    first = state[0]
    for val in state:
        if val != first:
            return False
    return True

@jit(nopython=True)
def simulate_automaton_numba_core(
    initial_state: np.ndarray,
    rules_array: np.ndarray,
    window_size: int,
    actual_max: int,
    initial_majority: int
) -> Tuple[np.ndarray, bool, int]:
    """
    Core da simulação - COMPILADO COM NUMBA.
    
    Esta função é compilada para código de máquina nativo,
    resultando em 5-10x speedup vs Python puro.
    """
    n_agents = len(initial_state)
    history = np.zeros((actual_max + 1, n_agents), dtype=np.int8)
    history[0] = initial_state
    current_state = initial_state.copy()
    
    for step in range(actual_max):
        # Extração vetorizada de janelas (compilada)
        window_indices = extract_windows_numba(current_state, window_size)
        
        # Lookup vetorizado de regras
        next_state = rules_array[window_indices].copy()
        
        # Tratamento de regras não definidas
        for i in range(n_agents):
            if next_state[i] == -1:
                next_state[i] = current_state[i]
        
        # Atualiza estado
        current_state = next_state
        history[step + 1] = current_state
        
        # Verifica consenso
        if check_consensus_numba(current_state):
            if current_state[0] == initial_majority:
                return history[:step + 2], True, step + 1
            else:
                return history[:step + 2], False, step + 1
    
    return history, False, -1

def simulate_automaton_run_numba(
    initial_state: np.ndarray,
    rules_array: np.ndarray,
    window_size: int,
    max_steps: int,
    initial_majority: int
) -> Tuple[np.ndarray, bool, Optional[int]]:
    """Wrapper para versão Numba JIT"""
    n_agents = len(initial_state)
    max_steps_2n = 2 * n_agents
    actual_max = min(max_steps, max_steps_2n)
    
    history, success, round_val = simulate_automaton_numba_core(
        initial_state, rules_array, window_size, actual_max, initial_majority
    )
    
    success_round = round_val if round_val >= 0 else None
    return history, success, success_round

# ═══════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE PLOT E I/O
# ═══════════════════════════════════════════════════════════════════════════

def plot_automaton_history(history: np.ndarray, output_dir: str, filename_prefix: str, 
                           n_agents: int, rule_name: str, consensus_info: Optional[Tuple[int, int]],
                           success: bool = False, initial_majority: Optional[int] = None,
                           generate_plot: bool = True):
    """Plota histórico com indicação de SUCESSO/FALHA"""
    if not generate_plot:
        return
    
    ensure_directory_exists(output_dir)
    n_rounds_completed = len(history)
    
    percent_k = np.sum(history == OPINION_K, axis=1) / n_agents * 100
    percent_z = np.sum(history == OPINION_Z, axis=1) / n_agents * 100

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3]})
    
    # Título
    if success:
        title = f"✅ SUCESSO - {rule_name}: {n_agents} Agentes"
        if consensus_info:
            round_c, opinion_c = consensus_info
            maj_label = OPINION_LABELS[initial_majority] if initial_majority is not None else '?'
            title += f"\nConsenso em {OPINION_LABELS[opinion_c]} (maioria inicial: {maj_label}) na Rodada {round_c}"
    else:
        title = f"❌ FALHA - {rule_name}: {n_agents} Agentes"
        if consensus_info:
            round_c, opinion_c = consensus_info
            maj_label = OPINION_LABELS[initial_majority] if initial_majority is not None else '?'
            title += f"\nConsenso INCORRETO em {OPINION_LABELS[opinion_c]} (maioria inicial: {maj_label})"
        else:
            title += f"\nSem Consenso em {n_rounds_completed-1} Rodadas (limite: 2N = {2*n_agents})"
        
    fig.suptitle(title, fontsize=14, fontweight='bold', 
                 color='green' if success else 'red')

    # Gráfico de Porcentagem
    # K (0) = BRANCO no heatmap → linha clara no gráfico
    # Z (1) = PRETO no heatmap → linha escura no gráfico
    axs[0].plot(percent_k, label=f'% K ({OPINION_K}) - Primeiro Token', color='lightgray', linestyle='-', linewidth=2.5, marker='o', markersize=3, markevery=max(1, len(percent_k)//20))
    axs[0].plot(percent_z, label=f'% Z ({OPINION_Z}) - Segundo Token', color='black', linestyle='-', linewidth=2.5, marker='s', markersize=3, markevery=max(1, len(percent_z)//20))
    axs[0].set_ylabel('Porcentagem de Agentes')
    axs[0].set_ylim([0, 100])
    axs[0].legend()
    axs[0].grid(True)

    # Gráfico de Evolução
    # Usa gray_r (reversed): 0=branco, 1=preto, com vmin=0, vmax=1 (sem inversão)
    cax = axs[1].imshow(history, cmap='gray_r', vmin=OPINION_K, vmax=OPINION_Z, 
                        aspect='auto', interpolation='nearest')
    cbar = fig.colorbar(cax, ax=axs[1], ticks=[OPINION_K, OPINION_Z])
    cbar.set_label('Opinião')
    cbar.set_ticklabels([f'K ({OPINION_K}) - Branco/Token1', f'Z ({OPINION_Z}) - Preto/Token2'])
    
    axs[1].set_xlabel('Agente ID')
    axs[1].set_ylabel('Rodada')
    axs[1].set_xticks(np.arange(0, n_agents, max(1, n_agents // 10)))
    axs[1].set_yticks(np.arange(0, n_rounds_completed, max(1, n_rounds_completed // 10)))
    
    if consensus_info:
        round_c, _ = consensus_info
        axs[0].axvline(x=round_c, color='black', linestyle=':', linewidth=2)
        axs[1].axhline(y=round_c, color='black', linestyle=':', linewidth=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    status_prefix = "SUCCESS" if success else "FAIL"
    if consensus_info:
        round_c, opinion_c = consensus_info
        status = f"{status_prefix}_{OPINION_LABELS[opinion_c]}_R{round_c}"
    else:
        status = f"{status_prefix}_no_consensus_{n_rounds_completed-1}R"
    
    plot_filename = os.path.join(output_dir, f"{filename_prefix}_{status}.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)

def load_rules_from_json(json_path: str):
    """Carrega regras do JSON e converte para array NumPy"""
    global OPINION_LABELS
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rules_dict = data['rules']
    
    if 'n_agents' in data:
        window_size = data['n_agents']
    else:
        first_key = next(iter(rules_dict.keys()))
        window_size = len(first_key)
    
    rules_array = build_rules_array(rules_dict, window_size)
    
    OPINION_LABELS[OPINION_K] = 'k'
    OPINION_LABELS[OPINION_Z] = 'z'
    
    print(f"✅ Regras carregadas de: {os.path.basename(json_path)}")
    print(f"📊 Fonte original: {data.get('source_csv', 'N/A')}")
    print(f"🪟 Tamanho da janela: {window_size}")
    print(f"📏 Total de regras: {data.get('total_rules', len(rules_dict))}")
    print(f"🚀 NUMBA JIT compilado e pronto!")
    
    return rules_array, data.get('source_csv', 'unknown'), window_size

# ═══════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL DE SIMULAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

def run_simulations_from_csv(
    csv_path: str,
    n_simulations: int = 100,
    n_agents: int = 100,
    max_iterations: int = 200,
    generate_plot: bool = True,
    force_extract: bool = False,
    majority_ratio: float = 0.51,
    initial_distribution_mode: str = "auto",
    source_tag: Optional[str] = None
):
    """
    Executa simulações a partir de um CSV.
    
    Fluxo:
    1. Calcula ICI e Token Bias do CSV original
    2. Verifica se regras já foram extraídas
    3. Se não existir OU force_extract: extrai regras do CSV
    4. Carrega regras (NumPy array)
    5. Roda N simulações sequencialmente (Numba JIT)
    6. Para cada simulação: gera gráfico se solicitado
    7. Gera relatório final
    """
    
    # 1. Calcula ICI e Token Bias
    ici, token_bias = calcular_ici_token_bias_from_csv(csv_path)
    
    # 2. Verifica/extrai regras
    if force_extract or not check_rules_exist(csv_path):
        json_path = extract_rules_from_csv(csv_path)
    else:
        json_path = get_rules_json_path(csv_path)
        print(f"✅ Usando regras já extraídas: {os.path.basename(json_path)}")
    
    # 2. Carrega regras
    rules_array, source_csv, window_size = load_rules_from_json(json_path)
    csv_basename = os.path.basename(csv_path).replace('.csv', '')

    # 3. Configura diretório de saída (estrutura: modelo/token/window/prompt_strategy/agents/majority/simId)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    meta = parse_csv_metadata(csv_path)
    model_slug = meta['model']
    token_pair = meta['token_type']
    token_slug = token_pair.replace('/', '')
    window_slug = f"n{window_size}"
    prompt_strategy = meta['prompt_strategy']  # 'com_cot' ou 'only_token'
    majority_percent = 50 if str(initial_distribution_mode).lower() == "half_split" else int(majority_ratio * 100)
    output_parts = [
        script_dir,
        "resultados",
        model_slug,
        token_slug,
        window_slug,
        prompt_strategy,
    ]
    if source_tag:
        output_parts.append(source_tag)
    output_parts.append(f"{n_agents}agents_{majority_percent}pct")
    base_output_dir = os.path.join(*output_parts)
    ensure_directory_exists(base_output_dir)
    
    max_iterations_2n = 2 * n_agents
    actual_max = min(max_iterations, max_iterations_2n)
    
    print("\n" + "=" * 70)
    print(f"🚀 SIMULAÇÃO NUMBA JIT - {n_simulations} SIMULAÇÕES")
    print("=" * 70)
    print(f"📋 CSV de entrada: {csv_basename}")
    print(f"👥 Agentes: {n_agents}")
    print(f"🎯 Limite 2N: {max_iterations_2n} iterações")
    print(f"🔄 Iterações máx: {actual_max}")
    print(f"⚡ Execução: SEQUENCIAL (Numba JIT)")
    print(f"📊 Gráficos: {'Habilitados' if generate_plot else 'Desabilitados'}")
    print(f"🎲 Distribuição inicial: mode={initial_distribution_mode} ratio={majority_ratio:.2f}")
    print(f"🔥 Numba JIT: ATIVADO (5.20x speedup esperado)")
    print("=" * 70)
    
    # ========================================================================
    # MÉTRICAS - Diferença Conceitual:
    # - CONVERGÊNCIA: Atingir 100% em qualquer estado (K ou Z)
    # - ACURÁCIA/SUCESSO: Convergir para o estado da MAIORIA INICIAL
    # - ERRO: Convergir para o estado OPOSTO à maioria inicial
    # - FALHA: Não convergir dentro do limite de 2N iterações
    # ========================================================================
    
    # Métricas de acurácia (convergiu para estado correto?)
    correct_convergence_count = 0  # Convergiu para estado da maioria inicial
    correct_to_k_count = 0          # Convergiu corretamente para K
    correct_to_z_count = 0          # Convergiu corretamente para Z
    
    # Métricas de erro/falha
    wrong_convergence_count = 0     # Convergiu para estado oposto
    no_convergence_count = 0        # Não convergiu (ficou sem consenso)
    
    # Totais
    total_convergence_count = 0     # Total que convergiu (correto + errado)
    
    # Auxiliares
    total_rounds_to_success = []    # Rodadas até convergência correta
    
    # Lista para armazenar resultados de cada simulação para CSV
    simulations_results = []
    
    # Execução SEQUENCIAL (mais rápido para simulações rápidas!)
    print(f"\n⚡ Executando {n_simulations} simulações sequencialmente...")
    start_time = time.time()
    
    for sim_num in range(1, n_simulations + 1):
        (
            initial_config,
            initial_majority,
            count_0,
            count_1,
            used_majority_ratio,
            used_distribution_mode,
        ) = generate_initial_configuration_shared(
            n_agents=n_agents,
            seed_distribution=sim_num,
            majority_ratio=majority_ratio,
            distribution_mode=initial_distribution_mode,
        )
        
        # Salvar distribuição inicial
        sim_dir = os.path.join(base_output_dir, f"sim{sim_num:03d}")
        ensure_directory_exists(sim_dir)
        
        initial_config_list = initial_config.tolist()
        
        initial_config_metadata = {
            'sim_number': sim_num,
            'seed': sim_num,
            'seed_distribution': sim_num,
            'n_agents': n_agents,
            'majority_ratio': used_majority_ratio,
            'initial_distribution_mode': used_distribution_mode,
            'initial_majority_opinion': int(initial_majority),
            'initial_config': initial_config_list,
            'rodada_0': initial_config_list,  # Distribuição inicial da primeira rodada (0 e 1)
            'rodada_0_stats': {
                'count_0': count_0,
                'count_1': count_1,
                'percent_0': (count_0 / n_agents) * 100,
                'percent_1': (count_1 / n_agents) * 100
            },
            'opinion_labels': {'0': OPINION_LABELS[0], '1': OPINION_LABELS[1]},
            'timestamp': time.strftime('%Y%m%d_%H%M%S')
        }
        
        initial_config_path = os.path.join(sim_dir, f"initial_config_sim{sim_num:03d}.json")
        with open(initial_config_path, 'w', encoding='utf-8') as f:
            json.dump(initial_config_metadata, f, indent=2)
        
        # Simula
        history, success, success_round = simulate_automaton_run_numba(
            initial_config, rules_array, window_size, actual_max, initial_majority
        )
        
        # Estatísticas
        consensus_reached = check_consensus_numba(history[-1])
        
        if success:
            # Convergiu CORRETAMENTE (para estado da maioria inicial)
            correct_convergence_count += 1
            total_convergence_count += 1
            total_rounds_to_success.append(success_round)
            
            if history[-1][0] == OPINION_K:
                correct_to_k_count += 1
            else:
                correct_to_z_count += 1
        else:
            # Não convergiu corretamente
            if consensus_reached:
                # Convergiu INCORRETAMENTE (para estado oposto)
                wrong_convergence_count += 1
                total_convergence_count += 1
            else:
                # NÃO convergiu (sem consenso)
                no_convergence_count += 1
        
        # Coletar dados de cada simulação para CSV
        final_opinion = int(history[-1][0])
        initial_k_count = int(np.sum(initial_config == OPINION_K))
        initial_z_count = int(np.sum(initial_config == OPINION_Z))
        final_k_count = int(np.sum(history[-1] == OPINION_K))
        final_z_count = int(np.sum(history[-1] == OPINION_Z))
        n_rounds = len(history) - 1
        # consensus_reached já foi calculado acima na seção de estatísticas
        
        sim_result = {
            'sim_number': sim_num,
            'success': success,
            'initial_majority': OPINION_LABELS[initial_majority],
            'final_opinion': OPINION_LABELS[final_opinion],
            'rounds': success_round if success_round is not None else n_rounds,
            'initial_k_count': initial_k_count,
            'initial_z_count': initial_z_count,
            'initial_k_percent': (initial_k_count / n_agents) * 100,
            'initial_z_percent': (initial_z_count / n_agents) * 100,
            'final_k_count': final_k_count,
            'final_z_count': final_z_count,
            'final_k_percent': (final_k_count / n_agents) * 100,
            'final_z_percent': (final_z_count / n_agents) * 100,
            'consensus_reached': consensus_reached,
            'correct_consensus': success,
            'majority_ratio': used_majority_ratio,
            'initial_distribution_mode': used_distribution_mode
        }
        simulations_results.append(sim_result)
        
        # Plot
        if generate_plot:
            rule_name = f"{csv_basename}_sim{sim_num:03d}"
            filename_prefix = f"sim{sim_num:03d}"
            final_opinion = int(history[-1][0])
            consensus_info = (success_round, final_opinion) if success_round is not None else None
            
            plot_automaton_history(
                history, sim_dir, filename_prefix, n_agents, rule_name,
                consensus_info, success=success, initial_majority=initial_majority,
                generate_plot=generate_plot
            )
        
        if sim_num % 10 == 0 or sim_num == 1:
            print(f"📊 Simulação {sim_num}/{n_simulations} concluída")
    
    total_time = time.time() - start_time
    
    # Relatório
    convergence_rate = (total_convergence_count / n_simulations) * 100
    accuracy_rate = (correct_convergence_count / n_simulations) * 100
    error_rate = (wrong_convergence_count / n_simulations) * 100
    no_conv_rate = (no_convergence_count / n_simulations) * 100
    
    print("\n" + "=" * 70)
    print("📊 RELATÓRIO FINAL")
    print("=" * 70)
    print(f"⏱️ Tempo total: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"⏱️ Tempo médio: {total_time/n_simulations:.3f}s por simulação")
    print(f"🎯 Total: {n_simulations} simulações")
    print()
    print(f"📈 CONVERGÊNCIA TOTAL: {total_convergence_count}/{n_simulations} ({convergence_rate:.1f}%)")
    print(f"   ✅ Convergência CORRETA: {correct_convergence_count}/{n_simulations} ({accuracy_rate:.1f}%)")
    print(f"      - Para K (correto): {correct_to_k_count}")
    print(f"      - Para Z (correto): {correct_to_z_count}")
    print(f"   ❌ Convergência INCORRETA: {wrong_convergence_count}/{n_simulations} ({error_rate:.1f}%)")
    print(f"      - Para estado oposto: {wrong_convergence_count}")
    print(f"   ⚠️  SEM convergência: {no_convergence_count}/{n_simulations} ({no_conv_rate:.1f}%)")
    
    if total_rounds_to_success:
        avg_rounds = sum(total_rounds_to_success) / len(total_rounds_to_success)
        print(f"\n📈 Rodadas até convergência CORRETA:")
        print(f"   - Média: {avg_rounds:.2f}")
        print(f"   - Mínimo: {min(total_rounds_to_success)}")
        print(f"   - Máximo: {max(total_rounds_to_success)}")
    
    # Salva relatório em arquivo
    report_path = os.path.join(base_output_dir, f"relatorio_{csv_basename}.txt")
    ensure_directory_exists(base_output_dir)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"RELATÓRIO DE SIMULAÇÕES - {csv_basename}\n")
        f.write(f"CRITÉRIO: Consenso na Maioria Inicial em até 2N iterações\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"CSV de entrada: {csv_path}\n")
        f.write(f"Total de simulações: {n_simulations}\n")
        f.write(f"Agentes por simulação: {n_agents}\n")
        f.write(f"Distribuição inicial: mode={initial_distribution_mode}, ratio={majority_ratio:.2f}\n")
        f.write(f"Limite 2N: {max_iterations_2n} iterações\n")
        f.write(f"Tempo total: {total_time:.2f}s ({total_time/60:.2f} min)\n")
        f.write(f"Tempo médio: {total_time/n_simulations:.3f}s por simulação\n\n")
        f.write(f"RESULTADOS:\n")
        f.write(f"Taxa de CONVERGÊNCIA: {total_convergence_count}/{n_simulations} ({convergence_rate:.1f}%)\n")
        f.write(f"Taxa de ACURÁCIA (convergiu corretamente): {correct_convergence_count}/{n_simulations} ({accuracy_rate:.1f}%)\n")
        f.write(f"  - Convergência correta para K: {correct_to_k_count}\n")
        f.write(f"  - Convergência correta para Z: {correct_to_z_count}\n")
        f.write(f"Taxa de ERRO (convergiu incorretamente): {wrong_convergence_count}/{n_simulations} ({error_rate:.1f}%)\n")
        f.write(f"  - Convergiu para estado oposto: {wrong_convergence_count}\n")
        f.write(f"  - Sem convergência alguma: {no_convergence_count}\n")
        
        if total_rounds_to_success:
            f.write(f"\nRODADAS ATÉ CONVERGÊNCIA CORRETA:\n")
            f.write(f"- Média: {avg_rounds:.2f}\n")
            f.write(f"- Mínimo: {min(total_rounds_to_success)}\n")
            f.write(f"- Máximo: {max(total_rounds_to_success)}\n")
    
    print(f"\n📄 Relatório salvo em: {report_path}")
    
    # Salvar CSV com resultados individuais de cada simulação
    csv_results_path = os.path.join(base_output_dir, "resultados_individuais.csv")
    df_results = pd.DataFrame(simulations_results)
    df_results.to_csv(csv_results_path, index=False)
    print(f"\n📊 CSV de resultados individuais salvo em: {csv_results_path}")
    print(f"   - {len(df_results)} simulações registradas")
    
    # Estatísticas do CSV
    print(f"\n📈 Resumo do CSV:")
    print(f"   - Convergências corretas: {df_results['success'].sum()} ({df_results['success'].sum()/len(df_results)*100:.1f}%)")
    print(f"   - Convergências incorretas: {(~df_results['success'] & df_results['consensus_reached']).sum()} ({(~df_results['success'] & df_results['consensus_reached']).sum()/len(df_results)*100:.1f}%)")
    print(f"   - Sem convergência: {(~df_results['consensus_reached']).sum()} ({(~df_results['consensus_reached']).sum()/len(df_results)*100:.1f}%)")
    print(f"   - Convergência total: {df_results['consensus_reached'].sum()} ({df_results['consensus_reached'].sum()/len(df_results)*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("✅ SIMULAÇÕES CONCLUÍDAS COM NUMBA JIT!")
    print("=" * 70)
    
    return {
        # NOVAS métricas (nomenclatura correta)
        'convergence_rate_percent': round(convergence_rate, 1),
        'accuracy_rate_percent': round(accuracy_rate, 1),
        'error_rate_percent': round(error_rate, 1),
        'correct_convergence_count': correct_convergence_count,
        'correct_to_k_count': correct_to_k_count,
        'correct_to_z_count': correct_to_z_count,
        'wrong_convergence_count': wrong_convergence_count,
        'no_convergence_count': no_convergence_count,
        'total_convergence_count': total_convergence_count,
        
        # ANTIGAS métricas (mantidas para retrocompatibilidade)
        'success_count': correct_convergence_count,  # Agora aponta para correct_convergence
        'success_k_count': correct_to_k_count,
        'success_z_count': correct_to_z_count,
        'fail_count': wrong_convergence_count + no_convergence_count,
        'fail_wrong_consensus': wrong_convergence_count,
        'fail_no_consensus': no_convergence_count,
        'success_rate_percent': round(accuracy_rate, 1),  # Na verdade é acurácia
        'fail_wrong_consensus_rate_percent': round(error_rate, 1) if n_simulations > 0 else 0,
        
        # Outras métricas
        'min_rounds': min(total_rounds_to_success) if total_rounds_to_success else None,
        'max_rounds': max(total_rounds_to_success) if total_rounds_to_success else None,
        'n_agents': n_agents,
        'max_iterations_2n': max_iterations_2n,
        'simulation_time': total_time,
        'total_simulations': n_simulations,
        'ici': ici,
        'token_bias': token_bias
    }

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_csv_metadata(csv_path):
    basename = os.path.basename(csv_path)
    model_map = {
        'google_gemma-3-4b-it': 'gemma4b',
        'google_gemma-3-4b': 'gemma4b',
        'gemma-3-4b-it': 'gemma4b',
        'gemma-3-4b': 'gemma4b',
        'google_gemma-3-12b': 'gemma12b',
        'gemma-3-27b-it': 'gemma27b',
        'qwen_qwen3-14b': 'qwen14b',
    }
    model = next((v for k, v in model_map.items() if k in csv_path), 'unknown')
    
    match_n = re.search(r'_n(\d+)_', basename)
    n_window = int(match_n.group(1)) if match_n else 0
    
    # Detecta COT: v20 (raciocinio) ou v21 (zero_shot_cot)
    is_cot = 'zero_shot_cot' in basename or ('v20_' in basename and 'raciocinio' in basename)
    prompt_strategy = 'com_cot' if is_cot else 'only_token'
    
    # Extraímos o prompt_variant do padrão do arquivo para evitar falso positivo
    # por timestamp (ex.: "..._20260303_013810_..." contendo "01").
    variant_match = re.search(r'^dados_combinados_(.+?)_n\d+_exp\d+_iter\d+_', basename)
    prompt_variant = variant_match.group(1) if variant_match else ''

    token_map = {
        'yesno': 'no/yes',
        'αβ': 'α/β',
        '△○': '△/○',
        '⊕⊖': '⊕/⊖',
        'łþ': 'ł/þ',
        'kz': 'k/z',
        'ab': 'a/b',
        'pq': 'p/q',
        '01': '0/1'
    }

    token_type = None
    for key, mapped in token_map.items():
        if prompt_variant.endswith(f"_{key}") or prompt_variant == key:
            token_type = mapped
            break

    if not token_type:
        # Casos históricos sem sufixo explícito no variant (ex.: v21_zero_shot_cot): k/z
        token_type = 'k/z'
    
    return {'model': model, 'n_window': n_window, 'token_type': token_type, 'prompt_strategy': prompt_strategy}

def write_consolidated_csv_header(csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['csv_source', 'n_window', 'token_type', 'model', 'prompt_strategy', 
                        'token_bias', 'ici', 'total_simulations', 'success_count', 'success_k_count', 
                        'success_z_count', 'fail_count', 'fail_wrong_consensus', 'fail_no_consensus',
                        'success_rate_percent', 'fail_wrong_consensus_rate_percent', 
                        'min_rounds_to_success', 'max_rounds_to_success', 'n_agents', 
                        'max_iterations_2n', 'extraction_time_sec', 'simulation_time_sec', 'total_time_sec'])

def append_to_consolidated_csv(csv_path, data):
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            data['csv_source'], data['n_window'], data['token_type'], data['model'], 
            data['prompt_strategy'], data['token_bias'], data['ici'],
            data['total_simulations'], data['success_count'], data['success_k_count'], 
            data['success_z_count'], data['fail_count'], data['fail_wrong_consensus'], 
            data['fail_no_consensus'], data['success_rate_percent'], 
            data['fail_wrong_consensus_rate_percent'], data['min_rounds'], data['max_rounds'], 
            data['n_agents'], data['max_iterations_2n'], data['extraction_time'], 
            data['simulation_time'], data['total_time']
        ])

def find_all_configs():
    base_dir = "../resultados_antes_do_teste_de_diferenca_grande_de_modelo"
    csvs = []
    for modelo in ['gemma12b', 'gemma27b', 'qwen14b']:
        for n in ['n7', 'n9', 'n11']:
            pattern = f"{base_dir}/{modelo}/{n}/csv/basic/dados_combinados_*.csv"
            csvs.extend(glob.glob(pattern))
    return sorted([c for c in csvs if 'dados_estendidos' not in c])

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Script único para simulações de autômato com Numba JIT. Aceita CSV, extrai regras automaticamente se necessário.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Uso básico (100 sims, 100 agentes, com gráficos)
  python run_automaton_numba.py --csv dados.csv

  # Personalizado (50 sims, 200 agentes, sem gráficos)
  python run_automaton_numba.py --csv dados.csv --n_simulations 50 --agents 200 --no-plots

  # Forçar re-extração de regras
  python run_automaton_numba.py --csv dados.csv --force-extract

  # Processar todos os 162 CSVs automaticamente
  python run_automaton_numba.py --all-configs
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv', type=str,
                        help='Caminho para o CSV de entrada')
    group.add_argument('--all-configs', action='store_true',
                        help='Processar automaticamente os 162 CSVs (n=7, n=9, n=11, apenas /basic/)')
    
    parser.add_argument('--n_simulations', type=int, default=100,
                        help='Número de simulações (default: 100)')
    parser.add_argument('--agents', type=int, default=100,
                        help='Número de agentes (default: 100)')
    parser.add_argument('--max_iterations', type=int, default=200,
                        help='Máximo de iterações por simulação (default: 200)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Desabilita geração de gráficos PNG')
    parser.add_argument('--force-extract', action='store_true',
                        help='Força re-extração de regras mesmo se JSON existir')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Nome do arquivo CSV de saída (default: all_configurations_results.csv)')
    parser.add_argument('--initial-ratio', type=float, default=0.51,
                        help='Proporção inicial da maioria (default: 0.51 = 51%%)')
    parser.add_argument('--initial-distribution-mode', choices=['auto', 'ratio', 'half_split'], default='auto',
                        help='Modo da distribuição inicial: auto|ratio|half_split (auto usa half_split só quando ratio=0.5)')
    parser.add_argument('--source-tag', type=str, default=None,
                        help='Subpasta opcional antes de <agents_pct> (ex.: macstudio)')
    
    args = parser.parse_args()
    
    print(f"👥 Executando com {args.agents} agentes")
    print(f"🚀 VERSÃO NUMBA JIT (5.20x SPEEDUP)")
    
    if args.all_configs:
        csvs = find_all_configs()
        print(f"📊 Processando {len(csvs)} CSVs...")
        
        consolidated_csv = args.output_csv if args.output_csv else "all_configurations_results.csv"
        write_consolidated_csv_header(consolidated_csv)
        
        for i, csv_path in enumerate(csvs, 1):
            print(f"\n{'='*70}")
            print(f"[{i}/{len(csvs)}] {os.path.basename(csv_path)}")
            print(f"{'='*70}")
            
            start_total = time.time()
            stats = run_simulations_from_csv(
                csv_path=csv_path,
                n_simulations=args.n_simulations,
                n_agents=args.agents,
                max_iterations=args.max_iterations,
                generate_plot=not args.no_plots,
                force_extract=args.force_extract,
                majority_ratio=args.initial_ratio,
                initial_distribution_mode=args.initial_distribution_mode,
                source_tag=args.source_tag
            )
            total_time = time.time() - start_total
            
            metadata = parse_csv_metadata(csv_path)
            result_data = {
                'csv_source': os.path.basename(csv_path),
                'n_window': metadata['n_window'],
                'token_type': metadata['token_type'],
                'model': metadata['model'],
                'prompt_strategy': metadata['prompt_strategy'],
                'token_bias': stats['token_bias'],
                'ici': stats['ici'],
                'total_simulations': stats['total_simulations'],
                'success_count': stats['success_count'],
                'success_k_count': stats['success_k_count'],
                'success_z_count': stats['success_z_count'],
                'fail_count': stats['fail_count'],
                'fail_wrong_consensus': stats['fail_wrong_consensus'],
                'fail_no_consensus': stats['fail_no_consensus'],
                'success_rate_percent': stats['success_rate_percent'],
                'fail_wrong_consensus_rate_percent': stats['fail_wrong_consensus_rate_percent'],
                'min_rounds': stats['min_rounds'],
                'max_rounds': stats['max_rounds'],
                'n_agents': stats['n_agents'],
                'max_iterations_2n': stats['max_iterations_2n'],
                'extraction_time': 0,
                'simulation_time': stats['simulation_time'],
                'total_time': total_time
            }
            append_to_consolidated_csv(consolidated_csv, result_data)
        
        print(f"\n📄 CSV consolidado salvo em: {consolidated_csv}")
    else:
        if not os.path.exists(args.csv):
            print(f"❌ ERRO: CSV não encontrado: {args.csv}")
            sys.exit(1)
        run_simulations_from_csv(
            csv_path=args.csv,
            n_simulations=args.n_simulations,
            n_agents=args.agents,
            max_iterations=args.max_iterations,
            generate_plot=not args.no_plots,
            force_extract=args.force_extract,
            majority_ratio=args.initial_ratio,
            initial_distribution_mode=args.initial_distribution_mode,
            source_tag=args.source_tag
        )

if __name__ == "__main__":
    main()
