#!/usr/bin/env python3
"""
Generate all final result images in English.
Outputs everything to: resultados_finais_en/

Generates:
  1. ICI Ranking tables (basic: top/bottom per model, neighborhood, global)
  2. ICI Ranking tables (extended: with efficiency columns E51-E70)
  3. Token Bias plots (all tokens, focused 2-token)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import re

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, 'resultados_finais_en')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROJECT_ROOT = os.path.dirname(BASE)          # .../projeto_final
REPO_ROOT = os.path.dirname(PROJECT_ROOT)     # .../conformidade_experimento_resultados

RESULTS_OLD = os.path.join(REPO_ROOT, 'resultados_antes_do_teste_de_diferenca_grande_de_modelo')
RESULTS_SERVER = os.path.join(REPO_ROOT, 'resultados_servidor')
AUTOMATOS_DIR = os.path.join(PROJECT_ROOT, 'experimentos_automatos')
EXTRACT_RULES_A100 = os.path.join(PROJECT_ROOT, 'extract_rules', 'A100')
GEMMA27B_EXTRACT_DIR = os.path.join(EXTRACT_RULES_A100, 'gemma27b')
GEMMA27B_AUTOMATON_RESULTS_DIR = os.path.join(AUTOMATOS_DIR, 'resultados', 'gemma27b')
TARGET_MODEL_ONLY = 'Gemma 27B'
TARGET_AUTOMATON_MAJORITY_PCTS = {51, 55, 60, 65, 70}


# ==============================================================================
# SHARED HELPERS
# ==============================================================================

def invert_config(config_str):
    config_str = str(config_str)
    if 'yes' in config_str or 'no' in config_str:
        return config_str.replace('yes', '___T___').replace('no', 'yes').replace('___T___', 'no')
    mapping = {
        'α': 'β', 'β': 'α', '△': '○', '○': '△', '⊕': '⊖', '⊖': '⊕',
        'p': 'q', 'q': 'p', 'ł': 'þ', 'þ': 'ł',
        '0': '1', '1': '0', 'a': 'b', 'b': 'a', 'k': 'z', 'z': 'k'
    }
    return ''.join(mapping.get(c, c) for c in config_str)


def extract_file_info(filepath):
    """Returns (format_str, token, neighborhood) from filename."""
    filename = os.path.basename(filepath)

    if 'v21_zero_shot_cot' in filename or 'v20_lista_completa_meio_raciocinio_primeiro' in filename:
        fmt = 'with_cot'
    elif 'v9_lista_completa_meio' in filename:
        fmt = 'only_token'
    else:
        fmt = 'unknown'

    if '_n7_' in filename:
        nbhood = 'n7'
    elif '_n9_' in filename:
        nbhood = 'n9'
    elif '_n11_' in filename:
        nbhood = 'n11'
    elif '_n3_' in filename:
        nbhood = 'n3'
    else:
        nbhood = 'unknown'

    parts = filename.split('_')
    token = None
    for i, part in enumerate(parts):
        if part in ('meio', 'cot') and i + 1 < len(parts):
            candidate = parts[i + 1]
            if candidate not in ('n7', 'n9', 'n11', 'n3', 'n5', 'exp', 'iter'):
                token = candidate
            break

    if token is None or token in ('n7', 'n9', 'n11', 'n3'):
        token = nbhood

    token_map = {
        'αβ': 'αβ', '△○': '△○', '⊕⊖': '⊕⊖', 'pq': 'pq', 'łþ': 'łþ',
        '01': '01', 'ab': 'ab', 'kz': 'kz', 'yesno': 'yesno',
        'n7': 'n7', 'n9': 'n9', 'n11': 'n11'
    }
    token = token_map.get(token, token)
    return fmt, token, nbhood


def detect_token_from_csv(df):
    col = 'configuracao_letras' if 'configuracao_letras' in df.columns else (
        'distribuicao_vizinhos' if 'distribuicao_vizinhos' in df.columns else None)
    if col is None:
        return None
    samples = df[col].head(20).tolist()
    text = ''.join(str(s) for s in samples)
    if 'yes' in text or 'no' in text:
        return 'yesno'
    chars = set(text)
    mapping = {
        frozenset('01'): '01', frozenset('ab'): 'ab', frozenset('pq'): 'pq',
        frozenset('kz'): 'kz', frozenset('łþ'): 'łþ', frozenset('αβ'): 'αβ',
        frozenset('△○'): '△○', frozenset('⊕⊖'): '⊕⊖'
    }
    return mapping.get(frozenset(chars))


def collect_ici_data(base_dir, model_name):
    """Calculates ICI and Token Bias for all experiments in base_dir."""
    results = []
    for filepath in glob.glob(os.path.join(base_dir, '*.csv')):
        try:
            df = pd.read_csv(filepath,
                             dtype={'configuracao_letras': str, 'distribuicao_vizinhos': str},
                             keep_default_na=False)
            col_cfg = ('configuracao_letras' if 'configuracao_letras' in df.columns
                       else 'distribuicao_vizinhos' if 'distribuicao_vizinhos' in df.columns
                       else None)
            if col_cfg is None:
                continue

            lookup = {row[col_cfg]: int(row['escolha']) for _, row in df.iterrows()}

            consistent_pairs = total_pairs = 0
            sum_bias = count0 = count1 = 0
            processed = set()

            for cfg in lookup:
                if cfg in processed:
                    continue
                cfg_inv = invert_config(cfg)
                if cfg_inv in lookup:
                    c = lookup[cfg]
                    c_inv = lookup[cfg_inv]
                    if c_inv == (1 - c):
                        consistent_pairs += 1
                    sum_bias += abs((2 * c - 1) + (2 * c_inv - 1)) / 2.0
                    count0 += (c == 0) + (c_inv == 0)
                    count1 += (c == 1) + (c_inv == 1)
                    total_pairs += 1
                    processed.add(cfg)
                    processed.add(cfg_inv)

            ici = consistent_pairs / total_pairs if total_pairs > 0 else 0

            fmt, token, nbhood = extract_file_info(filepath)
            if token in ('n7', 'n9', 'n11', 'n3', 'n5', 'unknown'):
                detected = detect_token_from_csv(df)
                if detected:
                    token = detected

            token_pairs = {
                'αβ': ('α', 'β'), '△○': ('△', '○'), '⊕⊖': ('⊕', '⊖'),
                'pq': ('p', 'q'), 'łþ': ('ł', 'þ'), '01': ('0', '1'),
                'ab': ('a', 'b'), 'kz': ('k', 'z'), 'yesno': ('no', 'yes'),
            }
            if total_pairs > 0 and token in token_pairs:
                delta = (count1 - count0) / (2 * total_pairs)
                t0, t1 = token_pairs[token]
                token_bias = f"{t0}/{t1} ({delta:+.4f})"
            else:
                token_bias = 'N/A'

            results.append({
                'Model': model_name, 'Neighborhood': nbhood, 'Format': fmt,
                'Token': token, 'ICI': ici, 'Token_Bias': token_bias,
            })
        except Exception as e:
            print(f"  [warn] {os.path.basename(filepath)}: {e}")
    return results


def fmt_token(token):
    return 'no/yes' if token == 'yesno' else token


def fmt_format(fmt, short=False):
    if short:
        return 'CoT' if fmt == 'with_cot' else 'Tok'
    return fmt


def _to_bool_series(series):
    return series.astype(str).str.strip().str.lower().isin({'1', 'true', 'yes', 'y'})


HEADER_COLOR = '#34495e'
TOP_ROW_COLOR = '#d4edda'
BTM_ROW_COLOR = '#f8d7da'


# ==============================================================================
# 1. BASIC RANKING TABLES (ICI + Token Bias)
# ==============================================================================

def create_top_bottom_table(data_sorted, title, output_path_base, top_n=10):
    if not data_sorted:
        print(f"  [skip] No data for: {title}")
        return

    dir_path = os.path.dirname(output_path_base)
    filename = os.path.basename(output_path_base)
    base_name = filename.replace('top_bottom_', '').replace('.png', '')
    top_path = os.path.join(dir_path, f'top_{base_name}.png')
    bot_path = os.path.join(dir_path, f'bottom_{base_name}.png')

    headers = ['Rank', 'Model', 'N', 'Format', 'Token', 'ICI', 'Token Bias']
    col_widths = [0.10, 0.16, 0.10, 0.18, 0.14, 0.14, 0.22]

    def _save_table(rows_data, subtitle, path, row_color):
        n_rows = len(rows_data) + 1
        fig, ax = plt.subplots(figsize=(20, n_rows * 0.22 + 2.5))
        ax.axis('off')
        fig.text(0.5, 0.990, title, ha='center', va='top', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.950, subtitle, ha='center', va='top', fontsize=12)

        table = ax.table(cellText=[headers] + rows_data, cellLoc='center',
                         loc='center', colWidths=col_widths)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
        for j in range(len(headers)):
            cell = table[(0, j)]
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(weight='bold', color='white', size=10)
        for i in range(1, len(rows_data) + 1):
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor(row_color)
                if j == 0:
                    cell.set_text_props(weight='bold', size=10)
                if j == 5:
                    cell.set_text_props(weight='bold', size=9)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.02)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.relpath(path, BASE)}")

    top_rows = [[f"{i+1}.", d['Model'], d['Neighborhood'], d['Format'],
                 fmt_token(d['Token']), f"{d['ICI']:.4f}", d['Token_Bias']]
                for i, d in enumerate(data_sorted[:top_n])]
    _save_table(top_rows, f'TOP {top_n} (Highest ICI)', top_path, TOP_ROW_COLOR)

    start = len(data_sorted) - top_n
    bot_rows = [[f"{start + i + 1}.", d['Model'], d['Neighborhood'], d['Format'],
                 fmt_token(d['Token']), f"{d['ICI']:.4f}", d['Token_Bias']]
                for i, d in enumerate(data_sorted[-top_n:])]
    _save_table(bot_rows, f'BOTTOM {top_n} (Lowest ICI)', bot_path, BTM_ROW_COLOR)


def create_combined_top_bottom_table(data_sorted, title, output_path, top_n=10):
    """Creates a single image with TOP N and BOTTOM N tables stacked vertically."""
    if not data_sorted:
        print(f"  [skip] No data for: {title}")
        return

    top_data = data_sorted[:top_n]
    bot_data = data_sorted[-top_n:]
    start_rank = len(data_sorted) - top_n

    headers = ['Rank', 'Model', 'N', 'Format', 'Token', 'ICI', 'Token Bias']
    col_widths = [0.09, 0.15, 0.09, 0.16, 0.13, 0.13, 0.22]

    top_rows = [[f"{i+1}.", d['Model'], d['Neighborhood'], d['Format'],
                 fmt_token(d['Token']), f"{d['ICI']:.4f}", d['Token_Bias']]
                for i, d in enumerate(top_data)]
    bot_rows = [[f"{start_rank + i + 1}.", d['Model'], d['Neighborhood'], d['Format'],
                 fmt_token(d['Token']), f"{d['ICI']:.4f}", d['Token_Bias']]
                for i, d in enumerate(bot_data)]

    n_rows = top_n + 1  # header + data rows per table
    row_h = 0.38        # inches per row
    header_h = 0.5      # inches for the sub-heading (TOP / BOTTOM)
    title_h = 0.55      # inches for the main title at the top
    sep_h = 0.25        # gap between the two tables
    total_h = title_h + 2 * (header_h + n_rows * row_h) + sep_h

    fig = plt.figure(figsize=(20, total_h))

    # ── shared title ──────────────────────────────────────────────────────────
    fig.text(0.5, 1 - title_h / total_h * 0.5,
             title, ha='center', va='center', fontsize=17, fontweight='bold')

    def _add_table(rows_data, subtitle, y_top, row_color):
        """y_top: fraction from bottom where this block starts (top edge)."""
        block_h = header_h + n_rows * row_h
        # subtitle
        sub_y = y_top - header_h / total_h * 0.5
        fig.text(0.5, sub_y, subtitle, ha='center', va='center', fontsize=12)

        # axes for table
        ax_h = (n_rows * row_h) / total_h
        ax_y = y_top - header_h / total_h - ax_h
        ax = fig.add_axes([0.03, ax_y, 0.94, ax_h])
        ax.axis('off')

        table = ax.table(cellText=[headers] + rows_data,
                         cellLoc='center', loc='center', colWidths=col_widths)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.35)

        for j in range(len(headers)):
            cell = table[(0, j)]
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(weight='bold', color='white', size=10)
        for i in range(1, len(rows_data) + 1):
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor(row_color)
                if j == 0:
                    cell.set_text_props(weight='bold', size=10)
                if j == 5:
                    cell.set_text_props(weight='bold', size=9)

    # TOP block — starts just below the title
    top_y = 1 - title_h / total_h
    _add_table(top_rows, f'TOP {top_n}  (Highest ICI)', top_y, TOP_ROW_COLOR)

    # BOTTOM block — below the top block + separator
    bot_y = top_y - (header_h + n_rows * row_h + sep_h) / total_h
    _add_table(bot_rows, f'BOTTOM {top_n}  (Lowest ICI)', bot_y, BTM_ROW_COLOR)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(output_path, BASE)}")


def generate_basic_ranking_tables(all_data, out_dir):
    print("\n[1] Generating basic ICI ranking tables...")
    if not all_data:
        print("  [skip] No ICI data available.")
        return

    all_sorted = sorted(all_data, key=lambda x: x['ICI'], reverse=True)

    models_present = sorted(set(x['Model'] for x in all_data))
    for model in models_present:
        d = sorted([x for x in all_data if x['Model'] == model],
                   key=lambda x: x['ICI'], reverse=True)
        safe = model.lower().replace(' ', '')
        create_top_bottom_table(d, f'ICI RANKING: {model} (n7, n9, n11)',
                                os.path.join(out_dir, f'top_bottom_{safe}.png'))

    n_configs = {'n7': 128, 'n9': 512, 'n11': 2048}
    for n in sorted(set(x['Neighborhood'] for x in all_data)):
        d = sorted([x for x in all_data if x['Neighborhood'] == n],
                   key=lambda x: x['ICI'], reverse=True)
        n_cfg = n_configs.get(n, '?')
        create_top_bottom_table(
            d, f'ICI RANKING: {n.upper()} ({n_cfg} configurations) - All Models',
            os.path.join(out_dir, f'top_bottom_{n}.png'))

    create_top_bottom_table(
        all_sorted, 'GLOBAL ICI RANKING',
        os.path.join(out_dir, 'top_bottom_global.png'))

    # Combined (top + bottom in one slide-ready image)
    create_combined_top_bottom_table(
        all_sorted, 'GLOBAL ICI RANKING',
        os.path.join(out_dir, 'combined_global.png'))


# ==============================================================================
# 2. EXTENDED RANKING TABLES (ICI + Efficiency columns E51-E70)
# ==============================================================================

def load_automaton_data():
    print("  Loading automaton data...")
    dfs = []

    # Current Gemma27B results tree (.../resultados/gemma27b/**/resultados_individuais.csv)
    token_map = {
        'kz': 'k/z', 'ab': 'a/b', '01': '0/1', 'pq': 'p/q',
        'łþ': 'ł/þ', 'αβ': 'α/β', '△○': '△/○', '⊕⊖': '⊕/⊖',
        'yesno': 'no/yes', 'noyes': 'no/yes',
    }
    rx = re.compile(
        r"/gemma27b/(?P<token>[^/]+)/n(?P<n>\d+)/(?P<fmt>[^/]+)/(?P<tag>[^/]+)/"
        r"(?P<agents>\d+)agents_(?P<pct>\d+)pct/resultados_individuais\.csv$"
    )
    pattern = os.path.join(GEMMA27B_AUTOMATON_RESULTS_DIR, '**', 'resultados_individuais.csv')
    paths = glob.glob(pattern, recursive=True)
    rows = []
    for p in paths:
        m = rx.search(p)
        if not m:
            continue
        try:
            d = m.groupdict()
            token_raw = d['token']
            token_type = token_map.get(token_raw, token_raw)
            n_window = int(d['n'])
            n_agents = int(d['agents'])
            majority_pct = int(d['pct'])
            if majority_pct not in TARGET_AUTOMATON_MAJORITY_PCTS:
                continue

            rdf = pd.read_csv(p)
            if 'correct_consensus' in rdf.columns:
                ok = _to_bool_series(rdf['correct_consensus'])
            elif 'success' in rdf.columns:
                ok = _to_bool_series(rdf['success'])
            else:
                continue
            success_count = int(ok.sum())
            total_simulations = int(len(rdf))
            success_rate = float((success_count / total_simulations) * 100.0) if total_simulations > 0 else np.nan

            fmt_raw = d.get('fmt', '')
            fmt_norm = 'with_cot' if fmt_raw == 'com_cot' else fmt_raw

            rows.append({
                'model': TARGET_MODEL_ONLY,
                'token_type': token_type,
                'n_window': n_window,
                'format': fmt_norm,
                'success_rate_percent': success_rate,
                'success_count': success_count,
                'total_simulations': total_simulations,
                'n_agents_config': n_agents,
                'majority_pct_config': majority_pct,
            })
        except Exception:
            continue

    if rows:
        dfs.append(pd.DataFrame(rows))

    if not dfs:
        print("  [error] No gemma27b automaton CSV files found for target majority percentages.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(combined)} automaton records.")
    return combined


def get_efficiencies(model, token, nbhood, fmt, automaton_df):
    token_map = {
        'kz': 'k/z', 'ab': 'a/b', '01': '0/1', 'pq': 'p/q',
        'łþ': 'ł/þ', 'αβ': 'α/β', '△○': '△/○', '⊕⊖': '⊕/⊖', 'yesno': 'no/yes'
    }
    tok = token_map.get(token, token)
    n_num = int(nbhood.replace('n', '')) if 'n' in nbhood else None
    if n_num is None:
        return {p: np.nan for p in (51, 55, 60, 65, 70)}
    mask = (
        (automaton_df['token_type'] == tok)
        & (automaton_df['n_window'] == n_num)
        & (automaton_df['format'] == fmt)
    )
    if 'model' in automaton_df.columns:
        mask = mask & (automaton_df['model'] == model)
    subset = automaton_df[mask]
    result = {}
    for p in (51, 55, 60, 65, 70):
        s = subset[subset['majority_pct_config'] == p]
        if len(s) == 0:
            result[p] = np.nan
            continue
        # Efficiency by category (model + token + n + fmt + majority + agents):
        # compute hit-rate per n_agents, then average across agent groups.
        if 'n_agents_config' in s.columns:
            per_agents = (
                s.groupby('n_agents_config', as_index=False)
                 .agg(hits=('success_count', 'sum'),
                      total=('total_simulations', 'sum'))
            )
            per_agents['eff'] = np.where(
                per_agents['total'] > 0,
                (per_agents['hits'] / per_agents['total']) * 100.0,
                np.nan
            )
            result[p] = float(per_agents['eff'].mean()) if len(per_agents) > 0 else np.nan
        else:
            total = int(s['total_simulations'].sum()) if 'total_simulations' in s.columns else 0
            hits = int(s['success_count'].sum()) if 'success_count' in s.columns else 0
            result[p] = (hits / total) * 100.0 if total > 0 else np.nan
    return result


def eff_color(val):
    if np.isnan(val):
        return '#f0f0f0'
    if val >= 30:
        return '#1a9850'
    if val >= 20:
        return '#91cf60'
    if val >= 10:
        return '#fee08b'
    return '#d73027'


def fmt_eff(val):
    return '-' if np.isnan(val) else f"{val:.1f}%"


def create_extended_table(data_sorted, title, output_path_base, top_n=10, mode='full5', section='top'):
    if not data_sorted:
        print(f"  [skip] No data for: {title}")
        return

    if section == 'bottom':
        selected = data_sorted[-top_n:]
        rank_start = max(len(data_sorted) - top_n, 0)
        subtitle_prefix = f'BOTTOM {top_n} (Lowest ICI)'
    else:
        selected = data_sorted[:top_n]
        rank_start = 0
        subtitle_prefix = f'TOP {top_n} (Highest ICI)'

    if mode == 'compact3':
        suffix, props, width = '_3prop', (51, 65, 70), 11
        font_h, font_c = 9, 8
        headers = ['Rank', 'Model', 'N', 'Format', 'Token', 'ICI', 'E@51%', 'E@65%', 'E@70%']
        col_widths = [0.08, 0.15, 0.08, 0.13, 0.11, 0.11, 0.11, 0.11, 0.11]
        eff_keys = ['Efic_51', 'Efic_65', 'Efic_70']
    else:
        suffix, props, width = '_5prop', (51, 55, 60, 65, 70), 13
        font_h, font_c = 8, 7
        headers = ['Rank', 'Model', 'N', 'Fmt', 'Tok', 'ICI', 'E51', 'E55', 'E60', 'E65', 'E70']
        col_widths = [0.07, 0.13, 0.07, 0.09, 0.10, 0.10, 0.088, 0.088, 0.088, 0.088, 0.088]
        eff_keys = ['Efic_51', 'Efic_55', 'Efic_60', 'Efic_65', 'Efic_70']

    if section == 'bottom':
        output_path = output_path_base.replace('.png', f'_bottom{suffix}.png')
    else:
        output_path = output_path_base.replace('.png', f'{suffix}.png')
    props_text = ', '.join(f'{p}%' for p in props)

    n_rows = len(selected) + 1
    height = n_rows * 0.5 + 1.5
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis('off')
    fig.text(0.5, 0.990, title, ha='center', va='top', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.955, f'{subtitle_prefix} - Efficiencies: {props_text}',
             ha='center', va='top', fontsize=11)

    rows = [headers]
    for i, d in enumerate(selected, 1):
        base_row = [
            f"{rank_start + i}.", d['Model'], d['Neighborhood'],
            fmt_format(d['Format'], short=(mode == 'full5')),
            fmt_token(d['Token']),
            f"{d['ICI']:.4f}"
        ]
        eff_row = [fmt_eff(d.get(k, np.nan)) for k in eff_keys]
        rows.append(base_row + eff_row)

    table = ax.table(cellText=rows, cellLoc='center', loc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(font_c)
    table.scale(1, 1.4)

    n_eff_start = len(headers) - len(eff_keys)
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(weight='bold', color='white', size=font_h)

    for i, d in enumerate(selected, 1):
        for j in range(n_eff_start):
            cell = table[(i, j)]
            cell.set_facecolor(TOP_ROW_COLOR)
            if j == 0:
                cell.set_text_props(weight='bold', size=font_c + 1)
            elif j == 5:
                cell.set_text_props(weight='bold', size=font_c)
        for j, ek in enumerate(eff_keys, start=n_eff_start):
            cell = table[(i, j)]
            val = d.get(ek, np.nan)
            cell.set_facecolor(eff_color(val))
            if not np.isnan(val) and val >= 20:
                cell.set_text_props(color='white', weight='bold', size=font_c)
            else:
                cell.set_text_props(weight='bold', size=font_c)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(output_path, BASE)}")


def generate_extended_ranking_tables(all_data, out_dir):
    print("\n[2] Generating extended ICI ranking tables (with efficiency columns)...")
    if not all_data:
        print("  [skip] No ICI data available.")
        return

    automaton_df = load_automaton_data()
    if automaton_df.empty:
        print("  [skip] No automaton data available.")
        return

    # Attach efficiency columns
    extended = []
    for exp in all_data:
        effs = get_efficiencies(exp['Model'], exp['Token'], exp['Neighborhood'], exp['Format'], automaton_df)
        row = dict(exp)
        row['Efic_51'] = effs.get(51, np.nan)
        row['Efic_55'] = effs.get(55, np.nan)
        row['Efic_60'] = effs.get(60, np.nan)
        row['Efic_65'] = effs.get(65, np.nan)
        row['Efic_70'] = effs.get(70, np.nan)
        extended.append(row)

    all_sorted = sorted(extended, key=lambda x: x['ICI'], reverse=True)

    for mode in ('compact3', 'full5'):
        create_extended_table(
            all_sorted, 'GLOBAL ICI RANKING',
            os.path.join(out_dir, 'extended_global.png'), top_n=10, mode=mode, section='top')
        create_extended_table(
            all_sorted, 'GLOBAL ICI RANKING',
            os.path.join(out_dir, 'extended_global.png'), top_n=10, mode=mode, section='bottom')

    models_present = sorted(set(x['Model'] for x in extended))
    for model in models_present:
        d = sorted([x for x in extended if x['Model'] == model],
                   key=lambda x: x['ICI'], reverse=True)
        safe = model.lower().replace(' ', '')
        for mode in ('compact3', 'full5'):
            create_extended_table(
                d, f'ICI RANKING: {model} (n7, n9, n11)',
                os.path.join(out_dir, f'extended_{safe}.png'), top_n=10, mode=mode, section='top')
            create_extended_table(
                d, f'ICI RANKING: {model} (n7, n9, n11)',
                os.path.join(out_dir, f'extended_{safe}.png'), top_n=10, mode=mode, section='bottom')

    n_configs = {'n7': 128, 'n9': 512, 'n11': 2048}
    for n in sorted(set(x['Neighborhood'] for x in extended)):
        d = sorted([x for x in extended if x['Neighborhood'] == n],
                   key=lambda x: x['ICI'], reverse=True)
        n_cfg = n_configs.get(n, '?')
        for mode in ('compact3', 'full5'):
            create_extended_table(
                d, f'ICI RANKING: {n.upper()} ({n_cfg} configurations) - All Models',
                os.path.join(out_dir, f'extended_{n}.png'), top_n=10, mode=mode, section='top')
            create_extended_table(
                d, f'ICI RANKING: {n.upper()} ({n_cfg} configurations) - All Models',
                os.path.join(out_dir, f'extended_{n}.png'), top_n=10, mode=mode, section='bottom')


# ==============================================================================
# 3. GEMMA27B AGGREGATED AUTOMATON RESULTS (CSV)
# ==============================================================================

def collect_gemma27b_automaton_runs():
    """Collects one aggregated row per resultados_individuais.csv from gemma27b runs."""
    token_norm = {'noyes': 'yesno'}
    rx = re.compile(
        r"/gemma27b/(?P<token>[^/]+)/n(?P<n>\d+)/(?P<fmt>[^/]+)/(?P<tag>[^/]+)/"
        r"(?P<agents>\d+)agents_(?P<pct>\d+)pct/resultados_individuais\.csv$"
    )
    pattern = os.path.join(GEMMA27B_AUTOMATON_RESULTS_DIR, '**', 'resultados_individuais.csv')
    paths = glob.glob(pattern, recursive=True)

    rows = []
    for p in paths:
        m = rx.search(p)
        if not m:
            continue
        d = m.groupdict()
        try:
            rdf = pd.read_csv(p)
            if rdf.empty:
                continue

            if 'consensus_reached' in rdf.columns:
                consensus = _to_bool_series(rdf['consensus_reached'])
            elif 'success' in rdf.columns:
                consensus = _to_bool_series(rdf['success'])
            else:
                consensus = pd.Series([False] * len(rdf))

            if 'correct_consensus' in rdf.columns:
                correct = _to_bool_series(rdf['correct_consensus'])
            elif 'success' in rdf.columns:
                correct = _to_bool_series(rdf['success'])
            else:
                correct = pd.Series([False] * len(rdf))

            wrong = consensus & (~correct)
            no_conv = ~consensus

            rounds_col = pd.to_numeric(rdf['rounds'], errors='coerce') if 'rounds' in rdf.columns else pd.Series(dtype=float)
            rounds_when_correct = rounds_col[correct] if len(rounds_col) == len(correct) else pd.Series(dtype=float)
            majority_pct = int(d['pct'])
            if majority_pct not in TARGET_AUTOMATON_MAJORITY_PCTS:
                continue

            rows.append({
                'model': 'Gemma 27B',
                'token': token_norm.get(d['token'], d['token']),
                'n_window': int(d['n']),
                'format': d['fmt'],
                'source_tag': d['tag'],
                'n_agents': int(d['agents']),
                'majority_pct': majority_pct,
                'num_simulations': int(len(rdf)),
                'correct_rate_percent': float(correct.mean() * 100.0),
                'convergence_rate_percent': float(consensus.mean() * 100.0),
                'wrong_consensus_rate_percent': float(wrong.mean() * 100.0),
                'no_convergence_rate_percent': float(no_conv.mean() * 100.0),
                'mean_rounds_when_correct': float(rounds_when_correct.mean()) if len(rounds_when_correct) > 0 else np.nan,
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(
        by=['token', 'n_window', 'format', 'majority_pct', 'n_agents', 'source_tag']
    ).reset_index(drop=True)


def generate_gemma27b_aggregated_results(out_dir):
    print("\n[3] Generating Gemma 27B aggregated automaton results (CSV)...")
    df = collect_gemma27b_automaton_runs()
    if df.empty:
        print("  [warn] No gemma27b automaton runs found.")
        return

    detailed_path = os.path.join(out_dir, 'gemma27b_aggregated_detailed.csv')
    df.to_csv(detailed_path, index=False)
    print(f"  Saved: {os.path.relpath(detailed_path, BASE)} ({len(df)} rows)")

    by_setting = (
        df.groupby(['model', 'token', 'n_window', 'format', 'majority_pct', 'source_tag'], as_index=False)
          .agg(
              n_scenarios=('num_simulations', 'size'),
              total_simulations=('num_simulations', 'sum'),
              mean_correct_rate_percent=('correct_rate_percent', 'mean'),
              mean_convergence_rate_percent=('convergence_rate_percent', 'mean'),
              mean_wrong_consensus_rate_percent=('wrong_consensus_rate_percent', 'mean'),
              mean_no_convergence_rate_percent=('no_convergence_rate_percent', 'mean'),
              mean_rounds_when_correct=('mean_rounds_when_correct', 'mean'),
          )
    )
    by_setting_path = os.path.join(out_dir, 'gemma27b_aggregated_by_setting.csv')
    by_setting.to_csv(by_setting_path, index=False)
    print(f"  Saved: {os.path.relpath(by_setting_path, BASE)} ({len(by_setting)} rows)")

    by_window = (
        df.groupby(['model', 'n_window', 'format', 'majority_pct'], as_index=False)
          .agg(
              n_scenarios=('num_simulations', 'size'),
              total_simulations=('num_simulations', 'sum'),
              mean_correct_rate_percent=('correct_rate_percent', 'mean'),
              mean_convergence_rate_percent=('convergence_rate_percent', 'mean'),
              mean_wrong_consensus_rate_percent=('wrong_consensus_rate_percent', 'mean'),
              mean_no_convergence_rate_percent=('no_convergence_rate_percent', 'mean'),
              mean_rounds_when_correct=('mean_rounds_when_correct', 'mean'),
          )
    )
    by_window_path = os.path.join(out_dir, 'gemma27b_aggregated_by_window.csv')
    by_window.to_csv(by_window_path, index=False)
    print(f"  Saved: {os.path.relpath(by_window_path, BASE)} ({len(by_window)} rows)")


# ==============================================================================
# 4. TOKEN BIAS PLOTS
# ==============================================================================

def calculate_token_bias(df):
    col_cfg = ('configuracao_letras' if 'configuracao_letras' in df.columns
               else 'distribuicao_vizinhos' if 'distribuicao_vizinhos' in df.columns
               else None)
    if col_cfg is None:
        return None, None, None, None

    lookup = {str(row[col_cfg]): int(row['escolha']) for _, row in df.iterrows()}

    count0 = count1 = total = 0
    processed = set()
    for cfg in lookup:
        if cfg in processed:
            continue
        cfg_inv = invert_config(cfg)
        if cfg_inv in lookup:
            c, c_inv = lookup[cfg], lookup[cfg_inv]
            count0 += (c == 0) + (c_inv == 0)
            count1 += (c == 1) + (c_inv == 1)
            total += 1
            processed.add(cfg)
            processed.add(cfg_inv)

    if total > 0:
        return (count1 - count0) / (2 * total), count0, count1, total
    return None, None, None, None


def extract_token_bias_info(filepath):
    """Returns (model, token, neighborhood_int, format_str) from filename."""
    filename = os.path.basename(filepath)

    if 'llama-3.1-70b' in filename or 'llama_70b' in filename or 'llama70b' in filename:
        model = 'Llama 70B'
    elif 'llama-3.1-8b' in filename or 'llama_8b' in filename or 'llama8b' in filename:
        model = 'Llama 8B'
    elif 'qwen3-32b' in filename or 'qwen_qwen3_32b' in filename:
        model = 'Qwen 32B'
    elif 'qwen3-4b' in filename:
        model = 'Qwen 4B'
    elif 'google_gemma-3-4b' in filename or 'gemma-3-4b' in filename or 'gemma4b' in filename:
        model = 'Gemma 4B'
    elif 'google_gemma-3-27b' in filename or 'gemma-3-27b' in filename or 'gemma27b' in filename:
        model = 'Gemma 27B'
    else:
        model = 'Unknown'

    fmt = ('with_cot' if 'v21_zero_shot_cot' in filename
           else 'only_token' if 'v9_lista_completa_meio' in filename
           else 'unknown')

    nbhood = None
    for n in (3, 7, 9, 11):
        if f'_n{n}_' in filename:
            nbhood = n
            break

    tokens = ['αβ', '△○', '⊕⊖', 'pq', 'łþ', '01', 'ab', 'kz', 'yesno', 'noyes']
    token = next((t for t in tokens if f'_{t}_' in filename), None)

    import re
    if token is None and 'v21_zero_shot_cot' in filename:
        if re.search(r'v21_zero_shot_cot_n\d+_exp\d+_', filename):
            token = 'kz'
    if token == 'noyes':
        token = 'yesno'

    return model, token, nbhood, fmt


def collect_token_bias_data(sources):
    results = []
    for model_name, base_path in sources.items():
        for n_dir in ('n7', 'n9', 'n11'):
            csv_dir = os.path.join(base_path, n_dir, 'csv', 'basic')
            if not os.path.exists(csv_dir):
                continue
            for filepath in glob.glob(os.path.join(csv_dir, '*.csv')):
                try:
                    df = pd.read_csv(filepath,
                                     dtype={'configuracao_letras': str, 'distribuicao_vizinhos': str},
                                     keep_default_na=False)
                    delta, c0, c1, total = calculate_token_bias(df)
                    if delta is None:
                        continue
                    model, token, nbhood, fmt = extract_token_bias_info(filepath)
                    if token is None or nbhood is None or model == 'Unknown':
                        continue
                    results.append({
                        'model': model, 'token': token, 'neighborhood': nbhood,
                        'format': fmt, 'delta': delta,
                    })
                except Exception:
                    pass
    return results


def plot_all_tokens(results, output_path):
    token_labels = {
        'αβ': 'αβ', '△○': '△○', '⊕⊖': '⊕⊖', 'pq': 'pq', 'łþ': 'łþ',
        '01': '01', 'ab': 'ab', 'kz': 'kz', 'yesno': 'no/yes',
    }

    models = sorted(set(r['model'] for r in results))
    n_models = max(1, len(models))
    ncols = min(3, n_models)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    fig.suptitle('Token Bias (\u0394) vs. Number of Neighbors \u2014 All Tokens\n(n=7, 9, 11)',
                 fontsize=18, fontweight='bold', y=0.995)
    axes = np.array(axes).reshape(-1)
    tokens_unique = sorted(set(r['token'] for r in results))
    colors = {t: plt.cm.tab10(i) for i, t in enumerate(tokens_unique)}
    fmt_markers = {'with_cot': 'o', 'only_token': 's'}
    fmt_lines = {'with_cot': '-', 'only_token': '--'}

    for idx, model in enumerate(models):
        ax = axes[idx]
        ax.set_title(model, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Number of Neighbors', fontsize=12, fontweight='bold')
        ax.set_ylabel('Token Bias (\u0394)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='black', linewidth=1.5, alpha=0.6, label='\u0394 = 0 (no bias)')
        ax.set_xticks([7, 9, 11])
        ax.set_ylim(-1.1, 1.1)

        model_data = [r for r in results if r['model'] == model]
        for token in tokens_unique:
            for fmt in ('with_cot', 'only_token'):
                pts = sorted([r for r in model_data if r['token'] == token and r['format'] == fmt],
                             key=lambda x: x['neighborhood'])
                if not pts:
                    continue
                ns = [p['neighborhood'] for p in pts]
                ds = [p['delta'] for p in pts]
                label = f"{token_labels.get(token, token)} ({fmt})"
                ax.plot(ns, ds, marker=fmt_markers.get(fmt, 'o'),
                        color=colors.get(token, '#7f8c8d'),
                        linestyle=fmt_lines.get(fmt, '-'),
                        linewidth=2.5, markersize=9, label=label, alpha=0.8)

        ax.text(7, 1.05, 'Token_1 bias', fontsize=9, ha='left', style='italic', alpha=0.7)
        ax.text(7, -1.05, 'Token_0 bias', fontsize=9, ha='left', style='italic', alpha=0.7)
        ax.legend(loc='best', fontsize=7, framealpha=0.95, edgecolor='gray', ncol=2)

    for ax in axes[n_models:]:
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {os.path.relpath(output_path, BASE)}")


def plot_focused_tokens(results, token1, token2, output_path):
    token_labels = {
        'αβ': 'αβ (α=0, β=1)', '△○': '△○ (△=0, ○=1)', '⊕⊖': '⊕⊖ (⊕=0, ⊖=1)',
        'pq': 'pq (p=0, q=1)', 'łþ': 'łþ (ł=0, þ=1)', '01': '01 (0=0, 1=1)',
        'ab': 'ab (a=0, b=1)', 'kz': 'kz (k=0, z=1)', 'yesno': 'no/yes (no=0, yes=1)',
    }
    label1 = token_labels.get(token1, token1)
    label2 = token_labels.get(token2, token2)

    models = sorted(set(r['model'] for r in results))
    n_models = max(1, len(models))
    ncols = min(3, n_models)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    fig.suptitle(
        f'Token Bias (\u0394) vs. Number of Neighbors\n(Tokens: {label1} and {label2} | n=7, 9, 11)',
        fontsize=18, fontweight='bold', y=0.995)
    axes = np.array(axes).reshape(-1)
    colors = {token1: '#e74c3c', token2: '#3498db'}
    fmt_markers = {'with_cot': 'o', 'only_token': 's'}
    fmt_lines = {'with_cot': '-', 'only_token': '--'}

    for idx, model in enumerate(models):
        ax = axes[idx]
        ax.set_title(model, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Number of Neighbors', fontsize=12, fontweight='bold')
        ax.set_ylabel('Token Bias (\u0394)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='black', linewidth=1.5, alpha=0.6, label='\u0394 = 0 (no bias)')
        ax.set_xticks([7, 9, 11])
        ax.set_ylim(-1.1, 1.1)

        model_data = [r for r in results if r['model'] == model]
        for token in (token1, token2):
            for fmt in ('with_cot', 'only_token'):
                pts = sorted([r for r in model_data if r['token'] == token and r['format'] == fmt],
                             key=lambda x: x['neighborhood'])
                if not pts:
                    continue
                ns = [p['neighborhood'] for p in pts]
                ds = [p['delta'] for p in pts]
                ax.plot(ns, ds, marker=fmt_markers.get(fmt, 'o'),
                        color=colors.get(token, '#7f8c8d'),
                        linestyle=fmt_lines.get(fmt, '-'),
                        linewidth=3, markersize=11,
                        label=f"{token} ({fmt})", alpha=0.85)

        ax.text(7, 1.05, 'Token_1 bias (second)', fontsize=9, ha='left', style='italic', alpha=0.7)
        ax.text(7, -1.05, 'Token_0 bias (first)', fontsize=9, ha='left', style='italic', alpha=0.7)
        ax.legend(loc='best', fontsize=9, framealpha=0.95, edgecolor='gray')

    for ax in axes[n_models:]:
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {os.path.relpath(output_path, BASE)}")


def generate_token_bias_plots(out_dir):
    print("\n[4] Generating token bias plots...")

    sources = {
        TARGET_MODEL_ONLY: GEMMA27B_EXTRACT_DIR,
    }

    for model, path in sources.items():
        if not os.path.exists(path):
            print(f"  [warn] Missing data for {model}: {path}")

    results = collect_token_bias_data(sources)
    print(f"  Collected {len(results)} data points.")
    if not results:
        print("  [skip] No token-bias data available.")
        return

    per_model = {}
    for r in results:
        per_model.setdefault(r['model'], 0)
        per_model[r['model']] += 1
    for m, c in sorted(per_model.items()):
        print(f"    {m}: {c}")

    # All-tokens plot
    plot_all_tokens(results, os.path.join(out_dir, 'token_bias_all_tokens.png'))

    # Focused plot: find 2 most extreme tokens by mean delta
    token_deltas = {}
    for r in results:
        token_deltas.setdefault(r['token'], []).append(r['delta'])
    means = {t: np.mean(v) for t, v in token_deltas.items()}
    if not means:
        print("  [skip] No token means available for focused plot.")
        return
    token_pos = max(means, key=means.get)
    token_neg = min(means, key=means.get)
    print(f"  Most biased tokens: positive={token_pos} ({means[token_pos]:+.4f}), "
          f"negative={token_neg} ({means[token_neg]:+.4f})")
    plot_focused_tokens(results, token_pos, token_neg,
                        os.path.join(out_dir, f'token_bias_focused_{token_pos}_{token_neg}.png'))


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("GENERATING FINAL RESULTS (English) -> resultados_finais_en/")
    print("=" * 70)

    # Load ICI data (shared by tables 1 and 2)
    print(f"\n[0] Loading ICI data ({TARGET_MODEL_ONLY}, A100 latest CSVs)...")
    all_data = []
    model_bases = [
        (TARGET_MODEL_ONLY, GEMMA27B_EXTRACT_DIR),
    ]
    for name, base_dir in model_bases:
        if not os.path.exists(base_dir):
            print(f"  [warn] Missing base directory for {name}: {base_dir}")
            continue
        for n in ('n7', 'n9', 'n11'):
            d = collect_ici_data(os.path.join(base_dir, n, 'csv', 'basic'), name)
            all_data.extend(d)
            print(f"  {name} ({n}): {len(d)} experiments")
    print(f"  Total: {len(all_data)} experiments")

    generate_basic_ranking_tables(all_data, OUTPUT_DIR)
    generate_extended_ranking_tables(all_data, OUTPUT_DIR)
    generate_gemma27b_aggregated_results(OUTPUT_DIR)
    generate_token_bias_plots(OUTPUT_DIR)

    # Summary
    images = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    print("\n" + "=" * 70)
    print(f"DONE. {len(images)} images saved to: {OUTPUT_DIR}")
    print("=" * 70)
    for img in sorted(images):
        print(f"  {img}")


if __name__ == '__main__':
    main()
