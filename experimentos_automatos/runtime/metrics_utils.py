#!/usr/bin/env python3
"""
Utilitários para cálculo de métricas: ICI e Token Bias
Sem dependências pesadas (usa csv module ao invés de pandas)
"""

import csv


def inverter_config(config_str):
    """
    Calcula C̄ (complemento bit-a-bit de C)
    
    Args:
        config_str: String de configuração (ex: "0110", "abba", "yesnoyes")
    
    Returns:
        String invertida (ex: "1001", "baab", "noyesno")
    """
    config_str = str(config_str)
    
    # Caso especial: yesno
    if 'yes' in config_str or 'no' in config_str:
        config_invertida = config_str.replace('yes', '___TEMP___').replace('no', 'yes').replace('___TEMP___', 'no')
        return config_invertida
    
    # Mapear tokens para seus inversos
    mapeamento = {
        'α': 'β', 'β': 'α',
        '△': '○', '○': '△',
        '⊕': '⊖', '⊖': '⊕',
        'p': 'q', 'q': 'p',
        'ł': 'þ', 'þ': 'ł',
        '0': '1', '1': '0',
        'a': 'b', 'b': 'a',
        'k': 'z', 'z': 'k'
    }
    return ''.join(mapeamento.get(c, c) for c in config_str)


def calcular_ici_token_bias_from_csv(csv_path):
    """
    Calcula ICI e Token Bias de um CSV usando pares {C, C̄}
    
    Args:
        csv_path: Caminho para o CSV com colunas 'configuracao_letras' ou 'distribuicao_vizinhos' e 'escolha'
    
    Returns:
        tuple: (ici: float, token_bias: float)
               - ICI = consistent_pairs / total_pairs
               - Token Bias (Δ) = (count_1 - count_0) / (2 * total_pairs)
    
    Retorna (0.0, 0.0) se não conseguir calcular.
    """
    try:
        # Ler CSV e construir lookup: config → escolha
        lookup = {}
        col_config = None
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Detectar coluna de configuração
            fieldnames = reader.fieldnames
            if 'configuracao_letras' in fieldnames:
                col_config = 'configuracao_letras'
            elif 'distribuicao_vizinhos' in fieldnames:
                col_config = 'distribuicao_vizinhos'
            else:
                return 0.0, 0.0
            
            # Construir lookup
            for row in reader:
                config = str(row[col_config])
                escolha = int(row['escolha'])
                lookup[config] = escolha
        
        # Calcular ICI e Token Bias usando pares {C, C̄}
        consistent_pairs = 0
        total_pairs = 0
        count_escolha_0 = 0
        count_escolha_1 = 0
        configs_processadas = set()
        
        for config in lookup.keys():
            if config in configs_processadas:
                continue
            
            config_inv = inverter_config(config)
            
            if config_inv in lookup:
                escolha = lookup[config]
                escolha_inv = lookup[config_inv]
                
                # ICI: conta se escolha_inv == (1 - escolha)
                if escolha_inv == (1 - escolha):
                    consistent_pairs += 1
                
                # Token Bias: conta escolhas
                count_escolha_0 += (1 if escolha == 0 else 0)
                count_escolha_1 += (1 if escolha == 1 else 0)
                count_escolha_0 += (1 if escolha_inv == 0 else 0)
                count_escolha_1 += (1 if escolha_inv == 1 else 0)
                
                total_pairs += 1
                configs_processadas.add(config)
                configs_processadas.add(config_inv)
        
        # Calcular métricas finais
        if total_pairs > 0:
            ici = consistent_pairs / total_pairs
            token_bias = (count_escolha_1 - count_escolha_0) / (2 * total_pairs)
            return ici, token_bias
        else:
            return 0.0, 0.0
            
    except Exception as e:
        print(f"⚠️  Erro ao calcular métricas de {csv_path}: {e}")
        return 0.0, 0.0


if __name__ == '__main__':
    # Teste rápido
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        ici, token_bias = calcular_ici_token_bias_from_csv(csv_path)
        print(f"📊 {csv_path}")
        print(f"   ICI: {ici:.4f}")
        print(f"   Token Bias (Δ): {token_bias:+.4f}")
    else:
        # Teste de inversão
        print("Teste de inversão:")
        tests = ['0101', 'abba', 'yesnoyes', 'αβα', '△○△']
        for t in tests:
            inv = inverter_config(t)
            print(f"  {t} → {inv}")







