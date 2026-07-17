#!/usr/bin/env python3
"""
Script para sincronizar mudanças do CSV para o SQLite
Uso: python csv_to_sqlite.py

Este script lê o experimentos_master.csv e atualiza o banco SQLite
com as modificações feitas manualmente no CSV.
"""

import os
import pandas as pd
import db_sqlite
from typing import Dict, Any

def convert_csv_value_to_db(value: Any, column: str) -> Any:
    """Converte valores do CSV para o tipo correto do banco"""
    if pd.isna(value) or value == "" or value is None:
        return None
    
    # Colunas numéricas inteiras
    if column in ['id_experimento', 'num_vizinhos']:
        try:
            return int(float(str(value)))
        except (ValueError, TypeError):
            return None
    
    # Colunas numéricas decimais
    if column in ['num_iteracoes']:
        try:
            return float(str(value))
        except (ValueError, TypeError):
            return None
    
    # Colunas de texto
    return str(value).strip() if str(value).strip() else None

def csv_to_sqlite(csv_path: str = None):
    """Sincroniza dados do CSV para o SQLite"""
    csv_path = csv_path or os.getenv('EXPERIMENTOS_CSV_PATH', 'experimentos/experimentos_master.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ Arquivo CSV não encontrado: {csv_path}")
        return False
    
    try:
        # Lê CSV
        print(f"📖 Lendo CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Verifica colunas obrigatórias
        required_cols = ['id_experimento', 'conjunto_experimento', 'status']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Colunas obrigatórias ausentes no CSV: {missing_cols}")
            return False
        
        # Remove linhas com id_experimento inválido
        df = df.dropna(subset=['id_experimento'])
        df = df[df['id_experimento'] != '']
        
        print(f"📊 Processando {len(df)} experimentos do CSV")
        
        # Obtém estado atual do SQLite
        current_status = db_sqlite.get_status()
        sqlite_experiments = {exp['id_experimento']: exp for exp in current_status['experiments']}
        
        updates_count = 0
        new_count = 0
        errors_count = 0
        
        # Processa cada linha do CSV
        for _, row in df.iterrows():
            try:
                exp_id = int(float(row['id_experimento']))
                
                # Converte valores do CSV para tipos corretos
                update_data = {}
                for col in df.columns:
                    if col != 'id_experimento':  # ID não deve ser atualizado
                        csv_value = convert_csv_value_to_db(row[col], col)
                        
                        # Só inclui na atualização se o valor mudou
                        if exp_id in sqlite_experiments:
                            sqlite_value = sqlite_experiments[exp_id].get(col)
                            if csv_value != sqlite_value:
                                update_data[col] = csv_value
                        else:
                            # Experimento novo - inclui todos os valores não-nulos
                            if csv_value is not None:
                                update_data[col] = csv_value
                
                # Aplica mudanças se houver
                if update_data:
                    if exp_id in sqlite_experiments:
                        # Atualiza experimento existente usando SQL direto para maior flexibilidade
                        try:
                            with db_sqlite.get_conn() as conn:
                                sets = [f"{k}=?" for k in update_data.keys()]
                                vals = list(update_data.values()) + [exp_id]
                                sql = f"UPDATE {db_sqlite.TABLE} SET {', '.join(sets)} WHERE id_experimento=?"
                                
                                conn.execute("BEGIN IMMEDIATE;")
                                conn.execute(sql, vals)
                                conn.execute("COMMIT;")
                                
                                updates_count += 1
                                changes_list = [f"{k}={v}" for k, v in update_data.items()]
                                print(f"  ✅ Experimento {exp_id} atualizado: {', '.join(changes_list)}")
                        except Exception as e:
                            errors_count += 1
                            print(f"  ❌ Falha ao atualizar experimento {exp_id}: {e}")
                    else:
                        # Cria novo experimento
                        # Para novos experimentos, precisamos garantir campos obrigatórios
                        if 'conjunto_experimento' not in update_data or 'status' not in update_data:
                            print(f"  ⚠️  Experimento {exp_id} ignorado - campos obrigatórios ausentes")
                            continue
                        
                        # Adiciona valores padrão para campos obrigatórios se ausentes
                        
                        # Simula criação via inserção SQL direta
                        with db_sqlite.get_conn() as conn:
                            cols = list(update_data.keys()) + ['id_experimento']
                            vals = list(update_data.values()) + [exp_id]
                            placeholders = ','.join(['?' for _ in cols])
                            
                            conn.execute("BEGIN IMMEDIATE;")
                            conn.execute(f"INSERT OR REPLACE INTO {db_sqlite.TABLE} ({','.join(cols)}) VALUES ({placeholders})", vals)
                            conn.execute("COMMIT;")
                            
                            new_count += 1
                            print(f"  ✅ Experimento {exp_id} criado")
                            
            except Exception as e:
                errors_count += 1
                print(f"  ❌ Erro processando linha {row.name}: {e}")
        
        # Resumo final
        print(f"\n🎯 SINCRONIZAÇÃO CONCLUÍDA:")
        print(f"   • {updates_count} experimentos atualizados")
        print(f"   • {new_count} experimentos criados")
        if errors_count > 0:
            print(f"   • {errors_count} erros encontrados")
        
        # REMOVIDO: Regeneração automática do CSV que causava duplicatas
        # O CSV deve ser mantido como está - SQLite é a fonte da verdade durante execução
        print(f"\n✅ Sincronização concluída - CSV mantido como fonte inicial")
        
        return errors_count == 0
        
    except Exception as e:
        print(f"❌ Erro geral na sincronização: {e}")
        return False

if __name__ == "__main__":
    print("=== 📥 SINCRONIZAÇÃO CSV → SQLite ===\n")
    success = csv_to_sqlite()
    if success:
        print("\n✅ Sincronização concluída com sucesso!")
    else:
        print("\n❌ Sincronização concluída com erros!")
