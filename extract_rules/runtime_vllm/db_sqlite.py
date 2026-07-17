# db_sqlite.py
import os, csv, sqlite3, datetime
from contextlib import contextmanager

# Permitir DB e CSV customizados via variáveis de ambiente
DB_PATH = os.getenv('EXPERIMENTOS_DB_PATH', os.path.join("experimentos", "experimentos.db"))
CSV_PATH = os.getenv('EXPERIMENTOS_CSV_PATH', 'experimentos/experimentos_master.csv')
TABLE = "experiments"

@contextmanager
def get_conn(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row  # CORREÇÃO: Permite acesso por nome de coluna
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
    finally:
        conn.close()

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
  id_experimento INTEGER PRIMARY KEY,
  conjunto_experimento TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pendente','executando','concluido','erro')),
  process_id TEXT,
  hora_inicio TEXT,
  hora_fim TEXT,
  modelo TEXT,
  modelo_executado TEXT,
  num_vizinhos INTEGER,
  num_iteracoes REAL,
  temperaturas TEXT,
  variante_prompt TEXT,
  notas TEXT,
  caminho_csv_saida TEXT,
  caminho_log_saida TEXT,
  caminho_log_prompt_saida TEXT
);
CREATE INDEX IF NOT EXISTS idx_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_modelo_executado ON experiments(modelo_executado);
"""

def init_db(from_csv=None):
    with get_conn() as conn:
        for stmt in SCHEMA.strip().split(";\n"):
            if stmt.strip():
                conn.execute(stmt)
        if from_csv and os.path.exists(from_csv):
            # Importa CSV inicial de forma simples
            with open(from_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Converte campos vazios para None
                    clean_row = {}
                    for k, v in row.items():
                        if v == '' or v is None:
                            clean_row[k] = None
                        else:
                            clean_row[k] = v
                    
                    # Insert sem ON CONFLICT (simples)
                    # Mapear campos do CSV para colunas do banco
                    variante = clean_row.get('variante_prompt') or clean_row.get('prompt_variant')
                    iteracoes = clean_row.get('num_iteracoes') or clean_row.get('iteracoes')
                    
                    try:
                        conn.execute(f"""
                            INSERT OR IGNORE INTO {TABLE} 
                            (id_experimento, conjunto_experimento, status, process_id,
                             hora_inicio, hora_fim, modelo, modelo_executado, num_vizinhos, num_iteracoes,
                             temperaturas, variante_prompt, notas, caminho_csv_saida, caminho_log_saida, caminho_log_prompt_saida)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            clean_row.get('id_experimento'),
                            clean_row.get('conjunto_experimento'),
                            clean_row.get('status'),
                            clean_row.get('process_id'),
                            clean_row.get('hora_inicio'),
                            clean_row.get('hora_fim'),
                            clean_row.get('modelo'),
                            clean_row.get('modelo_executado'),
                            clean_row.get('num_vizinhos'),
                            iteracoes,
                            clean_row.get('temperaturas'),
                            variante,
                            clean_row.get('notas'),
                            clean_row.get('caminho_csv_saida'),
                            clean_row.get('caminho_log_saida'),
                            clean_row.get('caminho_log_prompt_saida')
                        ))
                    except Exception as e:
                        print(f"Erro ao importar linha {clean_row.get('id_experimento')}: {e}")
            print(f"CSV importado: {from_csv}")

def _models_in_use(conn):
    rows = conn.execute(f"SELECT DISTINCT modelo_executado FROM {TABLE} WHERE status='executando' AND modelo_executado IS NOT NULL").fetchall()
    return {r[0] for r in rows}

def _first_available_model(available_models, models_in_use):
    for m in available_models:
        if m not in models_in_use:
            return m
    return None

def reserve_next_experiment(preferred_model, start_id, end_id, available_models, process_id):
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE;")  # trava para reserva atômica
        base_where = ["status='pendente'"]
        base_params = []

        # Preferimos reservar por POOL (permite shuffle entre deploys), mas aceitamos CSV/DB
        # que já venha com o modelo exato (ex.: google/gemma-3-12b:4) para mapeamento fixo.
        preferred_pool = None
        if preferred_model:
            pm = preferred_model.lower()
            if 'gemma' in pm and '4b' in pm:
                preferred_pool = 'gemma4b-pool'
            elif 'gemma' in pm and '12b' in pm:
                preferred_pool = 'gemma12b-pool'
            elif 'llama' in pm and '70b' in pm:
                preferred_pool = 'llama70b-pool'
            elif 'llama' in pm and '8b' in pm:
                preferred_pool = 'llama8b-pool'
            elif 'qwen' in pm and '4b' in pm:
                preferred_pool = 'qwen4b-pool'
            elif 'qwen' in pm and ('32b' in pm or 'qwq' in pm):
                preferred_pool = 'qwen32b-pool'

            print(f"🔍 [DB_POOL_DEBUG] Modelo: {preferred_model} → Pool: {preferred_pool}")

        if start_id is not None:
            base_where.append("id_experimento>=?")
            base_params.append(start_id)
        if end_id is not None:
            base_where.append("id_experimento<=?")
            base_params.append(end_id)

        def _select_one(where_extra, params_extra):
            where = base_where + where_extra
            params = base_params + params_extra
            sql = f"SELECT * FROM {TABLE} WHERE {' AND '.join(where)} ORDER BY id_experimento ASC LIMIT 1"
            print(f"🔍 [DB_SQL_DEBUG] Query: {sql}")
            print(f"🔍 [DB_SQL_DEBUG] Params: {params}")
            return conn.execute(sql, params).fetchone(), sql, params

        # 1) Tenta pelo pool (quando reconhecido) para shuffling por deploy.
        row = None
        last_sql = None
        last_params = None
        if preferred_model and preferred_pool:
            row, last_sql, last_params = _select_one(["modelo_executado=?"], [preferred_pool])

        # 2) Fallback: tenta pelo modelo exato (CSV/DB com mapping fixo por deploy).
        if row is None and preferred_model:
            row, last_sql, last_params = _select_one(["modelo_executado=?"], [preferred_model])

        # 3) Sem modelo preferido: pega o primeiro pendente no range.
        if row is None and not preferred_model:
            row, last_sql, last_params = _select_one([], [])
        
        if not row:
            # Debug: ver quantos experimentos existem no filtro mais relevante (pool/modelo),
            # para facilitar diagnóstico sem confundir com start_id/end_id.
            count_key = None
            if preferred_model and preferred_pool:
                count_key = preferred_pool
            elif preferred_model:
                count_key = preferred_model

            if count_key is not None:
                count_sql = f"SELECT COUNT(*) FROM {TABLE} WHERE status='pendente' AND modelo_executado=?"
                count = conn.execute(count_sql, [count_key]).fetchone()[0]
                print(f"🔍 [DB_SQL_DEBUG] Nenhum experimento encontrado. Total pendente em '{count_key}': {count}")
            conn.execute("COMMIT;")
            return None

        # CORREÇÃO: Com row_factory, row já é acessível por nome
        row_dict = dict(row)

        # Decide modelo - IMPORTANTE: usar o modelo REAL do deploy, não o pool
        if preferred_model:
            model_to_use = preferred_model  # Usa o modelo real (ex: google/gemma-3-4b:2)
        else:
            model_to_use = _first_available_model(available_models, _models_in_use(conn))
            if not model_to_use:
                conn.execute("COMMIT;")
                return None

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        exp_id = row_dict["id_experimento"]
        current_pool = row_dict["modelo_executado"]  # O pool atual (ex: gemma4b-pool)

        print(f"🔍 [DB_RESERVE_DEBUG] Reservando exp {exp_id}: pool='{current_pool}' → modelo_real='{model_to_use}'")

        # Efetiva reserva garantindo status pendente E pool correto
        # Atualiza o modelo_executado para o modelo REAL do deploy
        updated = conn.execute(
            f"""UPDATE {TABLE}
                SET status='executando',
                    process_id=?,
                    hora_inicio=?,
                    modelo_executado=?
              WHERE id_experimento=? AND status='pendente' AND modelo_executado=?""",
            (process_id, now, model_to_use, exp_id, current_pool),  # Verifica o pool, atualiza para modelo real
        )
        if updated.rowcount != 1:
            conn.execute("ROLLBACK;")
            return None

        conn.execute("COMMIT;")
        
        # Sanidade: Verifica se a reserva foi bem-sucedida
        verify = conn.execute(
            f"SELECT status, modelo_executado, process_id FROM {TABLE} WHERE id_experimento=?",
            [exp_id]
        ).fetchone()
        
        if not verify or verify['status'] != 'executando' or verify['modelo_executado'] != model_to_use:
            print(f"❌ [DB_ERROR] Reserva inconsistente para exp {exp_id}! Esperado: {model_to_use}, Real: {verify}")
            return None
        
        row_dict["status"] = "executando"
        row_dict["process_id"] = process_id
        row_dict["hora_inicio"] = now
        row_dict["modelo_executado"] = model_to_use
        return row_dict

def update_experiment_status(experiment_id, status, **fields):
    """Atualiza status do experimento com logging extensivo para debug do bug"""
    import traceback
    import time
    
    start_time = time.time()
    print(f"🔍 [DB_DEBUG] update_experiment_status: ID={experiment_id}, status={status}")
    if fields:
        print(f"🔍 [DB_DEBUG] Fields: {list(fields.keys())}")
    
    try:
        with get_conn() as conn:
            conn_time = time.time() - start_time
            print(f"🔍 [DB_DEBUG] SQLite connection obtained in {conn_time:.3f}s")
            
            # Verificar se experimento existe ANTES do update
            check_sql = f"SELECT id_experimento, status FROM {TABLE} WHERE id_experimento=?"
            existing = conn.execute(check_sql, [experiment_id]).fetchone()
            
            if not existing:
                print(f"❌ [DB_DEBUG] ERRO: Experimento {experiment_id} NÃO EXISTE no banco!")
                return False
            
            current_status = existing['status']
            print(f"🔍 [DB_DEBUG] Status atual no DB: {current_status} → {status}")
            
            # Begin transaction com logging
            conn.execute("BEGIN IMMEDIATE;")
            print(f"🔍 [DB_DEBUG] BEGIN IMMEDIATE executado")
            
            sets, vals = ["status=?"], [status]
            if status in ("concluido", "erro"):
                sets.append("hora_fim=?")
                vals.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                sets.append("process_id=NULL")
                print(f"🔍 [DB_DEBUG] Adicionado hora_fim e process_id=NULL")
                
            for k, v in fields.items():
                if k in {"caminho_csv_saida","caminho_log_saida","caminho_log_prompt_saida","notas","modelo_executado"}:
                    sets.append(f"{k}=?")
                    vals.append(v)
                    
            vals.append(experiment_id)
            sql = f"UPDATE {TABLE} SET {', '.join(sets)} WHERE id_experimento=?"
            
            print(f"🔍 [DB_DEBUG] SQL: {sql}")
            print(f"🔍 [DB_DEBUG] Valores: {len(vals)} parâmetros")
            
            # Execute update
            cur = conn.execute(sql, vals)
            affected_rows = cur.rowcount
            print(f"🔍 [DB_DEBUG] Rows affected: {affected_rows}")
            
            # Commit with verification
            conn.execute("COMMIT;")
            print(f"🔍 [DB_DEBUG] COMMIT executado")
            
            # Verificar se mudança foi aplicada
            verify = conn.execute(check_sql, [experiment_id]).fetchone()
            final_status = verify['status']
            print(f"🔍 [DB_DEBUG] Status APÓS update: {final_status}")
            
            success = affected_rows == 1
            total_time = time.time() - start_time
            
            if success and final_status == status:
                print(f"✅ [DB_DEBUG] Update SUCESSO em {total_time:.3f}s: {current_status} → {final_status}")
            else:
                print(f"❌ [DB_DEBUG] Update FALHOU em {total_time:.3f}s: affected_rows={affected_rows}, final_status={final_status}")
            
            return success
            
    except Exception as e:
        print(f"❌ [DB_DEBUG] EXCEÇÃO CAPTURADA: {type(e).__name__}: {e}")
        print(f"❌ [DB_DEBUG] Traceback completo:")
        traceback.print_exc()
        
        # Log crítico para arquivo também
        try:
            with open(f"debug_db_error_{experiment_id}_{int(time.time())}.log", 'w') as f:
                f.write(f"DB ERROR - Experiment {experiment_id}\n")
                f.write(f"Status: {status}\n")
                f.write(f"Fields: {fields}\n")
                f.write(f"Exception: {type(e).__name__}: {e}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n")
        except:
            pass  # Se não conseguir escrever log, não falhe tudo
            
        return False

def get_status():
    with get_conn() as conn:
        rows = conn.execute(f"SELECT * FROM {TABLE}").fetchall()
        # CORREÇÃO: Com row_factory, rows já são acessíveis por nome
        data = [dict(row) for row in rows]
        summary = {}
        for r in data:
            status = r.get("status", "unknown")
            if status:  # Skip se status for None/vazio
                summary[status] = summary.get(status, 0) + 1
        return {"total": len(data), "status_counts": summary, "experiments": data}
