# merge_precedences.py — mescla arquivo base de blocos com arquivo de precedências (prec1..precN)
import pandas as pd
import sys
import os

def read_with_sep(path):
    # Detecta separador ; ou , no primeiro trecho do arquivo
    with open(path, 'r', encoding='utf-8') as f:
        head = f.readline()
    sep = ';' if ';' in head else ','
    return pd.read_csv(path, sep=sep)

def merge_precedences(modelo_blocos_path, precedencias_path, output_path):
    #detecta delimitador
    df_blocos = read_with_sep(modelo_blocos_path)
    df_precedencias = read_with_sep(precedencias_path)

    # Normalizar nomes de colunas (sem espaços, minúsculas)
    df_blocos.columns = df_blocos.columns.str.strip().str.lower()
    df_precedencias.columns = df_precedencias.columns.str.strip().str.lower()

    # Garantir coluna id em ambos (tentar inferir)
    if 'id' not in df_blocos.columns:
        # Se a primeira coluna contém "id;..." provavelmente foi lido errado — tentar reparo simples
        # Mas aqui assumimos que id existe. Caso contrário, criamos índice sequencial.
        df_blocos.insert(0, 'id', range(len(df_blocos)))
    if 'id' not in df_precedencias.columns:
        # procurar coluna que contenha 'id' no nome
        possible_id = [c for c in df_precedencias.columns if 'id' in c]
        if possible_id:
            df_precedencias.rename(columns={possible_id[0]: 'id'}, inplace=True)
        else:
            # se não existir, assumir alinhamento por ordem e criar id sequencial
            df_precedencias.insert(0, 'id', df_blocos['id'].values if 'id' in df_blocos.columns else range(len(df_blocos)))

    # Detectar colunas de precedência (prec1, prec2, ...)
    prec_cols = [c for c in df_precedencias.columns if c.startswith('prec')]
    if not prec_cols:
        # tentar detectar colunas que contenham 'prec' em qualquer posição (fallback)
        prec_cols = [c for c in df_precedencias.columns if 'prec' in c]
    if not prec_cols:
        # Se não há colunas prec*, talvez o arquivo já tenha 'precedentes' (string)
        if 'precedentes' not in df_precedencias.columns:
            # criar precedentes vazios
            df_precedencias['precedentes'] = ""
            prec_cols = []
        else:
            prec_cols = []

    # Montar coluna 'precedentes' no formato string de lista Python: "[0,1,2]"
    def make_preced_list(row):
        if 'precedentes' in row and pd.notna(row['precedentes']) and str(row['precedentes']).strip() != "":
            # já existe algo — tentar converter "0;1" ou "0,1" em lista string "[0,1]"
            s = str(row['precedentes']).strip()
            s = s.replace(';', ',')
            parts = [p.strip() for p in s.split(',') if p.strip() != ""]
            nums = [int(p) for p in parts if p.replace('-', '').isdigit()]
            return "[" + ",".join(map(str, nums)) + "]"
        elif prec_cols:
            vals = []
            for c in prec_cols:
                v = row.get(c, None)
                if pd.isna(v):
                    continue
                try:
                    iv = int(v)
                except Exception:
                    continue
                if iv != -1:
                    vals.append(iv)
            return "[" + ",".join(map(str, vals)) + "]" if vals else "[]"
        else:
            return "[]"

    # Aplicar criação
    df_precedencias['precedentes'] = df_precedencias.apply(make_preced_list, axis=1)

    # Merge pelo id (left join)
    merged = pd.merge(df_blocos, df_precedencias[['id', 'precedentes']], on='id', how='left')

    # Normalizar nomes de colunas do bloco para os esperados pelo GA
    # Possíveis mapeamentos comuns
    rename_map = {}
    cols_low = [c.lower() for c in merged.columns]
    # mapear blockvalue -> val_ore, destination -> dest
    if 'blockvalue' in merged.columns:
        rename_map['blockvalue'] = 'val_ore'
    if 'destination' in merged.columns:
        rename_map['destination'] = 'dest'
    # caso tonn exista com outro nome, manter se já tiver 'tonn' ok
    merged.rename(columns=rename_map, inplace=True)

    # Garantir colunas esperadas: id, x,y,z,tonn,val_ore,dest,precedentes
    # Se alguma não existir, criar com fallback razoável
    if 'x' not in merged.columns: merged['x'] = 0
    if 'y' not in merged.columns: merged['y'] = 0
    if 'z' not in merged.columns: merged['z'] = 0
    if 'tonn' not in merged.columns:
        # tentar inferir de colunas possíveis
        for cand in ['ton', 'tonnage', 'mass', 'weight']:
            if cand in merged.columns:
                merged['tonn'] = merged[cand]
                break
        else:
            merged['tonn'] = 100  # fallback
    if 'val_ore' not in merged.columns:
        # tentar usar blockvalue ou criar estimativa
        if 'blockvalue' in merged.columns:
            merged['val_ore'] = merged['blockvalue']
        else:
            merged['val_ore'] = (merged['z'].max() - merged['z']) * 10.0
    if 'dest' not in merged.columns:
        # destination previously could be 1 ore, 2 waste — fallback: 1 if val_ore>0 else 0
        merged['dest'] = merged['val_ore'].apply(lambda v: 1 if v > 0 else 0)

    # Selecionar e reordenar colunas no formato esperado
    final_cols = ['id', 'x', 'y', 'z', 'tonn', 'val_ore', 'dest', 'precedentes']
    merged = merged[[c for c in final_cols if c in merged.columns]]

    # Salvar CSV com delimitador padrão (vírgula) — 'precedentes' é string tipo "[0,1]"
    merged.to_csv(output_path, index=False)
    print(f"\n Arquivo salvo com sucesso: {output_path}")
    print("Amostra (primeiras linhas):")
    print(merged.head().to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python merge_precedences.py <Modelo de Blocos.csv> <Modelo_com_Precedencias.csv> <modelo_final.csv>")
        sys.exit(1)
    modelo_blocos_path = sys.argv[1]
    precedencias_path = sys.argv[2]
    output_path = sys.argv[3]
    if not os.path.exists(modelo_blocos_path):
        print(f"Arquivo não encontrado: {modelo_blocos_path}")
        sys.exit(1)
    if not os.path.exists(precedencias_path):
        print(f"Arquivo não encontrado: {precedencias_path}")
        sys.exit(1)
    merge_precedences(modelo_blocos_path, precedencias_path, output_path)
