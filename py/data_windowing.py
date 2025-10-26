import os
import pandas as pd
import numpy as np
import math
from collections import Counter
from datetime import datetime
from typing import List, Dict, Union

def clean_column_names(columns: pd.Index) -> List[str]:
    return [col.replace(' ', '_') for col in columns]

def shannon_entropy(data: pd.Series) -> float:
    if data.empty:
        return 0.0
    
    data = data.astype(str).dropna()
    counts = data.value_counts(normalize=True)
    entropy = -np.sum(counts * np.log2(counts))
    return entropy

def aggregate_window_by_time(df_window: pd.DataFrame) -> pd.Series:
    timestamp_col_clean = TIMESTAMP_COL.replace(' ', '_')
    attack_label_col_clean = ATTACK_LABEL_COL.replace(' ', '_')
    diversity_cols_clean = [col.replace(' ', '_') for col in DIVERSITY_COLS]
    
    # Identifica colunas numéricas 
    numeric_cols = [col for col in df_window.columns 
                    if col not in [timestamp_col_clean, attack_label_col_clean] + diversity_cols_clean]

    # Calcular a Média para Colunas Numéricas
    numeric_data = df_window[numeric_cols].apply(pd.to_numeric, errors='coerce')
    aggregated_row = numeric_data.mean(axis=0)

    # Agregação Temporal
    try:
        # Garante que o Timestamp seja tratado como datetime, erros se tornam NaT
        timestamps = pd.to_datetime(df_window[timestamp_col_clean], errors='coerce')
        valid_timestamps = timestamps.dropna()
        
        delta_seconds = 0.0
        start_time = None
        
        if len(valid_timestamps) >= 2:
            delta_seconds = (valid_timestamps.iloc[-1] - valid_timestamps.iloc[0]).total_seconds()
        
        if not valid_timestamps.empty:
            start_time = valid_timestamps.iloc[0].strftime('%Y-%m-%d %H:%M:%S.%f')
        
    except Exception:
        delta_seconds = 0.0 
        start_time = 'Error'

    aggregated_row[f'{timestamp_col_clean}_Delta_Seconds'] = delta_seconds
    aggregated_row[f'{timestamp_col_clean}_Start'] = start_time
    
    if timestamp_col_clean in aggregated_row.index:
         aggregated_row = aggregated_row.drop(timestamp_col_clean)

    # Entropia de Shannon 
    for col_clean in diversity_cols_clean:
        entropy = shannon_entropy(df_window[col_clean].astype(str).dropna())
        aggregated_row[f'{col_clean}_Shannon_Entropy'] = entropy


      # O rótulo final é o mais frequente no intervalo de 1 segundo
    attack_labels = df_window[attack_label_col_clean].astype(str).str.strip().str.upper().replace('NAN', BENIGN_LABEL).dropna()
    final_string_label = attack_labels.mode().iloc[0] if not attack_labels.empty else BENIGN_LABEL

    # MANTÉM O RÓTULO AGREGADO NO LUGAR DA COLUNA ORIGINAL
    aggregated_row[attack_label_col_clean] = final_string_label 
    aggregated_row['Window_Packet_Count'] = len(df_window)
    
    return aggregated_row.to_frame().T


def concatenate_and_aggregate(date_folder: str, output_filename: str, data_path: str, file_names: List[str]):
    ordered_files = []
    base_path = os.path.join(data_path, date_folder)
    
    if not os.path.isdir(base_path):
        print(f"O caminho base '{base_path}' não foi encontrado.")
        return
    for file_name in file_names:
        file_path = os.path.join(base_path, file_name)
        if os.path.exists(file_path):
            ordered_files.append(file_path)
    if not ordered_files:
        print(f"Nenhum arquivo encontrado para o padrão '{base_path}/*.csv'.")
        return
    
    print(f"\nProcessando data: {date_folder} ({len(ordered_files)} arquivos)")
    print(f"-> Escrevendo o arquivo agregado para: {output_filename}")
    
    global_first_write = True
    
    # Buffer unificado para janelamento temporal
    aggregation_buffer = pd.DataFrame() 

    for file_path in ordered_files:
        print(f"-> Concatenando e agregando arquivo: {os.path.basename(file_path)}")
        
        try:
            chunker = pd.read_csv(
                file_path, 
                chunksize=PANDAS_CHUNK_SIZE, 
                low_memory=False, 
                skipinitialspace=True,
                dtype=str 
            )
        except Exception as e:
            print(f"!!! ERRO ao abrir o arquivo {os.path.basename(file_path)}: {e}")
            continue

        for chunk in chunker:
            
            chunk.columns = clean_column_names(chunk.columns)
            
            # Remove as colunas
            cols_to_drop_clean = [col.replace(' ', '_') for col in COLUMNS_TO_DROP]
            chunk = chunk.drop(columns=cols_to_drop_clean, errors='ignore')
            
            # Concatena o chunk lido ao buffer
            aggregation_buffer = pd.concat([aggregation_buffer, chunk], ignore_index=True)
            
            # Converte timestamps no buffer 
            buffer_ts = pd.to_datetime(aggregation_buffer[TIMESTAMP_COL.replace(' ', '_')], errors='coerce')
            valid_ts_buffer = buffer_ts.dropna()
            
            if valid_ts_buffer.empty:
                continue

            # Início da janela de tempo é o primeiro timestamp válido no buffer
            start_time = valid_ts_buffer.iloc[0]
            
            # Encontra o índice da última linha que está DENTRO da janela de 1 segundo
            time_diffs = (valid_ts_buffer - start_time).dt.total_seconds()
            
            # Índices de todas as linhas que estão dentro da janela de 1s
            window_indices = time_diffs[time_diffs < TIME_WINDOW_SECONDS].index
            
            # Verifica se há linhas suficientes para fechar uma janela de tempo
            while not window_indices.empty:
                # O último índice válido que faz parte da janela de 1s
                last_index_in_window = window_indices[-1]

                # A janela é tudo do início até o último índice
                window = aggregation_buffer.iloc[:last_index_in_window + 1]
                
                # Agrega a janela
                aggregated_row_df = aggregate_window_by_time(window)
                
                # Escrita
                header = global_first_write
                mode = 'w' if global_first_write else 'a'
                aggregated_row_df.to_csv(output_filename, mode=mode, header=header, index=False)
                global_first_write = False
                
                # Remove a janela processada do buffer e reseta o index
                aggregation_buffer = aggregation_buffer.iloc[last_index_in_window + 1:].reset_index(drop=True)

                # Re-calcula os tempos e índices para o próximo loop
                buffer_ts = pd.to_datetime(aggregation_buffer[TIMESTAMP_COL.replace(' ', '_')], errors='coerce')
                valid_ts_buffer = buffer_ts.dropna()
                
                if valid_ts_buffer.empty:
                    window_indices = pd.Index([]) # Força saída do while
                else:
                    start_time = valid_ts_buffer.iloc[0]
                    time_diffs = (valid_ts_buffer - start_time).dt.total_seconds()
                    window_indices = time_diffs[time_diffs < TIME_WINDOW_SECONDS].index
    
    # Processa o que sobrou no buffer como a última janela 
    if not aggregation_buffer.empty and global_first_write:
        # Se o buffer não estiver vazio e não houver sido escrito nada
        aggregated_row_df = aggregate_window_by_time(aggregation_buffer)
        aggregated_row_df.to_csv(output_filename, mode='w', header=True, index=False)
    elif not aggregation_buffer.empty:
        # Se o buffer não estiver vazio e já houver sido escrito algo
        aggregated_row_df = aggregate_window_by_time(aggregation_buffer)
        aggregated_row_df.to_csv(output_filename, mode='a', header=False, index=False)


    print(f"--- Processamento concluído para {date_folder}. O arquivo '{output_filename}' foi criado. ---")


# Caminho base dos datasets
DATASET_PATH = 'datasets/CICDDoS2019/'

ATTACK_ORDER = {
    '03-11': [
        'Portmap.csv', 'NetBIOS.csv', 'LDAP.csv', 'MSSQL.csv', 'UDP.csv', 'UDPLag.csv', 'Syn.csv'
    ],
    '01-12': [
        'DrDoS_NTP.csv', 'DrDoS_DNS.csv', 'DrDoS_LDAP.csv', 'DrDoS_MSSQL.csv', 'DrDoS_NetBIOS.csv', 'DrDoS_SNMP.csv', 'DrDoS_SSDP.csv', 'DrDoS_UDP.csv', 
        'UDPLag.csv', 'Syn.csv', 'TFTP.csv' 
    ]
}

OUTPUT_FILES = {
    '03-11': 'CICDDoS2019_03_11_Aggregated_Features_1sWindow.csv', 
    '01-12': 'CICDDoS2019_01_12_Aggregated_Features_1sWindow.csv' 
}

# Tamanho do chunking 
PANDAS_CHUNK_SIZE = 100000 

# Tamanho da janela temporal 
TIME_WINDOW_SECONDS = 0.05

# Colunas que serão tratadas de forma especial
TIMESTAMP_COL = 'Timestamp'
ATTACK_LABEL_COL = 'Label' 

# Colunas para cálculo de Entropia 
DIVERSITY_COLS = ['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'Protocol'] 

# Colunas lixo a ser removidas 
COLUMNS_TO_DROP = ['Unnamed: 0', 'Flow ID', 'SimillarHTTP']

# Constante para o rótulo Benigno
BENIGN_LABEL = 'BENIGN'


for date, output_file in OUTPUT_FILES.items():
    concatenate_and_aggregate(date, output_file, DATASET_PATH, ATTACK_ORDER[date])








