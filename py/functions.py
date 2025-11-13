import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from capymoa.stream import NumpyStream

def criar_stream(file_path, target_label_col, timestamp_col, cols_para_remover, features_selecionadas=None):
    print(f"--- Iniciando Pipeline: {file_path} ---")
    
    df = pd.read_csv(file_path)
    df_processed = df.copy()

    # --- Renomear Colunas ---
    df_processed.columns = df_processed.columns.str.strip()
    target_label_col = target_label_col.strip()
    timestamp_col = timestamp_col.strip()
    print("  [Passo 2/6] Colunas renomeadas.")

    # --- Ordenar por Timestamp ---
    if timestamp_col in df_processed.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_processed[timestamp_col] = pd.to_datetime(df_processed[timestamp_col], errors='coerce')
        
        if not df_processed[timestamp_col].isnull().all():
            print(f"  [Passo 3/6] Ordenando DataFrame por '{timestamp_col}'...")
            df_processed.sort_values(by=timestamp_col, inplace=True)
            df_processed.reset_index(drop=True, inplace=True)
        else:
             print(f"  [Passo 3/6] Coluna de Timestamp encontrada, mas vazia. Não foi possível ordenar.")
    else:
        print(f"  [Passo 3/6] Aviso: Coluna de Timestamp '{timestamp_col}' não encontrada. O stream seguirá a ordem do CSV.")

    # --- Tratar Infinitos ---
    print("  [Passo 4/6] Convertendo valores Infinitos para NaN...")
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # --- Limpeza e Preparação de X/y ---
    print("  [Passo 5/6] Removendo colunas, tratando nulos e codificando rótulos...")
    
    if target_label_col not in df_processed.columns:
        print(f"    - ERRO: Coluna de rótulo '{target_label_col}' não encontrada.")
        return None, None
        
    # Codificar Rótulos (Globalmente)
    le = LabelEncoder()
    y_data_series = le.fit_transform(df_processed[target_label_col].astype(str))
    print(f"    - LabelEncoder criado e ajustado. {len(le.classes_)} classes encontradas.")
    
    #  Remover colunas desnecessárias
    cols_para_remover_normalizadas = [col.strip() for col in cols_para_remover]
    todas_cols_para_remover = [target_label_col, timestamp_col] + cols_para_remover_normalizadas
    cols_existentes_para_remover = [col for col in todas_cols_para_remover if col in df_processed.columns]
    
    X_data_df = df_processed.drop(columns=cols_existentes_para_remover, errors='ignore')
    print(f"    - {len(cols_existentes_para_remover)} colunas removidas do conjunto de features.")

    # Garantir X numérico
    X_data_df_numeric = X_data_df.select_dtypes(include=np.number)
    non_numeric_cols = X_data_df.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"    - Aviso: Removendo {len(non_numeric_cols)} colunas não numéricas que sobraram (ex: {non_numeric_cols[:3]}).")
    
    # Imputar NaNs (com Mediana)
    nan_counts = X_data_df_numeric.isnull().sum().sum()
    if nan_counts > 0:
        print(f"    - Imputando {nan_counts} valores nulos/infinitos com a MEDIANA...")
        X_data_df_cleaned = X_data_df_numeric.fillna(X_data_df_numeric.median()).fillna(0)
    else:
        print("    - Nenhum valor nulo/infinito encontrado.")
        X_data_df_cleaned = X_data_df_numeric

    # --- Seleção de Features  ---
    if features_selecionadas:
        print(f"    - Aplicando seleção de features. Mantendo {len(features_selecionadas)} colunas.")
        
        # Garante que os nomes na lista também estejam sem espaços
        features_selecionadas_clean = [col.strip() for col in features_selecionadas]
        
        # Verifica se todas as features selecionadas existem no DataFrame
        features_existentes = [col for col in features_selecionadas_clean if col in X_data_df_cleaned.columns]
        features_faltantes = set(features_selecionadas_clean) - set(features_existentes)
        
        if features_faltantes:
            print(f"    - Aviso: As seguintes features não foram encontradas e serão ignoradas: {features_faltantes}")
        
        if not features_existentes:
            print("    - ERRO: Nenhuma das features selecionadas foi encontrada no DataFrame. Abortando.")
            return None, None
            
        # Filtra o DataFrame para conter APENAS as features selecionadas
        X_data_df_cleaned = X_data_df_cleaned[features_existentes]
        
    # --- Criar Stream ---
    print("  [Passo 6/6] Criando objeto NumpyStream...")
    X_data = X_data_df_cleaned.values.astype(np.float64)
    y_data = y_data_series 
    
    print(f"    - Dados finais preparados: X_shape={X_data.shape}, y_shape={y_data.shape}.")

    if NumpyStream:
        try:
            stream = NumpyStream(
                X_data,
                y_data,
                target_name=target_label_col, 
                dataset_name=file_path.split('/')[-1] 
            )
            stream.restart() 
            print("Stream criado e pronto para uso.")
            return stream, le 
        except Exception as e:
            print(f"    - ERRO AO CRIAR STREAM: {e}")
            return None, None
    else:
        print("    - ERRO: Biblioteca 'capymoa' não encontrada.")
        return None, None


def plot_confusion_matrix(y_true, y_pred, label_encoder, model_name=""):    
    # Pega os índices conhecidos (ex: [0, 1, 2]) direto do encoder
    labels_indices = label_encoder.transform(label_encoder.classes_)
    labels_names = label_encoder.classes_

    # --- Relatório de Classificação ---
    report = classification_report(
        y_true, y_pred, 
        labels=labels_indices, 
        target_names=labels_names,
        zero_division=0, digits=4
    )
    print(report)

    # --- Heatmap da Matriz de Confusão ---
    cm = confusion_matrix(y_true, y_pred, labels=labels_indices)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, fmt="d", cmap="Blues", 
        xticklabels=labels_names, 
        yticklabels=labels_names
    )
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.xlabel('Rótulo Predito')
    plt.ylabel('Rótulo Real')
    plt.tight_layout()
    plt.show()