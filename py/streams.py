import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from capymoa.stream import NumpyStream
import matplotlib.pyplot as plt
import seaborn as sns

def removeFeatures(X, y, 
                   threshold_var=None, 
                   threshold_corr=None, 
                   top_n_features=None):
    
    initial_count = X.shape[1]
    print(f"\n--- Iniciando Processo de Seleção de Features (Total: {initial_count}) ---")

    # Remoção por Variância 
    if threshold_var is not None:
        selector = VarianceThreshold(threshold=threshold_var)
        selector.fit(X)
        cols_var = X.columns[selector.get_support()]
        removed_count = initial_count - len(cols_var)
        X = X[cols_var]
        print(f"Variância: {removed_count} features removidas. Restantes: {X.shape[1]}")
    else:
        print(f"Remoção de Variância: Pular.")

    # Remoção por Correlação de Pearson 
    if threshold_corr is not None:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold_corr)]
        X = X.drop(columns=to_drop)
        print(f"Correlação (>{threshold_corr}): {len(to_drop)} features redundantes removidas. Restantes: {X.shape[1]}")
    else:
        print(f"Remover Correlação: Pular.")

    # Random Forest Importance
    if top_n_features is not None:
        if X.shape[1] > top_n_features:
            rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=X.columns)
            selected_feats = importances.nlargest(top_n_features).index.tolist()
            X = X[selected_feats]
            print(f"Random Forest: Top {top_n_features} selecionadas.")
        else:
            print(f"Random Forest: Ignorado (Features atuais <= Top N).")
    else:
        print(f"Random Forest: Pular.")

    print(f"Features Finais ({X.shape[1]}) - {X.columns.tolist()}")
    print(f"--- Fim do Processo de Seleção de Features ---\n")

    return X

def normalizeData(X, method=None): 
    match method:
        case "MinMaxScaler":
            scaler = MinMaxScaler()
        case "StandardScaler":
            scaler = StandardScaler()
        case "RobustScaler":
            scaler = RobustScaler()
        case _:
            print(f"Normalização: Dados originais mantidos.")
            return X.values if hasattr(X, 'values') else X

    # Se caiu em um dos cases acima, aplica a transformação
    X_scaled = scaler.fit_transform(X)
    print(f"Normalização: {method}")
    return X_scaled

def handleMissingValues(X, method='0'):
    match str(method).lower():
        case 'media':
            print("Tratamento de Nulos: Preenchendo com a MÉDIA das colunas...")
            return X.fillna(X.mean())
        case 'mediana':
            print("Tratamento de Nulos: Preenchendo com a MEDIANA das colunas...")
            return X.fillna(X.median())
        case 'moda':
            print("Tratamento de Nulos: Preenchendo com a MODA das colunas...")
            # .mode() retorna um DataFrame, pegamos a primeira linha (índice 0)
            return X.fillna(X.mode().iloc[0])
        case '0':
            print("Tratamento de Nulos: Preenchendo com ZERO.")
            return X.fillna(0)
        case _:
            print(f"Aviso: Método de preenchimento '{method}' desconhecido. Usando ZERO por padrão.")
            return X.fillna(0)

def newStream(df, target_label_col='Label', binary_label=True, 
              normalize_method=None,
              threshold_var=None,
              threshold_corr=None,
              top_n_features=None,
              stream=True,
              extra_ignore_cols=None,
              imputation_method='0'):

    # Limpeza Básica
    print(f"Limpeza: Removendo espaços, identificadores (Flow ID, Timestamp, Unnamed: 0) e colunas vazias...")
    df.columns = df.columns.str.strip()
    target_label_col = target_label_col.strip()
    
    ignore_cols = ['Flow ID', 'Timestamp', 'SimillarHTTP', 'Unnamed: 0']
    
    if extra_ignore_cols:
        if isinstance(extra_ignore_cols, str):
            ignore_cols.append(extra_ignore_cols)
        else:
            ignore_cols.extend(extra_ignore_cols)
            
    cols_to_drop = [c for c in ignore_cols if c in df.columns]
    
    X = df.drop(columns=[target_label_col] + cols_to_drop, errors='ignore')
    
    # Tratamento Numérico
    print("Pré-processamento: Convertendo infinitos...")
    X = X.select_dtypes(include=[np.number])
    X.replace([np.inf, -np.inf], [np.finfo(np.float32).max, np.finfo(np.float32).min], inplace=True)
    X = handleMissingValues(X, method=imputation_method)

    # Normalização
    if normalize_method:
        col_names_temp = X.columns
        X_array_temp = normalizeData(X, method=normalize_method)
        X = pd.DataFrame(X_array_temp, columns=col_names_temp)

    # Definição do Target (y)
    type_lbl = "Binário (0=Normal, 1=Attack)" if binary_label else "Multiclasse"
    print(f"Target: Processando coluna '{target_label_col}' como {type_lbl}...")
    
    if binary_label:
        is_benign = df[target_label_col].astype(str).str.strip().str.upper() == 'BENIGN'
        y = np.where(is_benign, 0, 1).astype(np.int8)
        target_names = ['Normal', 'Attack'] 
    else:
        le = LabelEncoder()
        y = le.fit_transform(df[target_label_col].astype(str))
        target_names = le.classes_.tolist()

    # Redução da dimensionalidade
    if threshold_var is not None or threshold_corr is not None or top_n_features is not None:
        print("Seleção de Features: Iniciando pipeline de redução de dimensionalidade...")
        X = removeFeatures(X, y, 
                           threshold_var=threshold_var,
                           threshold_corr=threshold_corr,
                           top_n_features=top_n_features)
    else:
        print("Seleção de Features: Nenhuma técnica selecionada. Mantendo todas as colunas.")

    # Extrai dados finais para retorno
    feature_names = X.columns.tolist()
    X_array_final = X.values

    # Criação do Retorno
    if stream:
        print("Finalização: Criando objeto NumpyStream para o CapyMOA.\n")
        stream_obj = NumpyStream(
            X_array_final, 
            y, 
            target_name="Class", 
            feature_names=feature_names,
            target_type="categorical"
        )
        return stream_obj, target_names, feature_names
    else:
        print("Finalização: Retornando DataFrame pandas processado.\n")
        X_df = pd.DataFrame(X_array_final, columns=feature_names)
        return X_df, y, target_names