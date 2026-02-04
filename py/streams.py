import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from capymoa.stream import NumpyStream
import matplotlib.pyplot as plt
import seaborn as sns
import math

def removeFeatures(X, y, 
                   apply_variance=False, threshold_var=0.0, 
                   apply_corr=False, threshold_corr=0.90, 
                   apply_rfi=False, top_n_features=20):
    
    initial_count = X.shape[1]
    print(f"\n--- Iniciando Processo de Seleção de Features (Total: {initial_count}) ---")

    # Remoção por Variância 
    if apply_variance:
        selector = VarianceThreshold(threshold=threshold_var)
        selector.fit(X)
        cols_var = X.columns[selector.get_support()]
        removed_count = initial_count - len(cols_var)
        X = X[cols_var]
        print(f"Variância: {removed_count} features removidas. Restantes: {X.shape[1]}")
    else:
        print(f"Remoção de Variância: Pular.")

    # Remoção por Correlação de Pearson 
    if apply_corr:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold_corr)]
        X = X.drop(columns=to_drop)
        print(f"Correlação (>{threshold_corr}): {len(to_drop)} features redundantes removidas. Restantes: {X.shape[1]}")
    else:
        print(f"Remover Correlação: Pular.")

    # Random Forest Importance
    if apply_rfi:
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

def newStream(df, target_label_col='Label', binary_label=True, 
              normalize_method="RobustScaler",
              apply_variance=False, threshold_var=0.0,
              apply_corr=False, threshold_corr=0.90,
              apply_rfi=False, top_n_features=20,
              perform_sanity_check=True,
              stream=True):

    # Limpeza Básica
    print(f"Limpeza: Removendo espaços, identificadores (Flow ID, Timestamp) e colunas vazias...")
    df.columns = df.columns.str.strip()
    target_label_col = target_label_col.strip()
    
    ignore_cols = ['Flow ID', 'Timestamp', 'SimillarHTTP', 'Unnamed: 0']
    cols_to_drop = [c for c in ignore_cols if c in df.columns]

    if perform_sanity_check:
        print("Sanity Check: Analisando consistência numérica...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # CASO 1: Tratamento de Flags (-1)
        # O CICFlowMeter usa -1 para "Missing Value" em Init_Win_bytes. 
        # Vamos converter -1 para 0, pois fisicamente bytes não podem ser negativos.
        # Isso resolve o problema dos "57%" sem perder dados.
        cols_with_neg1 = []
        for col in numeric_cols:
            if (df[col] == -1).any():
                cols_with_neg1.append(col)
        
        if cols_with_neg1:
            print(f"   -> Ajustando flags '-1' para '0' em {len(cols_with_neg1)} colunas (comum em UDP).")
            # Substitui apenas onde é exatamente -1
            df.replace({col: {-1: 0} for col in cols_with_neg1}, inplace=True)

        # CASO 2: Tratamento de Overflow (Valores < -1)
        # Agora procuramos por lixo real (ex: -2 bilhões)
        deep_neg_mask = (df[numeric_cols] < 0).any(axis=1)
        cnt_deep_neg = deep_neg_mask.sum()
        
        if cnt_deep_neg > 0:
            pct_bad = (cnt_deep_neg / len(df)) * 100
            print(f"   -> ERRO CRÍTICO: {cnt_deep_neg} linhas com Overflow (< 0) detectadas ({pct_bad:.4f}%).")
            
            if pct_bad < 2.0:
                print("   -> Ação: REMOÇÃO (Poucas linhas afetadas).")
                df = df[~deep_neg_mask]
            else:
                print("   -> Ação: CLIPAGEM para 0 (Muitas linhas afetadas).")
                df[numeric_cols] = df[numeric_cols].clip(lower=0)
        else:
            print("   -> Nenhum overflow crítico detectado após ajuste de flags.")
    
    X = df.drop(columns=[target_label_col] + cols_to_drop, errors='ignore')
    
    # Tratamento Numérico
    print("Pré-processamento: Convertendo infinitos e preenchendo valores nulos...")
    X = X.select_dtypes(include=[np.number])
    X.replace([np.inf, -np.inf], [np.finfo(np.float32).max, np.finfo(np.float32).min], inplace=True)
    X = X.fillna(0)

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

    # Salva nomes das colunas antes de virar array numpy
    feature_names = X.columns.tolist()

    # Normalização
    if normalize_method:
        X_array = normalizeData(X, method=normalize_method)
    else:
        X_array = X.values

    # Seleção de Features 
    if apply_variance or apply_corr or apply_rfi:
        print("Seleção de Features: Iniciando pipeline de redução de dimensionalidade...")
        X = removeFeatures(X, y, 
                           apply_variance=apply_variance, threshold_var=threshold_var,
                           apply_corr=apply_corr, threshold_corr=threshold_corr,
                           apply_rfi=apply_rfi, top_n_features=top_n_features)
    else:
        print("Seleção de Features: Nenhuma técnica selecionada. Mantendo todas as colunas.")

    # Criação do Retorno
    if stream:
        print("Finalização: Criando objeto NumpyStream para o CapyMOA.")
        stream_obj = NumpyStream(
            X_array, 
            y, 
            target_name="Class", 
            feature_names=feature_names,
            target_type="categorical"
        )
        return stream_obj, target_names, feature_names
    else:
        print("Finalização: Retornando DataFrame pandas processado.")
        X_df = pd.DataFrame(X_array, columns=feature_names)
        return X_df, y, target_names