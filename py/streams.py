from capymoa.stream import NumpyStream
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler

def removeFeatures(X, threshold_corr=0.95):
    # Remover colunas com desvio padrão zero
    X = X.loc[:, X.std() > 0]
    
    # Remover colunas altamente correlacionadas 
    corr_matrix = X.corr().abs()
    
    # Seleciona o triângulo superior da matriz para não processar duas vezes
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identifica colunas onde a correlação é maior que o threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold_corr)]
    
    if to_drop:
        print(f"Features redundantes removidas: {len(to_drop)}")
    
    return X.drop(columns=to_drop)

def newStream(df, target_label_col='Label', selected_features=None, 
              binary_label=True, remove_redundant=False):
    
    # Limpeza de nomes de colunas
    df.columns = df.columns.str.strip()
    target_label_col = target_label_col.strip()
    
    # Definição das Features
    X_raw = df.drop(columns=[target_label_col], errors='ignore')

    if selected_features:
        feats = [f.strip() for f in selected_features if f.strip() in X_raw.columns]
        if not feats: raise ValueError("Nenhuma feature selecionada encontrada!")
        X = X_raw[feats].copy()
    else:
        ignore = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 
                  'Destination Port', 'Timestamp', 'SimillarHTTP', 'Unnamed: 0']
        X = X_raw.drop(columns=[c for c in ignore if c in X_raw.columns])
    
    # Tratamento numérico
    X = X.select_dtypes(include=[np.number])
    X.replace([np.inf, -np.inf], [np.finfo(np.float32).max, np.finfo(np.float32).min], inplace=True)
    X = X.fillna(0)
    
    # Definição do y (Target)
    if binary_label:
        is_benign = df[target_label_col].astype(str).str.strip().str.upper() == 'BENIGN'
        y = np.where(is_benign, 0, 1).astype(np.int8)
        target_names = ['Normal', 'Attack'] 
    else:
        le = LabelEncoder()
        y = le.fit_transform(df[target_label_col].astype(str))
        target_names = le.classes_.tolist()

    # Normalização e Instanciação
    X_scaled = RobustScaler().fit_transform(X)
    
    stream = NumpyStream(
        X_scaled, 
        y, 
        target_name=target_label_col, 
        feature_names=X.columns.tolist(),
        target_type="categorical"
    )
    
    return stream, target_names, X.columns.tolist()