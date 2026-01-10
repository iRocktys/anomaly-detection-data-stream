import warnings
from sklearn.metrics import classification_report, confusion_matrix
from capymoa.stream import NumpyStream
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from math import pi, ceil

def carregar_e_unificar(lista_arquivos, CHUNK_SIZE):
    cols_to_ignore = [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 
        'Destination Port', 'Timestamp', 'SimillarHTTP', 'Unnamed: 0'
    ]
    
    df_list = []
    print(f"Iniciando processamento integral de {len(lista_arquivos)} arquivos...")
    
    for filepath in lista_arquivos:
        if not os.path.exists(filepath):
            continue
        
        with pd.read_csv(filepath, chunksize=CHUNK_SIZE, low_memory=False) as reader:
                for chunk in reader:
                    chunk.columns = chunk.columns.str.strip()
                    chunk = chunk.drop(columns=[c for c in cols_to_ignore if c in chunk.columns], errors='ignore')
                    chunk = chunk.dropna(axis=1, how='all')
                    df_list.append(chunk)

    if not df_list:
        return None
        
    df_final = pd.concat(df_list, ignore_index=True)
    df_final = df_final.replace([np.inf, -np.inf], np.nan)
    
    cols_numericas = df_final.select_dtypes(include=[np.number]).columns
    df_final[cols_numericas] = df_final[cols_numericas].fillna(df_final[cols_numericas].median())
    
    print(f"Dataset Unificado Pronto: {df_final.shape[0]} linhas.")
    return df_final

def visualizar_radares(df, features):
    df_grouped = df.groupby('Label')[features].mean()
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df_grouped), columns=features, index=df_grouped.index)
    
    try:
        df_benign = df_norm.loc[['BENIGN']]
        df_attacks = df_norm.drop('BENIGN')
    except KeyError:
        df_benign = None
        df_attacks = df_norm

    num_vars = len(features)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1] 

    if df_benign is not None:
        plt.figure(figsize=(5, 5))
        ax = plt.subplot(111, polar=True)
        values = df_benign.iloc[0].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, color='green', linestyle='solid')
        ax.fill(angles, values, color='green', alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, size=8, weight='bold', color='darkgreen')
        ax.tick_params(axis='x', pad=30)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color="grey", size=10)
        plt.show()

    if not df_attacks.empty:
        labels_ataques = df_attacks.index
        num_ataques = len(labels_ataques)
        cols = 4
        rows = ceil(num_ataques / cols)
        
        fig, axes = plt.subplots(rows, cols, subplot_kw=dict(polar=True), figsize=(20, 5 * rows))
        axes_flat = axes.flatten()
        colors = plt.cm.magma(np.linspace(0.2, 0.9, num_ataques))

        for i, ataque in enumerate(labels_ataques):
            ax = axes_flat[i]
            values = df_attacks.loc[ataque].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, color=colors[i])
            ax.fill(angles, values, color=colors[i], alpha=0.5)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_ylim(0, 1)
            ax.grid(True, color='grey', linestyle=':', alpha=0.5)
            ax.set_title(ataque, size=14, weight='bold', y=1.05)

        for i in range(num_ataques, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.tight_layout()
        plt.show()

def remover_features_redundantes(X, threshold_corr=0.95):
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

def criar_stream(df, target_label_col='Label'):
    cols_to_ignore = [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 
        'Destination Port', 'Timestamp', 'SimillarHTTP', 'Unnamed: 0'
    ]
    
    df.columns = df.columns.str.strip()
    target_label_col = target_label_col.strip()
    
    # Remoção das colunas ignoradas
    cols_present = [c for c in cols_to_ignore if c in df.columns]
    df = df.drop(columns=cols_present)
    
    # Separação X e y
    X = df.drop(columns=[target_label_col], errors='ignore')
    X = X.select_dtypes(include=[np.number]) 
    
    # Tratamento de infinitos e NaNs
    X.replace([np.inf], np.finfo(np.float32).max, inplace=True)
    X.replace([-np.inf], np.finfo(np.float32).min, inplace=True)
    X = X.fillna(X.median()).fillna(0) 
    
    # Redução de dimensionalidade
    X = remover_features_redundantes(X, threshold_corr=0.95)
    
    # Normalização 
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # Preparação do Target
    le = LabelEncoder()
    y = le.fit_transform(df[target_label_col].astype(str))
    
    stream = NumpyStream(
        X.values, 
        y, 
        target_name=target_label_col, 
        feature_names=X.columns.tolist(),
        target_type="categorical" 
    )
    
    return stream, le, X.columns.tolist()