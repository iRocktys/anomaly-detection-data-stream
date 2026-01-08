import warnings
from sklearn.metrics import classification_report, confusion_matrix
from capymoa.stream import NumpyStream
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from math import pi, ceil
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split

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

def visualizar_radares_separados_referencia(df, features):
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

def visualizar_assinaturas_barras(df, top_n=5, amostras_por_classe=5000, n_iteracoes=10):
    cols_ignoradas = ['Origem_Arquivo', 'Label', 'Unnamed: 0', 'Flow ID', 'Timestamp']
    
    cols_numericas = [c for c in df.select_dtypes(include=[np.number]).columns if c not in cols_ignoradas]
    
    lista_ataques = df['Label'].unique()
    dados_plot = []

    print(f"Iniciando cálculo de estabilidade ({n_iteracoes} iterações por ataque)...")

    for ataque in lista_ataques:
        soma_importancias = np.zeros(len(cols_numericas))
        
        for i in range(n_iteracoes):
            rnd_state = np.random.randint(100000)
            
            amostras = []
            for _, grupo in df.groupby('Label'):
                n = min(len(grupo), amostras_por_classe)
                amostras.append(grupo.sample(n=n, random_state=rnd_state))
            
            df_sample = pd.concat(amostras)
            
            X = df_sample[cols_numericas].fillna(0)
            y = (df_sample['Label'] == ataque).astype(int)
            
            rf = RandomForestClassifier(
                n_estimators=50, 
                max_depth=10, 
                class_weight='balanced', 
                n_jobs=-1,
                random_state=rnd_state
            )
            
            rf.fit(X, y)
            soma_importancias += rf.feature_importances_

        media_importancias = soma_importancias / n_iteracoes
        
        indices_ordenados = np.argsort(media_importancias)[::-1]
        top_indices = indices_ordenados[:top_n]
        
        for idx in top_indices:
            dados_plot.append({
                'Ataque': ataque,
                'Característica': cols_numericas[idx],
                'Importância': media_importancias[idx]
            })
        
        print(f"-> Assinatura calculada: {ataque}")

    df_plot = pd.DataFrame(dados_plot)
    
    unique_features = df_plot['Característica'].unique()
    
    paletas_combinadas = (
        sns.color_palette("tab20", 20) + 
        sns.color_palette("Set1", 9) + 
        sns.color_palette("Paired", 12)
    )
    
    color_map = dict(zip(unique_features, paletas_combinadas[:len(unique_features)]))
    
    fig, ax = plt.subplots(figsize=(22, 12))
    
    attacks = df_plot['Ataque'].unique()
    x_positions = np.arange(len(attacks))
    bar_width = 0.15
    
    handles = {}
    
    for i, attack in enumerate(attacks):
        attack_data = df_plot[df_plot['Ataque'] == attack]
        
        num_bars = len(attack_data)
        offsets = (np.arange(num_bars) - (num_bars - 1) / 2) * bar_width
        
        for j, (_, row) in enumerate(attack_data.iterrows()):
            feature_name = row['Característica']
            importance = row['Importância']
            color = color_map[feature_name]
            
            bar = ax.bar(
                x_positions[i] + offsets[j], 
                importance, 
                width=bar_width, 
                color=color, 
                edgecolor='black', 
                linewidth=0.5
            )
            
            if feature_name not in handles:
                handles[feature_name] = bar[0]

    ax.set_title(f'Top {top_n} Características Distintivas por Ataque (Agrupadas)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Tipo de Ataque', fontsize=14)
    ax.set_ylabel('Importância Média (Gini)', fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(attacks, rotation=45, ha='right', fontsize=12)
    
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    ax.legend(
        handles.values(), 
        handles.keys(), 
        bbox_to_anchor=(1.01, 1), 
        loc='upper left', 
        borderaxespad=0., 
        title='Características',
        fontsize=10,
        title_fontsize=12,
        frameon=True,
        shadow=True
    )
    
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
        # print(f"Lista: {to_drop}") # Descomente se quiser ver os nomes
    
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
    
    # Substituir inf pelo maior valor finito da coluna 
    #    Se for inf positivo, pega o max. Se for negativo, pega o min.
    #    Para simplificar, substituímos por um valor teto muito alto.
    X.replace([np.inf], np.finfo(np.float32).max, inplace=True)
    X.replace([-np.inf], np.finfo(np.float32).min, inplace=True)
    
    # Substituir NaN pela Mediana
    X = X.fillna(X.median()).fillna(0) 
    
    # Aplica a redução de dimensionalidade antes de criar a stream 
    # Para produção deveria ser calculado antes
    X = remover_features_redundantes(X, threshold_corr=0.95)
    
    # Preparação do Target
    le = LabelEncoder()
    # Garante que seja string para o LabelEncoder funcionar bem
    y = le.fit_transform(df[target_label_col].astype(str))
    
    # Criação da Stream CapyMOA
    stream = NumpyStream(X.values, y, target_name=target_label_col, feature_names=X.columns.tolist())
    
    return stream, le, X.columns.tolist() 