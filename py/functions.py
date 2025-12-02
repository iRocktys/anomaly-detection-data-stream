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
            
        nome_arquivo = os.path.basename(filepath)
        print(f"--> Lendo: {nome_arquivo}")
        
        try:
            with pd.read_csv(filepath, chunksize=CHUNK_SIZE, low_memory=False) as reader:
                for chunk in reader:
                    chunk.columns = chunk.columns.str.strip()
                    chunk = chunk.drop(columns=[c for c in cols_to_ignore if c in chunk.columns], errors='ignore')
                    chunk = chunk.dropna(axis=1, how='all')
                    chunk['Origem_Arquivo'] = nome_arquivo
                    df_list.append(chunk)
        except Exception as e:
            print(f"Erro crítico no arquivo {nome_arquivo}: {e}")

    if not df_list:
        return None
        
    df_final = pd.concat(df_list, ignore_index=True)
    df_final = df_final.replace([np.inf, -np.inf], np.nan)
    
    cols_numericas = df_final.select_dtypes(include=[np.number]).columns
    df_final[cols_numericas] = df_final[cols_numericas].fillna(df_final[cols_numericas].median())
    
    print(f"Dataset Unificado Pronto: {df_final.shape[0]} linhas.")
    return df_final

def remover_colunas_correlacionadas(df, limiar):
    print(f"Iniciando análise de correlação (Limiar: {limiar})...")
    df_numerico = df.select_dtypes(include=[np.number])
    matriz_corr = df_numerico.corr(method='pearson').abs()
    upper = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))
    colunas_para_remover = [column for column in upper.columns if any(upper[column] > limiar)]
    print(f"Removendo {len(colunas_para_remover)} colunas redundantes.")
    return df.drop(columns=colunas_para_remover)

def identificar_features_importantes(df, top_n=10):
    X = df.select_dtypes(include=[np.number]).fillna(0)
    y = df['Label']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X, y_encoded)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    melhores_features = [X.columns[i] for i in indices[:top_n]]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=importances[indices[:top_n]], y=melhores_features, hue=melhores_features, legend=False, palette='viridis')
    plt.title('Top Features Distintivas entre Ataques')
    plt.xlabel('Importância (Gini)')
    plt.tight_layout()
    plt.show()
    
    return melhores_features

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

def analisar_grupos_similares(df, features, limiar=0.85):
    print(f"Clusterizando grupos com similaridade rígida > {limiar*100}%...")
    
    df_grouped = df.groupby('Label')[features].mean()
    
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df_grouped), 
                           index=df_grouped.index, 
                           columns=features)
    
    sim_matrix = cosine_similarity(df_norm)
    df_sim = pd.DataFrame(sim_matrix, index=df_grouped.index, columns=df_grouped.index)
    
    plt.figure(figsize=(12, 10))
    mask = df_sim < limiar
    np.fill_diagonal(mask.values, True)
    
    sns.heatmap(df_sim, annot=True, fmt=".0%", cmap="Greens", mask=mask,
                linewidths=.5, cbar_kws={'label': 'Nível de Semelhança'})
    
    plt.title(f"Matriz de Similaridade (Corte: {int(limiar*100)}%)", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    dist_matrix = 1 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)
    
    condensed_dist = squareform(dist_matrix, checks=False)
    condensed_dist[condensed_dist < 0] = 0
    
    Z = linkage(condensed_dist, method='average')
    
    max_d = 1 - limiar
    labels = fcluster(Z, t=max_d, criterion='distance')
    
    grupos_dict = {}
    for ataque, cluster_id in zip(df_sim.index, labels):
        if cluster_id not in grupos_dict:
            grupos_dict[cluster_id] = []
        grupos_dict[cluster_id].append(ataque)
    
    tabela_dados = []
    
    for cluster_id, membros in grupos_dict.items():
        membros = sorted(membros)
        qtd = len(membros)
        
        identificacao = f"Grupo {cluster_id}"
        if qtd == 1:
            identificacao = "Isolado"
            sim_media = "N/A"
            conexoes_str = "Nenhuma (Ataque Único)"
        else:
            sub_matrix = df_sim.loc[membros, membros]
            mask_sub = np.ones(sub_matrix.shape, dtype=bool)
            np.fill_diagonal(mask_sub, False)
            mean_sim = sub_matrix.values[mask_sub].mean()
            sim_media = f"{mean_sim:.1%}"
            
            conexoes_fortes = []
            upper_tri = sub_matrix.where(np.triu(np.ones(sub_matrix.shape), k=1).astype(bool))
            for col in upper_tri.columns:
                for idx in upper_tri.index:
                    val = upper_tri.loc[idx, col]
                    if pd.notna(val) and val >= limiar:
                         conexoes_fortes.append(f"{idx}-{col} ({val:.0%})")
            
            conexoes_str = "; ".join(conexoes_fortes) if conexoes_fortes else "Alta coesão interna"

        tabela_dados.append({
            'Identificação': identificacao,
            'Membros': ", ".join(membros),
            'Qtd': qtd,
            'Similaridade Média': sim_media,
            'Detalhe das Conexões': conexoes_str
        })
        
    df_grupos = pd.DataFrame(tabela_dados)
    df_grupos = df_grupos.sort_values(by=['Identificação', 'Qtd'], ascending=[True, False]).reset_index(drop=True)
    
    pd.set_option('display.max_colwidth', None)
    
    return df_grupos

def identificar_assinaturas_por_ataque(df, top_n=5, amostras_por_classe=5000, n_iteracoes=10):
    cols_numericas = df.select_dtypes(include=[np.number]).columns
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

def criar_stream(df, target_label_col='Label'):
    cols_to_ignore = [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 
        'Destination Port', 'Timestamp', 'SimillarHTTP', 'Unnamed: 0'
    ]
    
    df.columns = df.columns.str.strip()
    target_label_col = target_label_col.strip()
    
    cols_present = [c for c in cols_to_ignore if c in df.columns]
    df = df.drop(columns=cols_present)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    X = df.drop(columns=[target_label_col], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median()).fillna(0)
    
    le = LabelEncoder()
    y = le.fit_transform(df[target_label_col].astype(str))
    
    stream = NumpyStream(X.values, y, target_name=target_label_col)
    
    return stream, le

def plot_confusion_matrix(y_true, y_pred, label_encoder, model_name=""):    
    labels_indices = label_encoder.transform(label_encoder.classes_)
    labels_names = label_encoder.classes_

    report = classification_report(
        y_true, y_pred, 
        labels=labels_indices, 
        target_names=labels_names,
        zero_division=0, digits=4
    )
    print(report)

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