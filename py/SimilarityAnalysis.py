import pandas as pd
import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

warnings.filterwarnings('ignore')

def create_balanced_dataset(
    src_dir, 
    dest_dir, 
    output_filename, 
    n_samples_per_class, 
    chunk_size=100000, 
    target_files=None,
    ignored_classes=None, 
    allow_insufficient=False, 
    logging=True
):
    """
    Parâmetros:
        src_dir (str): Pasta origem.
        dest_dir (str): Pasta destino.
        output_filename (str): Nome do arquivo final.
        n_samples_per_class (int): Meta de amostras.
        chunk_size (int): Quantidade de linhas para ler/processar por vez.
        target_files (list): Lista de nomes de arquivos específicos (ex: ['Wednesday.csv']). Se None, lê todos.
        ignored_classes (list): Classes a ignorar.
        allow_insufficient (bool): Permitir classes com menos amostras que a meta.
        logging (bool): Prints de status.
    """
    
    if ignored_classes is None: ignored_classes = []
    if not os.path.exists(dest_dir): os.makedirs(dest_dir)
    
    # Define quais arquivos processar
    if target_files:
        csv_files = [os.path.join(src_dir, f) for f in target_files]
        csv_files = [f for f in csv_files if os.path.exists(f)]
    else:
        csv_files = glob.glob(os.path.join(src_dir, "*.csv"))
    
    if not csv_files:
        if logging: print(f" [ERRO] Nenhum arquivo CSV válido encontrado em {src_dir}")
        return

    full_output_path = os.path.join(dest_dir, output_filename)
    
    if os.path.exists(full_output_path):
        os.remove(full_output_path)

    if logging: 
        print("="*80)
        print(f"PROCESSAMENTO OTIMIZADO (CHUNKS): {output_filename}")
        print(f"Arquivos Selecionados: {len(csv_files)}")
        print(f"Tamanho do Lote (Chunksize): {chunk_size}")
        if ignored_classes:
            print(f"Classes Ignoradas: {ignored_classes}")
        print("="*80)

    # PASSO 1: Contagem Global em Lotes
    if logging: print("\n[1/3] Varredura global (Lendo em lotes)...")
    
    global_counts = {}
    
    for file in csv_files:
        try:
            # ATUALIZAÇÃO: ignorando erros de encoding e bad lines
            header_df = pd.read_csv(file, nrows=0, encoding_errors='ignore', on_bad_lines='skip')
            col_label = next((col for col in header_df.columns if 'label' in col.lower()), None)
            
            if not col_label: continue

            # ATUALIZAÇÃO: ignorando erros de encoding e bad lines nos chunks
            for chunk in pd.read_csv(file, usecols=[col_label], chunksize=chunk_size, encoding_errors='ignore', on_bad_lines='skip'):
                chunk.columns = ['Label']
                counts = chunk['Label'].value_counts().to_dict()
                
                for label_class, qty in counts.items():
                    if label_class not in ignored_classes:
                        global_counts[label_class] = global_counts.get(label_class, 0) + qty
                
        except Exception as e:
            if logging: print(f" [!] Erro ao ler {os.path.basename(file)}: {e}")

    # PASSO 2: Validação da quantidade de amostras solicitadas
    if logging: print("[2/3] Validando quantidades disponíveis...")

    insufficient_classes = {}
    classes_to_collect = []

    for label_class, total in global_counts.items():
        if total < n_samples_per_class:
            insufficient_classes[label_class] = total
        else:
            classes_to_collect.append(label_class)

    if insufficient_classes:
        if not allow_insufficient:
            if logging:
                print("\n" + "!"*80)
                print(" [ERRO CRÍTICO] Classes insuficientes detectadas (Processo Interrompido):")
                for cls, total in insufficient_classes.items():
                    print(f"   -> {cls}: {total} (Meta: {n_samples_per_class})")
                print("!"*80)
            return
        else:
            # Mostrando quais são as classes insuficientes permitidas
            if logging: 
                insufficient_names = list(insufficient_classes.keys())
                print(f" [AVISO] Permitindo {len(insufficient_classes)} classes insuficientes: {insufficient_names}")
            classes_to_collect.extend(insufficient_classes.keys())
    
    if not classes_to_collect:
        if logging: print(" [ERRO] Nenhuma classe para coletar.")
        return

    # PASSO 3: Coleta e Escrita com buffer
    if logging: print(f"\n[3/3] Coletando e Salvando em disco (Lotes de {chunk_size})...")
    
    collected_samples_count = {cls: 0 for cls in classes_to_collect}
    
    buffer_list = []
    buffer_row_count = 0
    buffer_limit = chunk_size 
    first_write = True 

    for file in csv_files:
        all_done = True
        for cls in classes_to_collect:
            target = min(n_samples_per_class, global_counts[cls])
            if collected_samples_count[cls] < target:
                all_done = False
                break
        if all_done: break

        if logging: print(f"   -> Processando: {os.path.basename(file)}")

        try:
            # ignorando erros de encoding e bad lines nos chunks reais
            for chunk in pd.read_csv(file, chunksize=chunk_size, encoding_errors='ignore', on_bad_lines='skip'):
                chunk.columns = chunk.columns.str.strip()
                
                if 'Label' not in chunk.columns: continue

                chunk_selection = [] 

                for label_class in classes_to_collect:
                    target = min(n_samples_per_class, global_counts[label_class])
                    current = collected_samples_count[label_class]
                    
                    if current >= target:
                        continue
                    
                    df_class_chunk = chunk[chunk['Label'] == label_class]
                    
                    if df_class_chunk.empty:
                        continue

                    remaining = target - current
                    n_to_select = min(remaining, len(df_class_chunk))
                    
                    samples = df_class_chunk.sample(n=n_to_select, random_state=42)
                    chunk_selection.append(samples)
                    collected_samples_count[label_class] += n_to_select
                
                if chunk_selection:
                    chunk_merged = pd.concat(chunk_selection)
                    buffer_list.append(chunk_merged)
                    buffer_row_count += len(chunk_merged)

                if buffer_row_count >= buffer_limit:
                    df_buffer = pd.concat(buffer_list)
                    
                    mode = 'w' if first_write else 'a'
                    header = first_write 
                    
                    df_buffer.to_csv(full_output_path, mode=mode, header=header, index=False)
                    
                    if logging: print(f"      [IO] Buffer cheio ({buffer_row_count} linhas). Salvando lote no disco...")
                    
                    buffer_list = []
                    buffer_row_count = 0
                    first_write = False

        except Exception as e:
            if logging: print(f" [!] Erro ao processar chunk em {file}: {e}")

    # Escrita com o que ficou no buffer
    if buffer_list:
        df_buffer = pd.concat(buffer_list)
        mode = 'w' if first_write else 'a'
        header = first_write
        df_buffer.to_csv(full_output_path, mode=mode, header=header, index=False)
        if logging: print(f"      [IO] Salvando lote final ({len(df_buffer)} linhas)...")

    if logging:
        print("\n" + "="*80)
        print("CONCLUÍDO COM SUCESSO")
        print(f"Arquivo gerado: {full_output_path}")
        print("="*80)

def sort_dataset_by_timestamp(file_path, logging=True):
    """
    Parâmetros:
        file_path (str): Caminho completo para o arquivo CSV.
        logging (bool): Habilita os prints de progresso e análise de rótulos.
    """
    if not os.path.exists(file_path):
        if logging: print(f" [ERRO] Arquivo não encontrado: {file_path}")
        return
        
    if logging: 
        print("="*80)
        print(f"INICIANDO ORDENAÇÃO CRONOLÓGICA: {os.path.basename(file_path)}")
        print("="*80)
        
    try:
        # Lê o arquivo 
        df = pd.read_csv(file_path, encoding_errors='ignore', on_bad_lines='skip')
        
        # Limpa os nomes das colunas
        df.columns = df.columns.str.strip()
        
        # Identifica dinamicamente a coluna de tempo
        col_timestamp = None
        for col in df.columns:
            if 'timestamp' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
                col_timestamp = col
                break
                
        if not col_timestamp:
            if logging: print(" [ERRO] Nenhuma coluna de data/hora (Timestamp) foi encontrada no dataset.")
            return
            
        if logging: print(f" [INFO] Coluna de tempo identificada: '{col_timestamp}'. Convertendo dados...")
        
        # Converte a coluna para o tipo datetime do Pandas
        df[col_timestamp] = pd.to_datetime(df[col_timestamp], errors='coerce')
        
        if logging: print(" [INFO] Ordenando as amostras...")
        df = df.sort_values(by=col_timestamp)
        
        # Sobrescreve o mesmo arquivo
        if logging: print(f" [INFO] Salvando atualizações no arquivo: {file_path}...")
        df.to_csv(file_path, index=False)
        
        # Exibe a análise final
        if logging:
            print("\n" + "="*50)
            print(" ORDENAÇÃO CONCLUÍDA COM SUCESSO")
            print("="*50)
            if 'Label' in df.columns:
                print(f"Total de Amostras no Arquivo: {len(df)}")
                print("-" * 30)
                print("Contagem por Rótulo (Label):")
                print(df['Label'].value_counts())
            else:
                print(" [AVISO] Coluna 'Label' não encontrada para realizar a contagem.")
            print("="*50)
            
    except Exception as e:
        if logging: print(f" [!] Ocorreu um erro durante a ordenação: {e}")


def plot_feature_radar(df, y_labels, class_names):
    features = df.columns.tolist()
    num_features = len(features)
    
    if num_features == 0:
        return

    mean_df = df.groupby(y_labels).mean()

    feature_numbers = [str(i) for i in range(num_features)]
    
    angles = [n / float(num_features) * 2 * np.pi for n in range(num_features)]
    angles += angles[:1] 
    
    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw=dict(polar=True))
    
    for encoded_label in mean_df.index:
        try:
            name_of_class = class_names[int(encoded_label)]
        except:
            name_of_class = f"Classe {encoded_label}"
            
        values = mean_df.loc[encoded_label].tolist()
        values += values[:1] 
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=str(name_of_class))
        ax.fill(angles, values, alpha=0.15)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_numbers, fontsize=13, fontweight='bold', color='#333333')
    ax.set_ylim(0, 1)
    
    # Legenda (Classes)
    leg_classes = ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.05), 
                            title=r"$\bf{CLASSES}$", fontsize=11, title_fontsize=12, 
                            frameon=False) # Remove a caixa em volta
    ax.add_artist(leg_classes)
    
    # Legenda (Mapeamento)
    proxy_handles = [Line2D([0], [0], color='none', marker=f'${i}$', 
                            markersize=12, markerfacecolor='#333333', 
                            markeredgecolor='#333333') for i in range(num_features)]
    
    ax.legend(proxy_handles, features, loc='upper left', bbox_to_anchor=(1.15, 0.5), 
              title=r"$\bf{MAPEAMENTO\ DE\ FEATURES}$", fontsize=11, title_fontsize=12, 
              frameon=False)
                   
    plt.title("Perfil de Anomalias", y=1.08, fontweight='bold', fontsize=15)
    plt.subplots_adjust(left=0.05, right=0.65) 
    plt.show()

def plot_mini_radars(df, y_labels, class_names):
    features = df.columns.tolist()
    num_features = len(features)
    mean_df = df.groupby(y_labels).mean()
    num_classes = len(mean_df)
    
    feature_numbers = [str(i) for i in range(num_features)]
    angles = [n / float(num_features) * 2 * np.pi for n in range(num_features)]
    angles += angles[:1] 
    
    cols = 3 if num_classes >= 3 else num_classes
    rows = math.ceil(num_classes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5.5 * rows), subplot_kw=dict(polar=True))
    
    if num_classes == 1: axes = [axes]
    else: axes = axes.flatten()
        
    cmap = plt.get_cmap('tab20')
    lines_for_legend = []
    labels_for_legend = []
    
    # Escala
    r_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    r_labels = ['0.2', '0.4', '0.6', '0.8', '1.0']
    
    for idx, encoded_label in enumerate(mean_df.index):
        ax = axes[idx]
        color = cmap(idx % 20)
        
        try:
            name_of_class = class_names[int(encoded_label)]
        except:
            name_of_class = f"Classe {encoded_label}"
            
        values = mean_df.loc[encoded_label].tolist()
        values += values[:1] 
        
        # Plotagem principal
        line, = ax.plot(angles, values, linewidth=2.5, linestyle='solid', color=color)
        ax.fill(angles, values, alpha=0.2, color=color)
        
        # Ajustes de Eixos e Grid
        ax.grid(False) 
        ax.set_yticklabels([]) 
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_numbers, fontsize=11, fontweight='bold', color='#333333')
        
        # ângulo de 22.5 graus costuma ser um respiro entre as features 0 e 1
        ax.set_ylim(0, 1.1) 
        ax.spines['polar'].set_visible(True)
        ax.spines['polar'].set_color('#cccccc')
        
        # rgrids com ângulo ajustado para não sobrepor
        ax.set_rgrids(r_ticks, labels=r_labels, angle=22.5, fontsize=8, color='black')
        ax.set_title(str(name_of_class).upper(), y=1.1, fontweight='bold', color=color, fontsize=13)
        
        lines_for_legend.append(line)
        labels_for_legend.append(str(name_of_class))
        
    # Limpeza de subplots excedentes
    for i in range(num_classes, len(axes)):
        fig.delaxes(axes[i])
        
    # Legendas com títulos
    fig.legend(lines_for_legend, labels_for_legend, loc='upper left', bbox_to_anchor=(0.82, 0.92), 
               title=r"$\bf{CLASSES}$", title_fontsize=14, fontsize=12, frameon=False)
    
    proxy_handles = [Line2D([0], [0], color='none', marker=f'${i}$', 
                            markersize=12, markerfacecolor='#333333', 
                            markeredgecolor='#333333') for i in range(num_features)]
    
    fig.legend(proxy_handles, features, loc='upper left', bbox_to_anchor=(0.82, 0.75), 
               title=r"$\bf{MAPEAMENTO\ DE\ FEATURES}$", title_fontsize=14, fontsize=12, frameon=False)
               
    plt.suptitle("Perfil de Anomalias", y=0.95, fontweight='bold', fontsize=18)
    plt.subplots_adjust(left=0.05, right=0.8, wspace=0.45, hspace=0.7) 
    plt.show()