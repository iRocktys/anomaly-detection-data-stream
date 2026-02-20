import pandas as pd
import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math


warnings.filterwarnings('ignore')

class DatasetHandler:
    """
    classe responsável por manipular, balancear, ordenar e visualizar 
    datasets de anomalias e tráfego de rede.
    """
    
    def __init__(self, logging=True):
        self.logging = logging

    def _log(self, message):
        if self.logging:
            print(message)

    def create_balanced_dataset(
        self, 
        src_dir, 
        dest_dir, 
        output_filename, 
        n_samples_per_class, 
        chunk_size=100000, 
        target_files=None,
        ignored_classes=None, 
        allow_insufficient=False
    ):
        
        # inicializa lista de classes ignoradas
        if ignored_classes is None: ignored_classes = []
        if not os.path.exists(dest_dir): os.makedirs(dest_dir)
        
        # define quais arquivos processar
        if target_files:
            csv_files = [os.path.join(src_dir, f) for f in target_files]
            csv_files = [f for f in csv_files if os.path.exists(f)]
        else:
            csv_files = glob.glob(os.path.join(src_dir, "*.csv"))
        
        if not csv_files:
            self._log(f" [ERRO] Nenhum arquivo CSV válido encontrado em {src_dir}")
            return

        full_output_path = os.path.join(dest_dir, output_filename)
        
        if os.path.exists(full_output_path):
            os.remove(full_output_path)

        self._log("="*80)
        self._log(f"PROCESSAMENTO OTIMIZADO (CHUNKS): {output_filename}")
        self._log(f"Arquivos Selecionados: {len(csv_files)}")
        self._log(f"Tamanho do Lote (Chunksize): {chunk_size}")
        if ignored_classes:
            self._log(f"Classes Ignoradas: {ignored_classes}")
        self._log("="*80)

        # varredura global lendo em lotes
        self._log("\n[*] Varredura global (Lendo em lotes)...")
        
        global_counts = {}
        
        for file in csv_files:
            try:
                # ignorando erros de encoding e bad lines
                header_df = pd.read_csv(file, nrows=0, encoding_errors='ignore', on_bad_lines='skip')
                col_label = next((col for col in header_df.columns if 'label' in col.lower()), None)
                
                if not col_label: continue

                # ignorando erros de encoding e bad lines nos chunks
                for chunk in pd.read_csv(file, usecols=[col_label], chunksize=chunk_size, encoding_errors='ignore', on_bad_lines='skip'):
                    chunk.columns = ['Label']
                    counts = chunk['Label'].value_counts().to_dict()
                    
                    for label_class, qty in counts.items():
                        if label_class not in ignored_classes:
                            global_counts[label_class] = global_counts.get(label_class, 0) + qty
                            
            except Exception as e:
                self._log(f" [!] Erro ao ler {os.path.basename(file)}: {e}")

        # validação da quantidade de amostras solicitadas
        self._log("[*] Validando quantidades disponíveis...")

        insufficient_classes = {}
        classes_to_collect = []

        for label_class, total in global_counts.items():
            if total < n_samples_per_class:
                insufficient_classes[label_class] = total
            else:
                classes_to_collect.append(label_class)

        if insufficient_classes:
            if not allow_insufficient:
                self._log("\n" + "!"*80)
                self._log(" [ERRO CRÍTICO] Classes insuficientes detectadas (Processo Interrompido):")
                for cls, total in insufficient_classes.items():
                    self._log(f"   -> {cls}: {total} (Meta: {n_samples_per_class})")
                self._log("!"*80)
                return
            else:
                # mostrando quais são as classes insuficientes permitidas
                insufficient_names = list(insufficient_classes.keys())
                self._log(f" [AVISO] Permitindo {len(insufficient_classes)} classes insuficientes: {insufficient_names}")
                classes_to_collect.extend(insufficient_classes.keys())
        
        if not classes_to_collect:
            self._log(" [ERRO] Nenhuma classe para coletar.")
            return

        # coleta e escrita com buffer
        self._log(f"\n[*] Coletando e Salvando em disco (Lotes de {chunk_size})...")
        
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

            self._log(f"   -> Processando: {os.path.basename(file)}")

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
                        
                        class_chunk_df = chunk[chunk['Label'] == label_class]
                        
                        if class_chunk_df.empty:
                            continue

                        remaining = target - current
                        n_to_select = min(remaining, len(class_chunk_df))
                        
                        samples = class_chunk_df.sample(n=n_to_select, random_state=42)
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
                        
                        self._log(f"      [IO] Buffer cheio ({buffer_row_count} linhas). Salvando lote no disco...")
                        
                        buffer_list = []
                        buffer_row_count = 0
                        first_write = False

            except Exception as e:
                self._log(f" [!] Erro ao processar chunk em {file}: {e}")

        # escrita com o que ficou no buffer
        if buffer_list:
            df_buffer = pd.concat(buffer_list)
            mode = 'w' if first_write else 'a'
            header = first_write
            df_buffer.to_csv(full_output_path, mode=mode, header=header, index=False)
            self._log(f"      [IO] Salvando lote final ({len(df_buffer)} linhas)...")

        self._log("\n" + "="*80)
        self._log("CONCLUÍDO COM SUCESSO")
        self._log(f"Arquivo gerado: {full_output_path}")
        self._log("="*80)

    def sort_dataset_by_timestamp(self, file_path):
        if not os.path.exists(file_path):
            self._log(f" [ERRO] Arquivo não encontrado: {file_path}")
            return
            
        self._log("="*80)
        self._log(f"INICIANDO ORDENAÇÃO CRONOLÓGICA: {os.path.basename(file_path)}")
        self._log("="*80)
            
        try:
            # lê o arquivo 
            df = pd.read_csv(file_path, encoding_errors='ignore', on_bad_lines='skip')
            
            # limpa os nomes das colunas
            df.columns = df.columns.str.strip()
            
            # identifica dinamicamente a coluna de tempo
            timestamp_col = None
            for col in df.columns:
                if 'timestamp' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
                    timestamp_col = col
                    break
                    
            if not timestamp_col:
                self._log(" [ERRO] Nenhuma coluna de data/hora (Timestamp) foi encontrada no dataset.")
                return
                
            self._log(f" [INFO] Coluna de tempo identificada: '{timestamp_col}'. Convertendo dados...")
            
            # converte a coluna para o tipo datetime do pandas
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            
            self._log(" [INFO] Ordenando as amostras...")
            df = df.sort_values(by=timestamp_col)
            
            # sobrescreve o mesmo arquivo
            self._log(f" [INFO] Salvando atualizações no arquivo: {file_path}...")
            df.to_csv(file_path, index=False)
            
            # exibe a análise final
            self._log("\n" + "="*50)
            self._log(" ORDENAÇÃO CONCLUÍDA COM SUCESSO")
            self._log("="*50)
            if 'Label' in df.columns:
                self._log(f"Total de Amostras no Arquivo: {len(df)}")
                self._log("-" * 30)
                self._log("Contagem por Rótulo (Label):")
                self._log(df['Label'].value_counts())
            else:
                self._log(" [AVISO] Coluna 'Label' não encontrada para realizar a contagem.")
            self._log("="*50)
                
        except Exception as e:
            self._log(f" [!] Ocorreu um erro durante a ordenação: {e}")

    def plot_feature_radar(self, df, y_labels, class_names):
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
                class_name = class_names[int(encoded_label)]
            except:
                class_name = f"Classe {encoded_label}"
                
            values = mean_df.loc[encoded_label].tolist()
            values += values[:1] 
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=str(class_name))
            ax.fill(angles, values, alpha=0.15)
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_numbers, fontsize=13, fontweight='bold', color='#333333')
        ax.set_ylim(0, 1)
        
        # legenda principal das classes
        leg_classes = ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.05), 
                                title=r"$\bf{CLASSES}$", fontsize=11, title_fontsize=12, 
                                frameon=False) 
        ax.add_artist(leg_classes)
        
        # legenda de mapeamento
        proxy_handles = [
            Line2D([0], [0], color='none', marker=f'${i}$', 
                   markersize=12, markerfacecolor='#333333', 
                   markeredgecolor='#333333') 
            for i in range(num_features)
        ]
        
        ax.legend(proxy_handles, features, loc='upper left', bbox_to_anchor=(1.15, 0.5), 
                  title=r"$\bf{MAPEAMENTO\ DE\ FEATURES}$", fontsize=11, title_fontsize=12, 
                  frameon=False)
                   
        plt.title("Perfil de Anomalias", y=1.08, fontweight='bold', fontsize=15)
        plt.subplots_adjust(left=0.05, right=0.65) 
        plt.show()

    def plot_mini_radars(self, df, y_labels, class_names):
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
        
        # escala
        r_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        r_labels = ['0.2', '0.4', '0.6', '0.8', '1.0']
        
        for idx, encoded_label in enumerate(mean_df.index):
            ax = axes[idx]
            color = cmap(idx % 20)
            
            try:
                class_name = class_names[int(encoded_label)]
            except:
                class_name = f"Classe {encoded_label}"
                
            values = mean_df.loc[encoded_label].tolist()
            values += values[:1] 
            
            # plotagem principal
            line, = ax.plot(angles, values, linewidth=2.5, linestyle='solid', color=color)
            ax.fill(angles, values, alpha=0.2, color=color)
            
            # ajustes de eixos e grid
            ax.grid(False) 
            ax.set_yticklabels([]) 
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feature_numbers, fontsize=11, fontweight='bold', color='#333333')
            
            # ângulo de respiro
            ax.set_ylim(0, 1.1) 
            ax.spines['polar'].set_visible(True)
            ax.spines['polar'].set_color('#cccccc')
            
            # rgrids com ângulo ajustado para não sobrepor
            ax.set_rgrids(r_ticks, labels=r_labels, angle=22.5, fontsize=8, color='black')
            ax.set_title(str(class_name).upper(), y=1.1, fontweight='bold', color=color, fontsize=13)
            
            lines_for_legend.append(line)
            labels_for_legend.append(str(class_name))
            
        # limpeza de subplots excedentes
        for i in range(num_classes, len(axes)):
            fig.delaxes(axes[i])
            
        # legendas com títulos
        fig.legend(lines_for_legend, labels_for_legend, loc='upper left', bbox_to_anchor=(0.82, 0.92), 
                   title=r"$\bf{CLASSES}$", title_fontsize=14, fontsize=12, frameon=False)
        
        proxy_handles = [
            Line2D([0], [0], color='none', marker=f'${i}$', 
                   markersize=12, markerfacecolor='#333333', 
                   markeredgecolor='#333333') 
            for i in range(num_features)
        ]
        
        fig.legend(proxy_handles, features, loc='upper left', bbox_to_anchor=(0.82, 0.75), 
                   title=r"$\bf{MAPEAMENTO\ DE\ FEATURES}$", title_fontsize=14, fontsize=12, frameon=False)
                   
        plt.suptitle("Perfil de Anomalias", y=0.95, fontweight='bold', fontsize=18)
        plt.subplots_adjust(left=0.05, right=0.8, wspace=0.45, hspace=0.7) 
        plt.show()