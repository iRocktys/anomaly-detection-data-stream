import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

class Metrics:
    def get_metric_classifier(self, metrics_dict, metric_name, target_class=1):
        norm_metrics = {str(k).lower(): v for k, v in metrics_dict.items()}
        metric_name = str(metric_name).lower()
        
        if target_class is None:
            val_0 = norm_metrics.get(f'{metric_name}_0', 0.0)
            val_1 = norm_metrics.get(f'{metric_name}_1', 0.0)
            
            val_0 = 0.0 if val_0 is None or np.isnan(val_0) else float(val_0)
            val_1 = 0.0 if val_1 is None or np.isnan(val_1) else float(val_1)
            
            return (val_0 + val_1) / 2.0
            
        else:
            val = norm_metrics.get(f'{metric_name}_{target_class}')
            return float(val) if val is not None and not np.isnan(val) else 0.0

    def calc_sklearn_metrics(self, y_true, y_pred, target_class):
        if len(y_true) == 0:
            return 0.0, 0.0, 0.0
            
        if target_class is None:
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            f1 = f1_score(y_true, y_pred, pos_label=target_class, average='binary', zero_division=0)
            prec = precision_score(y_true, y_pred, pos_label=target_class, average='binary', zero_division=0)
            rec = recall_score(y_true, y_pred, pos_label=target_class, average='binary', zero_division=0)
            
        return f1 * 100.0, prec * 100.0, rec * 100.0

    def display_cumulative_metrics(self, predictions_history, warmup_instances=0, target_class=1):
        print(f"\n{'='*65}")
        print(f"{'RELATÓRIO ACUMULATIVO':^65}")
        print(f"{'='*65}")
        print(f"{'Algoritmo':<25} | {'F1 (%)':<10} | {'Prec (%)':<10} | {'Rec (%)':<10}")
        print(f"{'-'*65}")

        for name, data in predictions_history.items():
            y_true_list = data['true_labels'][warmup_instances:] if len(data['true_labels']) > warmup_instances else data['true_labels']
            y_pred_list = data['predicted_classes'][warmup_instances:] if len(data['predicted_classes']) > warmup_instances else data['predicted_classes']
            
            f1, prec, recall = self.calc_sklearn_metrics(y_true_list, y_pred_list, target_class)

            print(f"{name:<25} | {f1:<10.2f} | {prec:<10.2f} | {recall:<10.2f}")
        
        print(f"{'='*65}\n")


class Plots:
    def __init__(self, target_names):
        self.target_names = target_names
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        self.bg_colors = ['#F7C5CD', '#C5D9F7', '#C5F7C5', '#F7E6C5', '#E3C5F7', '#F7D9C5', '#C5F7E6']

    def plot_score(self, results, attack_regions, title, threshold=0.5):
        fig, ax = plt.subplots(figsize=(15, 6)) 
        
        for i, (name, data) in enumerate(results.items()):
            color = self.colors[i % len(self.colors)]
            scores = np.array(data['scores'])
            instances = np.arange(len(scores))
            window_size = 5
            
            moving_avg = np.array([np.mean(scores[max(0, j-window_size):j+1]) for j in range(len(scores))])
            ax.plot(instances, moving_avg, color=color, alpha=0.85, linewidth=1.5, label=f'{name}', zorder=3)

        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Threshold ({threshold})', zorder=4)

        added_attack_labels = set()
        for start, end, attack_idx in attack_regions:
            attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
            bg_color = self.bg_colors[attack_idx % len(self.bg_colors)]
            
            label_to_show = f'{attack_name}' if attack_name not in added_attack_labels else ""
            ax.axvspan(start, end, facecolor=bg_color, alpha=0.3, zorder=1, label=label_to_show)
            
            if label_to_show:
                added_attack_labels.add(attack_name)

        ax.set_title(f"Análise de Scores (Média Móvel - Janela {window_size})", fontsize=14, fontweight='bold')
        ax.set_ylabel("Score de Anomalia", fontsize=14)
        ax.set_xlabel("Instâncias", fontsize=14)
        
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(results) + 2, 
                  fontsize=12, frameon=False)
        for patch in leg.get_patches():
            patch.set_edgecolor('gray')
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)

        ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
        fig.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        plt.show()

    def plot_metrics(self, results, attack_regions, title, window_size):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        for i, (name, data) in enumerate(results.items()):
            color = self.colors[i % len(self.colors)]
            ax1.plot(data['instances'], data['f1_score'], label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
            ax2.plot(data['instances'], data['precision'], label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
            ax3.plot(data['instances'], data['recall'], label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)

        for ax in [ax1, ax2, ax3]:
            added_attack_labels = set()
            for start, end, attack_idx in attack_regions:
                attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
                bg_color = self.bg_colors[attack_idx % len(self.bg_colors)]
                
                label_to_show = f'{attack_name}' if attack_name not in added_attack_labels else ""
                ax.axvspan(start, end, facecolor=bg_color, alpha=0.4, zorder=2, label=label_to_show)
                
                if label_to_show:
                    added_attack_labels.add(attack_name)
                    
            ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax1.set_title(f"{title} (Resolução de {window_size} instâncias)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("F1-Score (%)", fontsize=14)
        ax2.set_ylabel("Precision (%)", fontsize=14)
        ax3.set_ylabel("Recall (%)", fontsize=14)
        ax3.set_xlabel("Instâncias", fontsize=14)

        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=len(results) + 2, 
                  fontsize=12, frameon=False)
        for patch in leg.get_patches():
            patch.set_edgecolor('gray')
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)
            
        fig.subplots_adjust(bottom=0.3)
        plt.tight_layout()
        plt.show()

    def plot_dspot_score(self, results, attack_regions, title):
        fig, ax = plt.subplots(figsize=(15, 6)) 
        
        window_size = 5
        
        for i, (name, data) in enumerate(results.items()):
            color = self.colors[i % len(self.colors)]
            scores = np.array(data['scores'])
            thresholds = np.array(data['thresholds'])
            trends = np.array(data['trends'])
            
            instances = np.arange(len(scores))
            
            # Aplicando média móvel para plotagem consistente com as janelas de métrica
            def moving_avg(arr, w):
                return np.array([np.mean(arr[max(0, j-w):j+1]) for j in range(len(arr))])
                
            mov_scores = moving_avg(scores, window_size)
            mov_thresholds = moving_avg(thresholds, window_size)
            mov_trends = moving_avg(trends, window_size)

            ax.plot(instances, mov_scores, color=color, alpha=0.85, linewidth=1.5, label=f'{name} (Score)', zorder=3)
            ax.plot(instances, mov_trends, color='yellow', alpha=0.8, linewidth=1.5, label=f'Tendência DSPOT', zorder=3)
            ax.plot(instances, mov_thresholds, color='red', alpha=0.8, linestyle='--', linewidth=1.5, label=f'Limiar DSPOT (zq)', zorder=4)

        added_attack_labels = set()
        for start, end, attack_idx in attack_regions:
            attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
            bg_color = self.bg_colors[attack_idx % len(self.bg_colors)]
            
            label_to_show = f'{attack_name}' if attack_name not in added_attack_labels else ""
            ax.axvspan(start, end, facecolor=bg_color, alpha=0.3, zorder=1, label=label_to_show)
            
            if label_to_show:
                added_attack_labels.add(attack_name)

        ax.set_title(f"{title} - EVT DSPOT (Média Móvel - Janela {window_size})", fontsize=14, fontweight='bold')
        ax.set_ylabel("Score de Anomalia", fontsize=14)
        ax.set_xlabel("Instâncias", fontsize=14)
        
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12, frameon=False)
        for patch in leg.get_patches():
            patch.set_edgecolor('gray')
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)

        ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
        fig.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        plt.show()

    def plot_poisoning_evolution(self, results_fixed, results_dspot, title, fixed_threshold=0.5, window_size=5):
        fig, axes = plt.subplots(len(results_fixed), 2, figsize=(20, 3 * len(results_fixed)))
        if len(results_fixed) == 1:
            axes = np.array([axes])

        fig.suptitle(title, fontsize=16, fontweight='bold')

        def moving_avg(arr, w):
            return np.array([np.mean(arr[max(0, j-w):j+1]) for j in range(len(arr))])

        for i, (ds_name, data_fixed) in enumerate(results_fixed.items()):
            data_dspot = results_dspot[ds_name]
            color = self.colors[i % len(self.colors)]

            ax_fixo = axes[i, 0]
            scores_f = np.array(data_fixed['scores'])
            instances_f = np.arange(len(scores_f))
            
            ax_fixo.plot(instances_f, moving_avg(scores_f, window_size), color=color, alpha=0.7, linewidth=1.5, label=f'Score (Cenário {ds_name})', zorder=3)
            ax_fixo.axhline(y=fixed_threshold, color='black', linestyle='--', linewidth=2, alpha=0.8, label=f'Threshold Fixo ({fixed_threshold})', zorder=4)

            added_labels_f = set()
            for start, end, attack_idx in data_fixed['attack_regions']:
                attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
                bg_color = self.bg_colors[attack_idx % len(self.bg_colors)]
                label_to_show = f'{attack_name}' if attack_name not in added_labels_f else ""
                ax_fixo.axvspan(start, end, facecolor=bg_color, alpha=0.3, zorder=1, label=label_to_show)
                if label_to_show: added_labels_f.add(attack_name)

            ax_fixo.set_ylabel("Scores", fontsize=14)
            ax_fixo.grid(True, alpha=0.3, linestyle=':', zorder=0)

            ax_dspot = axes[i, 1]
            scores_d = np.array(data_dspot['scores'])
            instances_d = np.arange(len(scores_d))
            
            ax_dspot.plot(instances_d, moving_avg(scores_d, window_size), color=color, alpha=0.7, linewidth=2.0, label=f'Score (Cenário {ds_name})', zorder=3)
            
            thresholds_d = np.array(data_dspot['thresholds'])
            trends_d = np.array(data_dspot['trends'])
            
            ax_dspot.plot(instances_d, moving_avg(trends_d, window_size), color='darkgoldenrod', alpha=0.9, linewidth=1.5, label='Tendência DSPOT', zorder=3)
            ax_dspot.plot(instances_d, moving_avg(thresholds_d, window_size), color='black', alpha=0.8, linestyle='--', linewidth=1.5, label='Limiar DSPOT', zorder=4)

            added_labels_d = set()
            for start, end, attack_idx in data_dspot['attack_regions']:
                attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
                bg_color = self.bg_colors[attack_idx % len(self.bg_colors)]
                label_to_show = f'{attack_name}' if attack_name not in added_labels_d else ""
                ax_dspot.axvspan(start, end, facecolor=bg_color, alpha=0.3, zorder=1, label=label_to_show)
                if label_to_show: added_labels_d.add(attack_name)

            ax_dspot.grid(True, alpha=0.3, linestyle=':', zorder=0)

            if i == 0:
                ax_fixo.set_title("Threshold Fixo", fontsize=14, fontweight='bold')
                ax_dspot.set_title("DSPOT", fontsize=14, fontweight='bold')

        axes[-1, 0].set_xlabel("Instâncias", fontsize=16)
        axes[-1, 1].set_xlabel("Instâncias", fontsize=16)

        handles, labels = [], []
        for ax in axes.flatten():
            h, l = ax.get_legend_handles_labels()
            for idx, label in enumerate(l):
                if label not in labels:
                    labels.append(label)
                    handles.append(h[idx])
        
        leg = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=5, fontsize=16, frameon=False)
        for patch in leg.get_patches():
            if hasattr(patch, 'set_edgecolor'):
                patch.set_edgecolor('gray')
                patch.set_linewidth(1.0)
                patch.set_alpha(0.8)

        plt.tight_layout()
        fig.subplots_adjust(top=0.94, bottom=0.12)
        plt.show()
    