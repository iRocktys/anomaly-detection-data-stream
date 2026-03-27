import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

class Metrics:
    def get_metric_classifier(self, metrics_dict, metric_name, target_class=None):
        norm_metrics = {str(k).lower(): v for k, v in metrics_dict.items()}
        metric_name = str(metric_name).lower()
        
        if target_class is None:
            tc = 0
        elif str(target_class).lower() == 'macro':
            tc = 'macro'
        else:
            tc = int(target_class)

        if tc == 'macro':
            val_0 = norm_metrics.get(f'{metric_name}_0', 0.0)
            val_1 = norm_metrics.get(f'{metric_name}_1', 0.0)
            val_0 = 0.0 if val_0 is None or np.isnan(val_0) else float(val_0)
            val_1 = 0.0 if val_1 is None or np.isnan(val_1) else float(val_1)
            return ((val_0 + val_1) / 2.0) * 100.0
        else:
            val = norm_metrics.get(f'{metric_name}_{tc}')
            return (float(val) * 100.0) if val is not None and not np.isnan(val) else 0.0

    def calc_sklearn_metrics(self, y_true, y_pred, target_class=None):
        if len(y_true) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        if target_class is None or str(target_class).lower() == 'macro':
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            else:
                fpr = 0.0
                tpr = 0.0
        else:
            tc = int(target_class)
            f1 = f1_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=0)
            prec = precision_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=0)
            rec = recall_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                if tc == 0:
                    fpr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                    tpr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            else:
                fpr = 0.0
                tpr = 0.0
            
        return f1 * 100.0, prec * 100.0, rec * 100.0, mcc, fpr * 100.0, tpr * 100.0

    def display_cumulative_metrics(self, predictions_history, warmup_instances=0, target_class=None, attack_regions=None):
        if attack_regions is None:
            attack_regions = []
            
        num_attacks = len(attack_regions)
        header_base = f"{'Algoritmo':<22} | {'F1 (%)':<8} | {'Prec (%)':<8} | {'Rec (%)':<8} | {'MCC':<8} | {'Tempo (s)':<10}"
        header_dyn = ""
        for i in range(num_attacks):
            header_dyn += f" | {'Drp'+str(i+1)+' %':<8} | {'Drp'+str(i+1)+' n':<8} | {'Rec'+str(i+1)+' %':<8}"
            
        full_header = header_base + header_dyn
        line_len = len(full_header)
        
        if target_class is None:
            tgt_str = "HIBRIDA (Macro Global, Queda Cls 0 - Janela 2000)"
        elif str(target_class).lower() == 'macro':
            tgt_str = "MACRO TOTAL (Janela 2000)"
        else:
            tgt_str = f"CLASSE {target_class} (Janela 2000)"

        print(f"\n{'='*line_len}")
        print(f"{f'RELATÓRIO COMPORTAMENTAL - {tgt_str}':^{line_len}}")
        print(f"{'='*line_len}")
        print(full_header)
        print(f"{'-'*line_len}")

        if target_class is None:
            timeline_class = 0
        elif str(target_class).lower() == 'macro':
            timeline_class = 'macro'
        else:
            timeline_class = int(target_class)

        window_size = 2000

        for name, data in predictions_history.items():
            y_true_full = np.array(data['true_labels'])
            y_pred_full = np.array(data['predicted_classes'])
            
            y_true_list = y_true_full[warmup_instances:] if len(y_true_full) > warmup_instances else y_true_full
            y_pred_list = y_pred_full[warmup_instances:] if len(y_pred_full) > warmup_instances else y_pred_full
            
            f1, prec, recall, mcc, fpr, tpr = self.calc_sklearn_metrics(y_true_list, y_pred_list, target_class)
            exec_time = data.get('exec_time', 0.0)

            if timeline_class == 'macro':
                is_tp0 = ((y_true_full == 0) & (y_pred_full == 0)).astype(float)
                is_fp0 = ((y_true_full != 0) & (y_pred_full == 0)).astype(float)
                is_fn0 = ((y_true_full == 0) & (y_pred_full != 0)).astype(float)
                
                tp0 = np.convolve(is_tp0, np.ones(window_size), mode='full')[:len(y_true_full)]
                fp0 = np.convolve(is_fp0, np.ones(window_size), mode='full')[:len(y_true_full)]
                fn0 = np.convolve(is_fn0, np.ones(window_size), mode='full')[:len(y_true_full)]
                
                den0 = 2 * tp0 + fp0 + fn0
                f1_0_timeline = np.zeros_like(tp0, dtype=float)
                mask0 = den0 > 0
                f1_0_timeline[mask0] = (2 * tp0[mask0] / den0[mask0])
                f1_0_timeline[den0 == 0] = 1.0

                is_tp1 = ((y_true_full == 1) & (y_pred_full == 1)).astype(float)
                is_fp1 = ((y_true_full != 1) & (y_pred_full == 1)).astype(float)
                is_fn1 = ((y_true_full == 1) & (y_pred_full != 1)).astype(float)
                
                tp1 = np.convolve(is_tp1, np.ones(window_size), mode='full')[:len(y_true_full)]
                fp1 = np.convolve(is_fp1, np.ones(window_size), mode='full')[:len(y_true_full)]
                fn1 = np.convolve(is_fn1, np.ones(window_size), mode='full')[:len(y_true_full)]
                
                den1 = 2 * tp1 + fp1 + fn1
                f1_1_timeline = np.zeros_like(tp1, dtype=float)
                mask1 = den1 > 0
                f1_1_timeline[mask1] = (2 * tp1[mask1] / den1[mask1])
                f1_1_timeline[den1 == 0] = 1.0

                f1_timeline = ((f1_0_timeline + f1_1_timeline) / 2.0) * 100.0
            else:
                tc = timeline_class
                is_tp = ((y_true_full == tc) & (y_pred_full == tc)).astype(float)
                is_fp = ((y_true_full != tc) & (y_pred_full == tc)).astype(float)
                is_fn = ((y_true_full == tc) & (y_pred_full != tc)).astype(float)
                
                tp = np.convolve(is_tp, np.ones(window_size), mode='full')[:len(y_true_full)]
                fp = np.convolve(is_fp, np.ones(window_size), mode='full')[:len(y_true_full)]
                fn = np.convolve(is_fn, np.ones(window_size), mode='full')[:len(y_true_full)]
                
                denominator = 2 * tp + fp + fn
                f1_timeline = np.zeros_like(tp, dtype=float)
                valid_mask = denominator > 0
                f1_timeline[valid_mask] = (2 * tp[valid_mask] / denominator[valid_mask]) * 100.0
                f1_timeline[denominator == 0] = 100.0

            drops_perc = []
            drops_inst = []
            recs_perc = []
            
            for i, (start, end, label) in enumerate(attack_regions):
                start_idx = max(start, warmup_instances)
                end_idx = max(end, warmup_instances)
                
                if start_idx >= len(f1_timeline):
                    break
                
                f1_before = f1_timeline[start_idx - 1] if start_idx > 0 else 0.0
                
                if end_idx >= start_idx:
                    window_attack = f1_timeline[start_idx:end_idx + 1]
                    min_relative_idx = np.argmin(window_attack)
                    f1_lowest = window_attack[min_relative_idx]
                    
                    drop_p = f1_lowest - f1_before
                    drop_n = min_relative_idx  
                else:
                    drop_p, drop_n, f1_lowest = 0.0, 0, f1_before
                    
                drops_perc.append(drop_p)
                drops_inst.append(drop_n)
                
                target_idx = end_idx + 2000
                if target_idx >= len(f1_timeline):
                    target_idx = len(f1_timeline) - 1
                
                if target_idx > end_idx:
                    f1_at_target = f1_timeline[target_idx]
                    rec_p = f1_at_target - f1_lowest
                else:
                    rec_p = 0.0
                    
                recs_perc.append(rec_p)

            row_base = f"{name:<22} | {f1:<8.2f} | {prec:<8.2f} | {recall:<8.2f} | {mcc:<8.3f} | {exec_time:<10.2f}"
            row_dyn = ""
            for dp, dn, rp in zip(drops_perc, drops_inst, recs_perc):
                row_dyn += f" | {dp:<8.2f} | {dn:<8} | {rp:<+8.2f}"
                
            print(row_base + row_dyn)
        
        print(f"{'='*line_len}\n")


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