from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from capymoa.evaluation import ClassificationEvaluator
from src.Anomaly.Threshold import DSPOT

class AnomalyExperimentRunner:
    def __init__(self, target_names):
        self.target_names = target_names
        self.normal_class_idx = 0
        for i, name in enumerate(target_names):
            if str(name).strip().upper() in ['BENIGN', 'NORMAL', '0']:
                self.normal_class_idx = i
                break

    def _get_metric_classifier(self, metrics_dict, metric_name, target_class=1):
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

    def _calc_sklearn_metrics(self, y_true, y_pred, target_class):
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
    
    def plot_score(self, results, attack_regions, title, threshold=0.5):
        fig, ax = plt.subplots(figsize=(15, 6)) 
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        bg_colors = ['#F7C5CD', '#C5D9F7', '#C5F7C5', '#F7E6C5', '#E3C5F7', '#F7D9C5', '#C5F7E6']
        
        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            scores = np.array(data['scores'])
            instances = np.arange(len(scores))
            window_size = 50
            
            moving_avg = np.array([np.mean(scores[max(0, j-window_size):j+1]) for j in range(len(scores))])
            ax.plot(instances, moving_avg, color=color, alpha=0.85, linewidth=1.5, label=f'{name}', zorder=3)

        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Threshold ({threshold})', zorder=4)

        added_attack_labels = set()
        for start, end, attack_idx in attack_regions:
            attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
            bg_color = bg_colors[attack_idx % len(bg_colors)]
            
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
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'] 
        bg_colors = ['#F7C5CD', '#C5D9F7', '#C5F7C5', '#F7E6C5', '#E3C5F7', '#F7D9C5', '#C5F7E6']
        
        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            ax1.plot(data['instances'], data['f1_score'], label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
            ax2.plot(data['instances'], data['precision'], label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
            ax3.plot(data['instances'], data['recall'], label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)

        for ax in [ax1, ax2, ax3]:
            added_attack_labels = set()
            for start, end, attack_idx in attack_regions:
                attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
                bg_color = bg_colors[attack_idx % len(bg_colors)]
                
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

    def display_cumulative_metrics(self, predictions_history, warmup_instances=0, target_class=1):
        print(f"\n{'='*65}")
        print(f"{'RELATÓRIO ACUMULATIVO':^65}")
        print(f"{'='*65}")
        print(f"{'Algoritmo':<25} | {'F1 (%)':<10} | {'Prec (%)':<10} | {'Rec (%)':<10}")
        print(f"{'-'*65}")

        for name, data in predictions_history.items():
            y_true_list = data['true_labels'][warmup_instances:] if len(data['true_labels']) > warmup_instances else data['true_labels']
            y_pred_list = data['predicted_classes'][warmup_instances:] if len(data['predicted_classes']) > warmup_instances else data['predicted_classes']
            
            f1, prec, recall = self._calc_sklearn_metrics(y_true_list, y_pred_list, target_class)

            print(f"{name:<25} | {f1:<10.2f} | {prec:<10.2f} | {recall:<10.2f}")
        
        print(f"{'='*65}\n")

    def _run_anomaly_evaluation(self, stream, algorithms, window_size, title, warmup_instances=0, target_class=None, threshold=0.5):
        results_metrics = {}
        results_scores = {}
        attack_regions = []
        
        predictions_history = {}
        schema = stream.get_schema()

        for alg_idx, (alg_name, learner) in enumerate(algorithms.items()):
            stream.restart()
            evaluator_class = ClassificationEvaluator(schema=schema, window_size=window_size)
            
            history = {'instances': [], 'f1_score': [], 'precision': [], 'recall': []}
            results_scores[alg_name] = {'scores': []}
            
            alg_true_labels = []
            alg_predicted_classes = []
            
            count = 0
            in_attack = False
            start_attack = 0
            current_attack_label = None

            while stream.has_more_instances():
                instance = stream.next_instance()
                
                true_label_multiclass = instance.y_index 
                true_label_binary = 1 if true_label_multiclass != self.normal_class_idx else 0
                
                if alg_idx == 0:
                    is_attack = (true_label_binary == 1)
                    
                    if is_attack:
                        if not in_attack:
                            in_attack = True
                            start_attack = count
                            current_attack_label = true_label_multiclass
                        elif current_attack_label != true_label_multiclass:
                            attack_regions.append((start_attack, count, current_attack_label))
                            start_attack = count
                            current_attack_label = true_label_multiclass
                    else:
                        if in_attack:
                            in_attack = False
                            attack_regions.append((start_attack, count, current_attack_label))

                score = learner.score_instance(instance) 
                results_scores[alg_name]['scores'].append(score)
                
                predicted_class = 1 if score > threshold else 0
                
                alg_true_labels.append(true_label_binary)
                alg_predicted_classes.append(predicted_class)
                
                if count >= warmup_instances:
                    evaluator_class.update(true_label_binary, predicted_class)
               
                try:
                    learner.train(instance)
                except ValueError:
                    pass

                if count >= warmup_instances and count > 0 and count % window_size == 0:
                    class_metrics = evaluator_class.metrics_dict()
                    
                    f1_val = self._get_metric_classifier(class_metrics, 'f1_score', target_class=target_class)
                    prec_val = self._get_metric_classifier(class_metrics, 'precision', target_class=target_class)
                    recall_val = self._get_metric_classifier(class_metrics, 'recall', target_class=target_class)

                    history['instances'].append(count)
                    history['f1_score'].append(f1_val)
                    history['precision'].append(prec_val)
                    history['recall'].append(recall_val)
                        
                count += 1
                
            if alg_idx == 0 and in_attack:
                attack_regions.append((start_attack, count, current_attack_label))
                
            results_metrics[alg_name] = history
            
            predictions_history[alg_name] = {
                'true_labels': alg_true_labels,
                'predicted_classes': alg_predicted_classes
            }

        self.display_cumulative_metrics(predictions_history, warmup_instances=warmup_instances, target_class=target_class)
        self.plot_score(results_scores, attack_regions, title, threshold)
        self.plot_metrics(results_metrics, attack_regions, title, window_size)

    def plot_dspot_score(self, results, attack_regions, title):
        fig, ax = plt.subplots(figsize=(15, 6)) 
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        bg_colors = ['#F7C5CD', '#C5D9F7', '#C5F7C5', '#F7E6C5', '#E3C5F7', '#F7D9C5', '#C5F7E6']
        
        window_size = 50
        
        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
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
            ax.plot(instances, mov_trends, color='black', alpha=0.5, linestyle='-.', linewidth=1.5, label=f'Tendência DSPOT', zorder=3)
            ax.plot(instances, mov_thresholds, color='red', alpha=0.8, linestyle='--', linewidth=2, label=f'Limiar DSPOT (zq)', zorder=4)

        added_attack_labels = set()
        for start, end, attack_idx in attack_regions:
            attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
            bg_color = bg_colors[attack_idx % len(bg_colors)]
            
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

    def _run_anomaly_DSPOT(self, stream, algorithms, window_size, title, warmup_instances=0, target_class=None, dspot_q=1e-3, dspot_depth=50):
        results_metrics = {}
        results_scores = {}
        attack_regions = []
        
        predictions_history = {}
        schema = stream.get_schema()

        for alg_idx, (alg_name, learner) in enumerate(algorithms.items()):
            stream.restart()
            evaluator_class = ClassificationEvaluator(schema=schema, window_size=window_size)
            dspot = DSPOT(q=dspot_q, depth=dspot_depth)
            
            history = {'instances': [], 'f1_score': [], 'precision': [], 'recall': []}
            results_scores[alg_name] = {'scores': [], 'thresholds': [], 'trends': []}
            
            alg_true_labels = []
            alg_predicted_classes = []
            
            count = 0
            in_attack = False
            start_attack = 0
            current_attack_label = None

            while stream.has_more_instances():
                instance = stream.next_instance()
                
                true_label_multiclass = instance.y_index 
                true_label_binary = 1 if true_label_multiclass != self.normal_class_idx else 0
                
                if alg_idx == 0:
                    is_attack = (true_label_binary == 1)
                    
                    if is_attack:
                        if not in_attack:
                            in_attack = True
                            start_attack = count
                            current_attack_label = true_label_multiclass
                        elif current_attack_label != true_label_multiclass:
                            attack_regions.append((start_attack, count, current_attack_label))
                            start_attack = count
                            current_attack_label = true_label_multiclass
                    else:
                        if in_attack:
                            in_attack = False
                            attack_regions.append((start_attack, count, current_attack_label))

                score = learner.score_instance(instance) 
                
                # DSPOT processando o score e atualizando o threshold em streaming
                predicted_class, dyn_thresh, local_trend = dspot.update_and_predict(score, warmup_instances)
                
                results_scores[alg_name]['scores'].append(score)
                results_scores[alg_name]['thresholds'].append(dyn_thresh)
                results_scores[alg_name]['trends'].append(local_trend)
                
                alg_true_labels.append(true_label_binary)
                alg_predicted_classes.append(predicted_class)
                
                if count >= warmup_instances:
                    evaluator_class.update(true_label_binary, predicted_class)
               
                try:
                    learner.train(instance)
                except ValueError:
                    pass

                if count >= warmup_instances and count > 0 and count % window_size == 0:
                    class_metrics = evaluator_class.metrics_dict()
                    
                    f1_val = self._get_metric_classifier(class_metrics, 'f1_score', target_class=target_class)
                    prec_val = self._get_metric_classifier(class_metrics, 'precision', target_class=target_class)
                    recall_val = self._get_metric_classifier(class_metrics, 'recall', target_class=target_class)

                    history['instances'].append(count)
                    history['f1_score'].append(f1_val)
                    history['precision'].append(prec_val)
                    history['recall'].append(recall_val)
                        
                count += 1
                
            if alg_idx == 0 and in_attack:
                attack_regions.append((start_attack, count, current_attack_label))
                
            results_metrics[alg_name] = history
            
            predictions_history[alg_name] = {
                'true_labels': alg_true_labels,
                'predicted_classes': alg_predicted_classes
            }

        self.display_cumulative_metrics(predictions_history, warmup_instances=warmup_instances, target_class=target_class)
        self.plot_dspot_score(results_scores, attack_regions, title)
        self.plot_metrics(results_metrics, attack_regions, title, window_size)