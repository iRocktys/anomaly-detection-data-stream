from turtle import color
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from capymoa.evaluation import ClassificationEvaluator

class AnomalyExperimentRunner:
    def __init__(self):
        pass

    def _get_metric_classifier(self, metrics_dict, metric_name, target_class=1):
        if target_class is None:
            val = metrics_dict.get(metric_name)
            if val is None or np.isnan(val):
                val = metrics_dict.get(f'macro_{metric_name}', 0.0)
        else:
            val = metrics_dict.get(f'{metric_name}_{target_class}')
            if val is None or np.isnan(val):
                val = metrics_dict.get(metric_name, 0.0)
                
        return float(val) if val is not None and not np.isnan(val) else 0.0
    
    def plot_score(self, results, attack_regions, title):
        fig, ax = plt.subplots(figsize=(15, 6)) 
        
        # Paleta de cores para diferenciar os algoritmos
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            scores = np.array(data['scores'])
            instances = np.arange(len(scores))
            window_size = 50
            
            # Recalcula a média móvel
            moving_avg = np.array([np.mean(scores[max(0, j-window_size):j+1]) for j in range(len(scores))])
            
            # Plota apenas a linha contínua da média móvel
            ax.plot(instances, moving_avg, color=color, alpha=0.85, linewidth=1.5, label=f'{name}', zorder=3)

        # Adiciona as regiões de ataque (apenas uma vez para a legenda não duplicar)
        added_attack_label = False
        for start, end in attack_regions:
            ax.axvspan(start, end, facecolor="#F7C5CD", alpha=0.3, zorder=1,
                       label='Região de Ataque' if not added_attack_label else "")
            added_attack_label = True

        ax.set_title(f"Análise de Scores (Média Móvel - Janela {window_size})", fontsize=14, fontweight='bold')
        ax.set_ylabel("Score de Anomalia", fontsize=14)
        ax.set_xlabel("Instâncias", fontsize=14)
        ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
        ax.set_ylim(0.0, 1.1)
        plt.tight_layout()
        plt.show()

    def plot_metrics(self, results, attack_regions, title):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'] 
        
        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            
            ax1.plot(data['instances'], data['f1_score'], label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
            ax2.plot(data['instances'], data['precision'], label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
            ax3.plot(data['instances'], data['recall'], label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)

        for ax in [ax1, ax2, ax3]:
            added_attack_label = False
            for start, end in attack_regions:
                ax.axvspan(start, end, facecolor="#F7C5CD", alpha=0.4, zorder=2,
                            label='Região de Ataque' if not added_attack_label else "")
                added_attack_label = True
                
            ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax1.set_ylabel("F1-Score", fontsize=14)
        ax2.set_ylabel("Precision", fontsize=14)
        ax3.set_ylabel("Recall", fontsize=14)
        ax3.set_xlabel("Instâncias", fontsize=14)
        ax1.legend(loc='best', fontsize=14, frameon=True, framealpha=0.9)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def display_cumulative_metrics(self, predictions_history, schema=None):
        print(f"\n{'='*65}")
        print(f"{'RESUMO DE MÉTRICAS ACUMULATIVAS':^65}")
        print(f"{'='*65}")
        print(f"{'Algoritmo':<25} | {'F1':<10} | {'Prec':<10} | {'Rec':<10}")
        print(f"{'-'*65}")

        for name, data in predictions_history.items():
            y_true_list = data['true_labels']
            y_pred_list = data['predicted_classes']
            
            # Sklearn: processamento em lote (vetorizado)
            f1 = f1_score(y_true_list, y_pred_list, zero_division=0) * 100
            prec = precision_score(y_true_list, y_pred_list, zero_division=0) * 100
            recall = recall_score(y_true_list, y_pred_list, zero_division=0) * 100

            print(f"{name:<25} | {f1:<10.2f} | {prec:<10.2f} | {recall:<10.2f}")
        
        print(f"{'='*65}\n")

    def _run_anomaly_evaluation(self, stream, algorithms, window_size, title, target_class=0):
        results_metrics = {}
        results_scores = {}
        attack_regions = []
        
        # Dicionário para armazenar o histórico completo de predições e rótulos
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
            in_attack, start_attack = False, 0

            while stream.has_more_instances():
                instance = stream.next_instance()
                true_label = instance.y_index 
                
                if alg_idx == 0:
                    is_attack = (true_label == 1)
                    if is_attack and not in_attack:
                        in_attack, start_attack = True, count
                    elif not is_attack and in_attack:
                        in_attack = False
                        attack_regions.append((start_attack, count))

                score = learner.score_instance(instance) 
                results_scores[alg_name]['scores'].append(score)
                
                predicted_class = 1 if score > 0.5 else 0
                
                # Salvando rótulos reais e predições nas listas do dicionário
                alg_true_labels.append(true_label)
                alg_predicted_classes.append(predicted_class)
                
                evaluator_class.update(true_label, predicted_class)
               
                try: 
                    if alg_name in ['AE', 'Autoencoder']:
                        if predicted_class == 0:
                            learner.train(instance)
                    else:
                        learner.train(instance)
                except ValueError: 
                    pass

                if count > 0 and count % window_size == 0:
                    class_metrics = evaluator_class.metrics_dict()
                    
                    # Continua usando _get_metric_classifier para manter a flexibilidade de target_class (aqui em 0 como no seu código base)
                    f1_val = self._get_metric_classifier(class_metrics, 'f1_score', target_class=target_class)
                    prec_val = self._get_metric_classifier(class_metrics, 'precision', target_class=target_class)
                    recall_val = self._get_metric_classifier(class_metrics, 'recall', target_class=target_class)

                    history['instances'].append(count)
                    history['f1_score'].append(f1_val)
                    history['precision'].append(prec_val)
                    history['recall'].append(recall_val)
                        
                count += 1
                
            results_metrics[alg_name] = history
            
            # Guarda as listas no dicionário usando o nome do algoritmo como chave
            predictions_history[alg_name] = {
                'true_labels': alg_true_labels,
                'predicted_classes': alg_predicted_classes
            }

        self.plot_score(results_scores, attack_regions, title)
        self.plot_metrics(results_metrics, attack_regions, title)
        self.display_cumulative_metrics(predictions_history, schema)