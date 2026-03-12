from turtle import color

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from capymoa.evaluation import AnomalyDetectionEvaluator, ClassificationEvaluator
from capymoa.drift.detectors import ADWIN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.DynamicThresholding import DynamicQuantile, EmpiricalPOT, RollingOtsu, RollingZScore
class AnomalyExperimentRunner:
    def __init__(self):
        pass

    def _get_metric_classifier(self, metrics_dict, metric_name):
        val = metrics_dict.get(f'{metric_name}_0')
        if val is None or np.isnan(val):
            val = metrics_dict.get(metric_name, 0.0)
        return float(val) if not np.isnan(val) else 0.0

    def _get_metric_anomaly(self, metrics_dict, metric_name):
        val = metrics_dict.get(metric_name)
        if val is None or np.isnan(val):
            # O MOA às vezes capitaliza a primeira letra (ex: 'Recall' em vez de 'recall')
            val = metrics_dict.get(metric_name.capitalize(), 0.0) 
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

    def display_cumulative_metrics(self, algorithms_results):
        print(f"\n{'='*65}")
        print(f"{'RESUMO DE MÉTRICAS ACUMULATIVAS (VALORES REAIS)':^65}")
        print(f"{'='*65}")
        print(f"{'Algoritmo':<25} | {'F1':<10} | {'Prec':<10} | {'Rec':<10}")
        print(f"{'-'*65}")

        for name, evaluators in algorithms_results.items():
            class_m = evaluators['evaluator_class'].metrics_dict()
            anom_m = evaluators['evaluator_anomaly'].metrics_dict()
            f1 = self._get_metric_classifier(class_m, 'f1_score')
            prec = self._get_metric_classifier(class_m, 'precision')
            recall = anom_m.get('Recall', 0.0)
            
            print(f"{name:<25} | {f1:<10.2f} | {prec:<10.2f} | {recall:<10.2f}")
        
        print(f"{'='*65}\n")

    def _run_anomaly_evaluation(self, stream, algorithms, window_size, title):
        results_metrics = {}
        results_scores = {}
        attack_regions = []
        final_evaluators = {}

        for alg_idx, (alg_name, learner) in enumerate(algorithms.items()):
            stream.restart()
            
            # Inicialização mantida conforme evaluation.py
            evaluator_anomaly = AnomalyDetectionEvaluator(schema=stream.get_schema(), window_size=window_size)
            evaluator_class = ClassificationEvaluator(schema=stream.get_schema(), window_size=window_size)
            
            history = {'instances': [], 'f1_score': [], 'precision': [], 'recall': []}
            results_scores[alg_name] = {'scores': []}
            
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
                
                # Discretização mantida em 0.5 (padrão MOA)
                predicted_class = 1 if score > 0.5 else 0
                
                evaluator_anomaly.update(true_label, score)
                evaluator_class.update(true_label, predicted_class)

                try: 
                    learner.train(instance)
                except ValueError: 
                    pass

                if count > 0 and count % window_size == 0:
                    class_metrics = evaluator_class.metrics_dict()
                    anomaly_metrics = evaluator_anomaly.metrics_dict()
                    
                    # VALORES BRUTOS: Sem multiplicação por 100
                    f1_val = self._get_metric_classifier(class_metrics, 'f1_score')
                    prec_val = self._get_metric_classifier(class_metrics, 'precision')
                    recall_val = self._get_metric_anomaly(anomaly_metrics, 'Recall')
                    
                    history['instances'].append(count)
                    history['f1_score'].append(f1_val)
                    history['precision'].append(prec_val)
                    history['recall'].append(recall_val)
                        
                count += 1
                
            results_metrics[alg_name] = history
            final_evaluators[alg_name] = {
                'evaluator_class': evaluator_class, 
                'evaluator_anomaly': evaluator_anomaly
            }

        self.plot_score(results_scores, attack_regions, title)
        self.plot_metrics(results_metrics, attack_regions, title)
        self.display_cumulative_metrics(final_evaluators)