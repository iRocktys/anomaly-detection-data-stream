import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from capymoa.evaluation import AnomalyDetectionEvaluator, ClassificationEvaluator
from capymoa.drift.detectors import ADWIN

class AnomalyExperimentRunner:
    def __init__(self):
        pass

    def adaptation(self, window_scores, beta=0.3, anomaly_threshold=0.6):
        if not window_scores: return False, 0.0
        anomalies = [s for s in window_scores if s > anomaly_threshold]
        anomaly_rate = len(anomalies) / len(window_scores)
        triggered = anomaly_rate > beta
        return triggered, anomaly_rate

    def plot(self, results, attack_regions, title):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] 
        
        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            for d_idx, drift_pos in enumerate(data['drifts']):
                ax1.axvline(x=drift_pos, color='black', alpha=0.8, linestyle='-', linewidth=1, zorder=1)
                ax2.axvline(x=drift_pos, color='black', alpha=0.8, linestyle='-', linewidth=1, zorder=1)
                ax3.axvline(x=drift_pos, color='black', alpha=0.8, linestyle='-', linewidth=1, zorder=1)
                
                if i == 0 and d_idx == 0:
                    ax1.plot([], [], color='black', label='Drift Detectado')
                    ax2.plot([], [], color='black', label='Drift Detectado')

            ax1.plot(data['instances'], data['accuracy'], label=f'accuracy {name}', color=color, linewidth=2.5, zorder=3)
            ax2.plot(data['instances'], data['precision'], label=f'precision {name}', color=color, linewidth=2.5, zorder=3)
            ax3.plot(data['instances'], data['Recall'], label=f'Recall {name}', color=color, linewidth=2.5, zorder=3)

        for ax in [ax1, ax2, ax3]:
            added_attack_label = False
            for start, end in attack_regions:
                ax.axvspan(start, end, facecolor="#F7C5CD", alpha=0.4, zorder=2,
                           label='Região de Ataque' if not added_attack_label else "")
                added_attack_label = True
                
            ax.legend(loc='best', fontsize=10, frameon=True, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax1.set_ylabel("F1-Score", fontsize=14)
        ax2.set_ylabel("precision", fontsize=14)
        ax3.set_ylabel("Recall", fontsize=14)
        ax2.set_xlabel("Instâncias", fontsize=14)
        # ax1.set_ylim(0.0, 1.05) 
        # ax2.set_ylim(0.0, 1.05) 
        # ax3.set_ylim(0.0, 1.05) 
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

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
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
        ax.set_ylim(0.0, 1.1)
        plt.tight_layout()
        plt.show()

    def _run_standard_execution(self, stream, algorithms, window_size, anomaly_threshold, title):
        results = {} 
        attack_regions = []

        for i, (alg_name, learner) in enumerate(algorithms.items()):
            stream.restart()
            
            evaluator = AnomalyDetectionEvaluator(schema=stream.get_schema(), window_size=window_size)
            classification = ClassificationEvaluator(schema=stream.get_schema(), window_size=window_size)
            drift_detector = ADWIN()
            
            history = {'instances': [], 'accuracy': [], 'precision': [], 'scores': [], 'Recall': [], 'drifts': []}
            count = 0
            in_attack, start_attack = False, 0

            while stream.has_more_instances():
                instance = stream.next_instance()
                
                if i == 0:
                    is_attack = (instance.y_index == 1)
                    if is_attack and not in_attack:
                        in_attack, start_attack = True, count
                    elif not is_attack and in_attack:
                        in_attack = False
                        attack_regions.append((start_attack, count))

                score = learner.score_instance(instance) 
                history['scores'].append(score)
                # predicted_class = 1 if score >= anomaly_threshold else 0

                drift_detector.add_element(score)
                if drift_detector.detected_change():
                    history['drifts'].append(count)

                try:
                    learner.train(instance)
                except ValueError:
                    pass

                evaluator.update(instance.y_index, score)
                metrics = evaluator.metrics_per_window()
                # print(metrics.keys())
                if count % window_size == 0:
                    history['instances'].append(count)
                    history['accuracy'].append(metrics.get('accuracy', 1.0))
                    history['precision'].append(metrics.get('precision', 1.0))
                    history['Recall'].append(metrics.get('recall', 1.0))
                        
                count += 1
                
            results[alg_name] = history
  
        # self.plot(results, attack_regions, title)
        self.plot_score(results, attack_regions, title)