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
                
            ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax1.set_ylabel("F1-Score", fontsize=14)
        ax2.set_ylabel("precision", fontsize=14)
        ax3.set_ylabel("Recall", fontsize=14)
        ax2.set_xlabel("Instâncias", fontsize=14)
        ax3.legend(loc='best', fontsize=14, frameon=True, framealpha=0.9)
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
        ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
        ax.set_ylim(0.0, 1.1)
        plt.tight_layout()
        plt.show()

    def _get_metric_classifier(self, metrics_dict, metric_name):
        """Busca métrica do ClassificationEvaluator (focada na Classe 0)"""
        val = metrics_dict.get(f'{metric_name}_0')
        if val is None or np.isnan(val):
            val = metrics_dict.get(metric_name, 0.0)
        return float(val) if not np.isnan(val) else 0.0

    def _get_metric_anomaly(self, metrics_dict, metric_name):
        """Busca métrica do AnomalyDetectionEvaluator"""
        val = metrics_dict.get(metric_name)
        if val is None or np.isnan(val):
            # O MOA às vezes capitaliza a primeira letra (ex: 'Recall' em vez de 'recall')
            val = metrics_dict.get(metric_name.capitalize(), 0.0) 
        return float(val) if val is not None and not np.isnan(val) else 0.0

    def plot_metrics_per_technique(self, results_metrics, attack_regions, base_title):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for tech_name, alg_data in results_metrics.items():
            fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
            metric_keys = ['auc', 'recall', 'f1_score', 'precision']
            metric_titles = ["AUC (Anomaly)", "Recall (Anomaly)", "F1-Score (Classifier)", "Precision (Classifier)"]
            
            for i, (alg_name, data) in enumerate(alg_data.items()):
                color = colors[i % len(colors)]
                instances = data['instances']
                
                for ax_idx, m_key in enumerate(metric_keys):
                    # Multiplica por 100 SOMENTE o AUC e o Recall
                    if m_key in ['auc', 'recall']:
                        plot_data = np.array(data[m_key]) * 100
                    else:
                        plot_data = np.array(data[m_key])
                        
                    axes[ax_idx].plot(instances, plot_data, label=alg_name, color=color, linewidth=2)

            for ax_idx, ax in enumerate(axes):
                added_attack = False
                for start, end in attack_regions:
                    ax.axvspan(start, end, facecolor="#F7C5CD", alpha=0.3, zorder=0, label='Ataque' if not added_attack else "")
                    added_attack = True
                
                ax.set_ylabel(metric_titles[ax_idx], fontsize=11)
                ax.grid(True, alpha=0.3, linestyle=':')
                
                # Trava o eixo Y perfeitamente entre 0 e 100 (com uma pequena margem visual)
                # ax.set_ylim(-2, 102)
            
            axes[-1].legend(loc='best', fontsize=14)
            axes[-1].set_xlabel("Instâncias da Stream", fontsize=12)
            plt.suptitle(f"Técnica de Discretização: {tech_name}", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()

    def _run_standard_execution(self, stream, algorithms, window_size, anomaly_threshold, title):
        technique_names = ['Z-Score', 'Quantile', 'Otsu', 'POT']
        results_metrics = {tech: {} for tech in technique_names}
        results_scores = {}
        attack_regions = []

        for alg_idx, (alg_name, learner) in enumerate(algorithms.items()):
            stream.restart()
            
            # Instancia as técnicas de thresholding
            thresholding_methods = {
                'Z-Score': RollingZScore(window_size=window_size),
                'Quantile': DynamicQuantile(window_size=window_size),
                'Otsu': RollingOtsu(window_size=window_size),
                'POT': EmpiricalPOT(window_size=window_size)
            }
            
            # UM avaliador de Anomalia (para AUC e Recall do score contínuo)
            evaluator_anomaly = AnomalyDetectionEvaluator(schema=stream.get_schema(), window_size=window_size)
            
            # QUATRO avaliadores de Classificação (para F1 e Precision de cada técnica)
            evaluators_classifier = {
                tech: ClassificationEvaluator(schema=stream.get_schema(), window_size=window_size)
                for tech in technique_names
            }
            
            results_scores[alg_name] = {'scores': [], 'drifts': []}
            for tech in technique_names:
                results_metrics[tech][alg_name] = {'instances': [], 'auc': [], 'recall': [], 'f1_score': [], 'precision': []}
                
            count = 0
            in_attack, start_attack = False, 0
            drift_detector = ADWIN()

            y_true_stream = []
            y_pred_stream = {tech: [] for tech in technique_names}

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

                # SCORE BRUTO (Sem inversão)
                score = learner.score_instance(instance) 
                results_scores[alg_name]['scores'].append(score)

                # Atualiza o avaliador de Anomalia com o score contínuo
                evaluator_anomaly.update(true_label, score)

                drift_detector.add_element(score)
                if drift_detector.detected_change():
                    results_scores[alg_name]['drifts'].append(count)

                try: learner.train(instance)
                except ValueError: pass

                # Guarda o rótulo real na lista GLOBAL para o Sklearn
                y_true_stream.append(true_label)

                # Transforma o score e avalia cada discretizador no ClassificationEvaluator
                for tech_name, method in thresholding_methods.items():
                    discrete_prediction = method.update_and_predict(score)
                    y_pred_stream[tech_name].append(discrete_prediction)
                    evaluators_classifier[tech_name].update(true_label, discrete_prediction)

                # FIM DA JANELA: Extração para os gráficos
                if count > 0 and count % window_size == 0:
                    # Pega as métricas do AnomalyDetectionEvaluator (comuns a todas as técnicas para este modelo)
                    anomaly_metrics = evaluator_anomaly.metrics_dict()
                    auc_val = self._get_metric_anomaly(anomaly_metrics, 'auc')
                    recall_val = self._get_metric_anomaly(anomaly_metrics, 'recall')

                    # Pega as métricas do ClassificationEvaluator (específicas por técnica)
                    for tech_name in technique_names:
                        classifier_metrics = evaluators_classifier[tech_name].metrics_dict()
                        f1_val = self._get_metric_classifier(classifier_metrics, 'f1_score')
                        prec_val = self._get_metric_classifier(classifier_metrics, 'precision')
                        
                        history = results_metrics[tech_name][alg_name]
                        history['instances'].append(count)
                        history['auc'].append(auc_val)
                        history['recall'].append(recall_val)
                        history['f1_score'].append(f1_val)
                        history['precision'].append(prec_val)
                        
                count += 1
            
            print("-" * 70)
            print(f"Desempenho Global das Técnicas de Discretização para {alg_name}:")
            for tech_name in technique_names:
                y_t = y_true_stream
                y_p = y_pred_stream[tech_name]
                
                acc_sk = accuracy_score(y_t, y_p)
                prec_sk = precision_score(y_t, y_p, zero_division=0)
                rec_sk = recall_score(y_t, y_p, zero_division=0)
                f1_sk = f1_score(y_t, y_p, zero_division=0)
                
                print(f" -> {tech_name:>10}: F1-Score={f1_sk:.4f} | Precision={prec_sk:.4f} | Recall={rec_sk:.4f} | Acurácia={acc_sk:.4f}")
            print("-" * 70)
                
        # Gera os gráficos no final de toda a execução
        self.plot_score(results_scores, attack_regions, title)
        self.plot_metrics_per_technique(results_metrics, attack_regions, title)