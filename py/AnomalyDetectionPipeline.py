import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from capymoa.evaluation import AnomalyDetectionEvaluator
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] 
        
        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            for d_idx, drift_pos in enumerate(data['drifts']):
                ax1.axvline(x=drift_pos, color='black', alpha=0.8, linestyle='-', linewidth=1, zorder=1)
                ax2.axvline(x=drift_pos, color='black', alpha=0.8, linestyle='-', linewidth=1, zorder=1)
                
                if i == 0 and d_idx == 0:
                    ax1.plot([], [], color='black', label='Drift Detectado')
                    ax2.plot([], [], color='black', label='Drift Detectado')

            ax1.plot(data['instances'], data['auc'], label=f'AUC {name}', color=color, linewidth=2.5, zorder=3)
            ax2.plot(data['instances'], data['acc'], label=f'Acc {name}', color=color, linestyle='--', linewidth=2.5, zorder=3)

        for ax in [ax1, ax2]:
            added_attack_label = False
            for start, end in attack_regions:
                ax.axvspan(start, end, facecolor="#F7C5CD", alpha=0.4, zorder=2,
                           label='Região de Ataque' if not added_attack_label else "")
                added_attack_label = True
                
            ax.legend(loc='best', fontsize=10, frameon=True, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax1.set_ylabel("AUC", fontsize=14)
        ax2.set_ylabel("Accuracy", fontsize=14)
        ax2.set_xlabel("Instâncias", fontsize=14)
        ax1.set_ylim(0.0, 1.05) 
        ax2.set_ylim(0.0, 1.05) 
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_score(self, results, attack_regions, title):
        num_models = len(results)
        fig, axes = plt.subplots(num_models, 1, figsize=(15, 5 * num_models), sharex=True)
        if num_models == 1: axes = [axes]

        for i, (name, data) in enumerate(results.items()):
            ax = axes[i] if num_models > 1 else axes[0]
            scores = np.array(data['scores'])
            instances = np.arange(len(scores))
            window_size = 30
            moving_avg = np.array([np.mean(scores[max(0, j-window_size):j]) if j > 0 else scores[0] for j in range(len(scores))])
            
            ax.scatter(instances, scores, color="#1f77b4", s=15, alpha=0.4, edgecolors='none', label='Amostra (Score)', zorder=1)
            ax.plot(instances, moving_avg, color="#0400F7", alpha=0.6, linewidth=1.5, label='Média Móvel', zorder=2)

            added_attack_label = False
            for start, end in attack_regions:
                ax.axvspan(start, end, facecolor="#F7C5CD", alpha=0.3, zorder=0,
                           label='Região de Ataque' if not added_attack_label else "")
                added_attack_label = True

            ax.set_title(f"Análise de Scores - {name}", fontsize=12, fontweight='bold')
            ax.set_ylabel("Score de Anomalia", fontsize=14)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.15, linestyle=':')
            ax.set_ylim(0.0, 1.1)

        axes[-1].set_xlabel("Instâncias", fontsize=14)
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

    def _run_adaptive_blocking(self, stream, algorithms, warmup_windows, window_size, title, anomaly_threshold, beta, logging):
        results = {} 
        attack_regions = []
        warmup_instances = warmup_windows * window_size

        for i, (alg_name, learner) in enumerate(algorithms.items()):
            stream.restart()
            
            evaluator = AnomalyDetectionEvaluator(schema=stream.get_schema())
            
            history = {'instances': [], 'auc': [], 'acc': [], 'scores': [], 'drifts': []}
            window_instances = []
            window_scores = []
            count = 0
            in_attack, start_attack = False, 0

            while stream.has_more_instances():
                instance = stream.next_instance()
                
                # Rastreamento da região de ataque
                if i == 0:
                    is_attack = (instance.y_index == 1)
                    if is_attack and not in_attack:
                        in_attack, start_attack = True, count
                    elif not is_attack and in_attack:
                        in_attack = False
                        attack_regions.append((start_attack, count))

                score = learner.score_instance(instance) 
                history['scores'].append(score)
                window_scores.append(score)
                window_instances.append(instance)

                # Treino preliminar se score for baixo (confiável)
                if score < anomaly_threshold:
                    try: learner.train(instance)
                    except ValueError: pass

                # Lógica de Adaptação ao fim da janela
                if count > 0 and count % window_size == 0:
                    triggered, rate = self.adaptation(window_scores, beta=beta, anomaly_threshold=anomaly_threshold)
                    
                    if count <= warmup_instances:
                        for inst in window_instances:
                            try: learner.train(inst)
                            except ValueError: pass
                        status = "TREINO REALIZADO (WARMUP)"
                    elif not triggered:
                        for inst in window_instances:
                            try: learner.train(inst)
                            except ValueError: pass
                        status = "TREINO REALIZADO"
                    else:
                        history['drifts'].append(count)
                        status = "TREINO BLOQUEADO (JANELA ANÔMALA)"
                    
                    if logging:
                        print(f"[{alg_name}] Instância: {count} | {status} | Taxa de Anomalia: {rate:.2%}")
                    
                    window_scores = []
                    window_instances = []
                
                evaluator.update(instance.y_index, score)
                if count % window_size == 0:
                    metrics = evaluator.metrics_dict()
                    history['instances'].append(count)
                    history['auc'].append(evaluator.auc())
                    history['acc'].append(metrics.get('Accuracy', 0))
                        
                count += 1
                
            results[alg_name] = history

        self.plot(results, attack_regions, title)
        self.plot_score(results, attack_regions, title)

    def _run_standard_execution(self, stream, algorithms, window_size, title):
        results = {} 
        attack_regions = []

        for i, (alg_name, learner) in enumerate(algorithms.items()):
            stream.restart()
            
            evaluator = AnomalyDetectionEvaluator(schema=stream.get_schema())
            drift_detector = ADWIN()
            
            history = {'instances': [], 'auc': [], 'acc': [], 'scores': [], 'drifts': []}
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

                drift_detector.add_element(score)
                if drift_detector.detected_change():
                    history['drifts'].append(count)

                try:
                    learner.train(instance)
                except ValueError:
                    pass

                evaluator.update(instance.y_index, score)
                if count % window_size == 0:
                    metrics = evaluator.metrics_dict()
                    history['instances'].append(count)
                    history['auc'].append(evaluator.auc())
                    history['acc'].append(metrics.get('Accuracy', 0))
                        
                count += 1
                
            results[alg_name] = history

        self.plot(results, attack_regions, title)
        self.plot_score(results, attack_regions, title)

    def _run_memory_stream(self, stream, algorithms, window_size, title, K, anomaly_threshold, logging):
        results = {} 
        attack_regions = []

        for i, (alg_name, learner) in enumerate(algorithms.items()):
            stream.restart()
            
            evaluator = AnomalyDetectionEvaluator(schema=stream.get_schema())
            drift_detector = ADWIN()
            
            history = {'instances': [], 'auc': [], 'acc': [], 'scores': [], 'drifts': []}
            
            memory_buffer = deque(maxlen=K)
            total_accepted = 0
            window_accepted = 0
            
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

                drift_detector.add_element(score)
                if drift_detector.detected_change():
                    history['drifts'].append(count)

                # LÓGICA MEMSTREAM
                should_train = (len(memory_buffer) < K) or (score < anomaly_threshold)

                if should_train:
                    memory_buffer.append(instance)
                    total_accepted += 1
                    window_accepted += 1
                    try: 
                        learner.train(instance)
                    except ValueError: 
                        pass
                
                evaluator.update(instance.y_index, score)
                if count % window_size == 0:
                    metrics = evaluator.metrics_dict()
                    history['instances'].append(count)
                    history['auc'].append(evaluator.auc())
                    history['acc'].append(metrics.get('Accuracy', 0))
                    
                    if logging:
                        print(f"[{alg_name}] Instância: {count}")
                        print(f"  > Novas amostras aceitas: {window_accepted} | Histórico: {total_accepted}")
                        print(f"  > Ocupação da Fila: {len(memory_buffer)}/{K}")       
                        
                        if window_accepted > K:
                            print(f"  > AVISO: A memória (K={K}) foi totalmente renovada {window_accepted/K:.1f}x nesta janela.")
                        
                count += 1
                
            results[alg_name] = history

        self.plot(results, attack_regions, title)
        self.plot_score(results, attack_regions, title)

    def ExecuteExperiments(self, stream, pipeline_name, algorithms, 
                           warmup_windows=5, window_size=200, 
                           anomaly_threshold=0.4, beta=0.3, 
                           title="Desempenho dos Algoritmos", logging=False, K_MEM=500):

        # Mapeamento de nomes para funções
        pipelines = {
            'adaptive': lambda: self._run_adaptive_blocking(stream, algorithms, warmup_windows, window_size, title, anomaly_threshold, beta, logging),
            'standard': lambda: self._run_standard_execution(stream, algorithms, window_size, title),
            'memory': lambda: self._run_memory_stream(stream, algorithms, window_size, title, K_MEM, anomaly_threshold, logging)
        }

        if pipeline_name in pipelines:
            pipelines[pipeline_name]()
        else:
            raise ValueError(f"Pipeline '{pipeline_name}' não encontrado. Opções: {list(pipelines.keys())}")