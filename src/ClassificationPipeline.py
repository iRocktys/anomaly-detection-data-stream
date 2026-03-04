import numpy as np
import matplotlib.pyplot as plt
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import ClassificationWindowedEvaluator, ClassificationEvaluator

class ClassificationExperimentRunner:
    def __init__(self):
        pass

    def _get_metric_class_0(self, m_dict, metric_name, fallback_val=0.0):
        """Busca a métrica focada na Classe 0 (Normal) e já converte para 0-100"""
        val = m_dict.get(f'{metric_name}_0')
        if val is None or np.isnan(val):
            val = m_dict.get(metric_name, fallback_val)
        
        if val is None or np.isnan(val) or np.isinf(val):
            return 0.0
            
        return float(val)

    def print_metrics(self, results):
        print("\n" + "="*80)
        print(f"{'RELATÓRIO ACUMULATIVO ':^80}")
        print("="*80)
        print(f"{'Modelo':<25} | {'Prec (%)':<10} | {'Rec (%)':<10} | {'F1 (%)':<10}")
        print("-" * 80)
        
        for name, data in results.items():
            c_eval = data['cumulative_evaluator']
            m_dict = c_eval.metrics_dict()
            
            prec = self._get_metric_class_0(m_dict, 'precision')
            rec = self._get_metric_class_0(m_dict, 'recall')
            f1 = self._get_metric_class_0(m_dict, 'f1_score')
            
            print(f"{name:<25} | {prec:>8.2f}   | {rec:>8.2f}   | {f1:>8.2f}")
        print("="*80 + "\n")

    def plot(self, results, attack_regions=None, title="Métricas Janeladas (Foco: Tráfego Normal)"):
        # Ajustado para 3 subplots exatos
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        (ax1, ax2, ax3) = axes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            x_axis = data['instances']
            
            def clean(d_list): return [0.0 if (v is None or np.isnan(v)) else v for v in d_list]

            # Linhas de métricas limitadas às 3 solicitadas
            ax1.plot(x_axis, clean(data['precision']), label=name, color=color, linewidth=2, zorder=3)
            ax2.plot(x_axis, clean(data['recall']), label=name, color=color, linewidth=2, zorder=3)
            ax3.plot(x_axis, clean(data['f1']), label=name, color=color, linewidth=2, zorder=3)

            # Drifts
            for drift_pos in data['drifts']:
                style = {'color': 'black', 'linestyle': '--', 'alpha': 0.6, 'linewidth': 1.5, 'zorder': 2}
                for ax in axes: ax.axvline(x=drift_pos, **style)

        # Fundo Rosa e Configurações de Eixos
        for ax in axes:
            if attack_regions:
                added_lbl = False
                for start, end in attack_regions:
                    lbl = "Ataque" if not added_lbl else ""
                    ax.axvspan(start, end, color='#ffb6c1', alpha=0.4, label=lbl, zorder=0)
                    added_lbl = True
            
            ax.grid(True, alpha=0.3, zorder=1)
            # Trava os eixos perfeitamente entre 0 e 100
            # ax.set_ylim(-3, 103) 

        ax1.set_ylabel("Precision (%)", fontsize=12)
        ax3.legend(loc='lower right', fontsize=14)
        ax2.set_ylabel("Recall (%)", fontsize=12)
        ax3.set_ylabel("F1-Score (%)", fontsize=12)
        ax3.set_xlabel("Instâncias", fontsize=14)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()

    def run_experiments(self, stream, models, window_size=1000, logging=True):
        results = {}
        attack_regions = []
        in_attack = False
        start_attack_idx = 0
        
        # Inicializa estruturas apenas com as 3 métricas
        for name in models:
            results[name] = {
                'instances': [],
                'f1': [], 'precision': [], 'recall': [], 
                'drifts': [],
                'windowed_evaluator': ClassificationWindowedEvaluator(schema=stream.get_schema(), window_size=window_size),
                'cumulative_evaluator': ClassificationEvaluator(schema=stream.get_schema()),
                'adwin': ADWIN()
            }

        count = 0
        stream.restart()

        while stream.has_more_instances():
            instance = stream.next_instance()
            
            # Detecção de Regiões de Ataque
            is_attack = (instance.y_index >= 1)
            if is_attack and not in_attack:
                in_attack = True
                start_attack_idx = count
            elif not is_attack and in_attack:
                in_attack = False
                attack_regions.append((start_attack_idx, count))

            for name, model in models.items():
                res = results[name]
                w_eval = res['windowed_evaluator']
                c_eval = res['cumulative_evaluator']

                # Predição
                prediction = model.predict(instance)
                w_eval.update(instance.y_index, prediction)
                c_eval.update(instance.y_index, prediction)

                # Drift
                error = 0.0 if prediction == instance.y_index else 1.0
                res['adwin'].add_element(error)
                if res['adwin'].detected_change():
                    res['drifts'].append(count)

                # Treino
                model.train(instance)

                # Coleta de Métricas Janeladas
                if (count + 1) % window_size == 0:
                    res['instances'].append(count)
                    
                    m_dict = w_eval.metrics_dict()
                    
                    # Extração direta e focada na Classe 0 (Normal)
                    res['precision'].append(self._get_metric_class_0(m_dict, 'precision'))
                    res['recall'].append(self._get_metric_class_0(m_dict, 'recall'))
                    res['f1'].append(self._get_metric_class_0(m_dict, 'f1_score'))
                    
                    if logging:
                        print(f"[{name}] Inst: {count} | Prec: {res['precision'][-1]:>6.2f}% | Rec: {res['recall'][-1]:>6.2f}% | F1: {res['f1'][-1]:>6.2f}%")

            count += 1
        
        if in_attack:
            attack_regions.append((start_attack_idx, count))
            
        return results, attack_regions

    