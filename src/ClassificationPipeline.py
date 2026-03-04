import numpy as np
import matplotlib.pyplot as plt
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import ClassificationWindowedEvaluator, ClassificationEvaluator

class ClassificationExperimentRunner:
    def __init__(self):
        pass

    def _sanitize_metric(self, value):
        try:
            if value is None: return 0.0
            
            # Se for lista/array, tira a média ignorando NaNs
            if isinstance(value, (list, np.ndarray, tuple)):
                if len(value) == 0: return 0.0
                value = np.nanmean(value)
            
            val_float = float(value)
            
            if np.isnan(val_float) or np.isinf(val_float):
                return 0.0
            
            # Normalização (se vier em %, converte para 0-1)
            if val_float > 1.0:
                return val_float / 100.0
                
            return val_float
        except Exception:
            return 0.0

    def run_experiments(self, stream, models, window_size=1000, logging=True):
        results = {}
        attack_regions = []
        in_attack = False
        start_attack_idx = 0
        
        # Inicializa estruturas
        for name in models:
            results[name] = {
                'instances': [],
                'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 
                'kappa': [], 'kappa_t': [], # Adicionado Kappa e Kappa T
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

                # Coleta de Métricas 
                if (count + 1) % window_size == 0:
                    res['instances'].append(count)
                    
                    # Extração segura
                    m_dict = w_eval.metrics_dict()
                    
                    raw_acc = m_dict.get('accuracy', w_eval.accuracy())
                    raw_f1 = m_dict.get('f1_score', w_eval.f1_score())
                    raw_prec = m_dict.get('precision', w_eval.precision())
                    raw_rec = m_dict.get('recall', w_eval.recall())
                    # Coleta de Kappa e Kappa T
                    raw_kappa = m_dict.get('kappa', w_eval.kappa())
                    raw_kappa_t = m_dict.get('kappa_t', w_eval.kappa_t())

                    # Sanitização
                    acc = self._sanitize_metric(raw_acc)
                    f1 = self._sanitize_metric(raw_f1)
                    prec = self._sanitize_metric(raw_prec)
                    rec = self._sanitize_metric(raw_rec)
                    kappa = self._sanitize_metric(raw_kappa)
                    kappa_t = self._sanitize_metric(raw_kappa_t)

                    res['accuracy'].append(acc)
                    res['f1'].append(f1)
                    res['precision'].append(prec)
                    res['recall'].append(rec)
                    res['kappa'].append(kappa)
                    res['kappa_t'].append(kappa_t)
                    
                    if logging:
                        print(f"[{name}] Inst: {count} | Acc: {acc:.2%} | F1: {f1:.2%} | Kap: {kappa:.2f} | KapT: {kappa_t:.2f}")

            count += 1
        
        if in_attack:
            attack_regions.append((start_attack_idx, count))
            
        return results, attack_regions

    def print_metrics(self, results):
        print("\n" + "="*95)
        print(f"{'RELATÓRIO FINAL ACUMULATIVO':^95}")
        print("="*95)
        # Ajuste de formatação para caber as novas métricas
        print(f"{'Modelo':<25} | {'Acc':<8} | {'Prec':<8} | {'Recall':<8} | {'F1':<8} | {'Kap':<8} | {'KapT':<8}")
        print("-" * 95)
        
        for name, data in results.items():
            c_eval = data['cumulative_evaluator']
            acc = self._sanitize_metric(c_eval.accuracy())
            prec = self._sanitize_metric(c_eval.precision())
            rec = self._sanitize_metric(c_eval.recall())
            f1 = self._sanitize_metric(c_eval.f1_score())
            kappa = self._sanitize_metric(c_eval.kappa())
            kappa_t = self._sanitize_metric(c_eval.kappa_t())
            
            print(f"{name:<25} | {acc:.4f}   | {prec:.4f}   | {rec:.4f}   | {f1:.4f}   | {kappa:.4f}   | {kappa_t:.4f}")
        print("="*95 + "\n")

    def plot_metrics(self, results, attack_regions=None, title="Métricas Janeladas"):
        # Aumentado para 6 subplots
        fig, axes = plt.subplots(6, 1, figsize=(14, 20), sharex=True)
        (ax1, ax2, ax3, ax4, ax5, ax6) = axes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            x_axis = data['instances']
            
            def clean(d_list): return [0.0 if (v is None or np.isnan(v)) else v for v in d_list]

            # Linhas de métricas
            ax1.plot(x_axis, clean(data['accuracy']), label=name, color=color, linewidth=2, zorder=3)
            ax2.plot(x_axis, clean(data['precision']), label=name, color=color, linewidth=2, zorder=3)
            ax3.plot(x_axis, clean(data['recall']), label=name, color=color, linewidth=2, zorder=3)
            ax4.plot(x_axis, clean(data['f1']), label=name, color=color, linewidth=2, zorder=3)
            ax5.plot(x_axis, clean(data['kappa']), label=name, color=color, linewidth=2, zorder=3)
            ax6.plot(x_axis, clean(data['kappa_t']), label=name, color=color, linewidth=2, zorder=3)

            # Drifts
            for drift_pos in data['drifts']:
                style = {'color': 'black', 'linestyle': '--', 'alpha': 0.6, 'linewidth': 1.5, 'zorder': 2}
                for ax in axes: ax.axvline(x=drift_pos, **style)

        # Fundo Rosa e Configurações
        for ax in axes:
            if attack_regions:
                added_lbl = False
                for start, end in attack_regions:
                    lbl = "Ataque" if not added_lbl else ""
                    ax.axvspan(start, end, color='#ffb6c1', alpha=0.4, label=lbl, zorder=0)
                    added_lbl = True
            
            if ax in [ax1, ax2, ax3, ax4]:
                ax.set_ylim(-0.05, 1.05) 
                ax.grid(True, alpha=0.3, zorder=1)
        

        ax1.set_ylabel("Acurácia", fontsize=12)
        ax1.legend(loc='lower right')
        ax2.set_ylabel("Precision", fontsize=12)
        ax3.set_ylabel("Recall", fontsize=12)
        ax4.set_ylabel("F1-Score", fontsize=12)
        ax5.set_ylabel("Kappa", fontsize=12)
        ax6.set_ylabel("Kappa T", fontsize=12)
        ax6.set_xlabel("Instâncias", fontsize=14)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()

    def plot_precision_recall(self, results):
        plt.figure(figsize=(10, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            prec = data['precision']
            rec = data['recall']
            
            valid = [(p, r) for p, r in zip(prec, rec) if (p > 0 or r > 0)]
            if not valid: continue
            p_c, r_c = zip(*valid)

            plt.plot(r_c, p_c, color=color, alpha=0.6, label=name)
            plt.scatter(r_c[0], p_c[0], color=color, marker='o', alpha=0.6)
            plt.scatter(r_c[-1], p_c[-1], color=color, marker='*', s=120, edgecolors='k', zorder=10)

        plt.title("Precision x Recall", fontsize=14, fontweight='bold')
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def pipeline(self, stream, models, window_size=500, logging=True, title="Análise de Classificação"):
        results, attacks = self.run_experiments(stream, models, window_size, logging)
        
        # Relatório
        self.print_metrics(results)
        
        # Gráficos
        self.plot_metrics(results, attacks, title=title)
        self.plot_precision_recall(results)
        
        return results