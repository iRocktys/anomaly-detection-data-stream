import numpy as np
import matplotlib.pyplot as plt
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import ClassificationWindowedEvaluator, ClassificationEvaluator
from sklearn.metrics import f1_score, precision_score, recall_score

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
                # 'kappa': [],
                'kappa_t': [],
                'drifts': [],
                'windowed_evaluator': ClassificationWindowedEvaluator(schema=stream.get_schema(), window_size=window_size),
                'cumulative_evaluator': ClassificationEvaluator(schema=stream.get_schema()),
                'adwin': ADWIN(),
                # Buffers para sklearn
                'y_true_window': [],
                'y_pred_window': [],
                'y_true_cumulative': [],  # (para F1 acumulativo)
                'y_pred_cumulative': []   # (para F1 acumulativo)
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

                # NOVO:
                # Armazenar para sklearn
                res['y_true_window'].append(instance.y_index)
                res['y_pred_window'].append(prediction)
                res['y_true_cumulative'].append(instance.y_index)
                res['y_pred_cumulative'].append(prediction)

                # NOVO:
                # Manter apenas últimas window_size no buffer
                if len(res['y_true_window']) > window_size:
                    res['y_true_window'].pop(0)
                    res['y_pred_window'].pop(0)

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
                    
                    # raw_f1 = m_dict.get('f1_score', w_eval.f1_score())

                    # Calcular Precision, Recall e F1 com sklearn (BINARY - foco em ataques)
                    if len(res['y_true_window']) >= 2:  # Mínimo de amostras
                        raw_prec = precision_score(
                            res['y_true_window'],
                            res['y_pred_window'],
                            average='binary',
                            pos_label=1,
                            zero_division=1.0  # Se não há erros, considera perfeito
                        )
                        raw_rec = recall_score(
                            res['y_true_window'],
                            res['y_pred_window'],
                            average='binary',
                            pos_label=1,
                            zero_division=1.0  # Se não há erros, considera perfeito
                        )
                        raw_f1 = f1_score(
                            res['y_true_window'],
                            res['y_pred_window'],
                            average='binary',
                            pos_label=1,
                            zero_division=1.0  # Se não há erros, considera perfeito
                        )
                    else:
                        raw_prec = 0.0
                        raw_rec = 0.0
                        raw_f1 = 0.0
                    
                    # Coleta de Kappa T
                    # raw_kappa = m_dict.get('kappa', w_eval.kappa())
                    raw_kappa_t = m_dict.get('kappa_t', w_eval.kappa_t())

                    # Sanitização
                    acc = self._sanitize_metric(raw_acc)
                    f1 = self._sanitize_metric(raw_f1)
                    prec = self._sanitize_metric(raw_prec)
                    rec = self._sanitize_metric(raw_rec)
                    # kappa = self._sanitize_metric(raw_kappa)
                    kappa_t = self._sanitize_metric(raw_kappa_t)

                    res['accuracy'].append(acc)
                    res['f1'].append(f1)
                    res['precision'].append(prec)
                    res['recall'].append(rec)
                    # res['kappa'].append(kappa)
                    res['kappa_t'].append(kappa_t)
                    
                    if logging:
                        print(f"[{name}] Inst: {count} | Acc: {acc:.2%} | Prec(Atk): {prec:.2%} | Rec(Atk): {rec:.2%} | F1(Atk): {f1:.2%}")

            count += 1
        
        if in_attack:
            attack_regions.append((start_attack_idx, count))
            
        return results, attack_regions

    def print_metrics(self, results):
        print("\n" + "="*100)
        print(f"{'RELATÓRIO FINAL ACUMULATIVO':^100}")
        print("="*100)
        # print("NOTA: Precision, Recall e F1-Score são calculados para CLASSE ATAQUE (Binary, pos_label=1)")
        # print("      zero_division=1.0: Em janelas sem ataques e sem erros, métrica = 1.0 (perfeito)")
        # print("      Kappa T: Kappa Temporal - concordância ponderada por tempo (apenas Kappa T ativo)")
        # print("="*100)
        # Ajuste de formatação (Kappa removido, mantém apenas Kappa T)
        print(f"{'Modelo':<25} | {'Acc':<8} | {'Prec':<8} | {'Recall':<8} | {'F1':<8} | {'KapT':<8}")
        print("-" * 100)
        
        for name, data in results.items():
            c_eval = data['cumulative_evaluator']
            acc = self._sanitize_metric(c_eval.accuracy())
            
            # Precision, Recall e F1 acumulativos com sklearn (BINARY - foco em ataques)
            if len(data['y_true_cumulative']) >= 2:
                prec_raw = precision_score(
                    data['y_true_cumulative'],
                    data['y_pred_cumulative'],
                    average='binary',
                    pos_label=1,
                    zero_division=1.0  # Se não há erros, considera perfeito
                )
                rec_raw = recall_score(
                    data['y_true_cumulative'],
                    data['y_pred_cumulative'],
                    average='binary',
                    pos_label=1,
                    zero_division=1.0
                )
                f1_raw = f1_score(
                    data['y_true_cumulative'],
                    data['y_pred_cumulative'],
                    average='binary',
                    pos_label=1,
                    zero_division=1.0
                )
                prec = self._sanitize_metric(prec_raw)
                rec = self._sanitize_metric(rec_raw)
                f1 = self._sanitize_metric(f1_raw)
            else:
                prec = 0.0
                rec = 0.0
                f1 = 0.0

            # kappa = self._sanitize_metric(c_eval.kappa())
            kappa_t = self._sanitize_metric(c_eval.kappa_t())
            
            # Validações para detectar comportamentos problemáticos
            total_instances = len(data['y_true_cumulative'])  # total de instâncias processadas
            total_true_attacks = sum(data['y_true_cumulative'])  # quantidade de ataques reais
            total_pred_attacks = sum(data['y_pred_cumulative'])  # quantidade de ataques que o modelo previu
            
            attack_ratio_true = total_true_attacks / total_instances if total_instances > 0 else 0  # % de ataques reais
            attack_ratio_pred = total_pred_attacks / total_instances if total_instances > 0 else 0  # % de ataques previstos
            
            # matriz de confusão completa
            y_true_arr = np.array(data['y_true_cumulative'])
            y_pred_arr = np.array(data['y_pred_cumulative'])
            TP = np.sum((y_true_arr == 1) & (y_pred_arr == 1))  # ataques reais que o modelo detectou corretamente
            FP = np.sum((y_true_arr == 0) & (y_pred_arr == 1))  # benignos que o modelo classificou como ataques (falsos positivos)
            TN = np.sum((y_true_arr == 0) & (y_pred_arr == 0))  # benignos que o modelo classificou corretamente
            FN = np.sum((y_true_arr == 1) & (y_pred_arr == 0))  # ataques reais que o modelo deixou passar (falsos negativos)
            
            # Imprimir métricas
            print(f"{name:<25} | {acc:.4f}   | {prec:.4f}   | {rec:.4f}   | {f1:.4f}   | {kappa_t:.4f}")
            
            # Alertas de comportamento problemático 
            alerts = []  #
            
            if total_true_attacks > 0 and total_pred_attacks == 0:  # havia ataques reais, mas o modelo não previu nenhum 
                alerts.append(f"Nunca prevê ataques (FN={FN}, perdeu todos os {total_true_attacks} ataques)")
            elif attack_ratio_pred > 0.9:
                alerts.append(f"Prevê ataques demais ({attack_ratio_pred:.1%} do dataset, FP={FP})")  # Preve que mais de 90% das instâncias são ataques
            elif total_true_attacks > 0 and attack_ratio_pred < attack_ratio_true * 0.1:
                alerts.append(f"Prevê poucos ataques ({attack_ratio_pred:.1%} vs {attack_ratio_true:.1%} real, FN={FN})") # preve muito menos ataques do deveria - FN alto
            
            if prec == 1.0 and total_pred_attacks == 0:
                alerts.append("Precision=1.0 devido ao zero_division (sem predições de ataque)")  #
            
            if rec == 1.0 and total_true_attacks == 0:  #
                alerts.append("Recall=1.0 devido ao zero_division (sem ataques no dataset)")  #
            
            # Alertas sobre desbalanceamento de erros
            if FP > 0 and FN > 0:  #
                if FP > FN * 3:  #
                    alerts.append(f"Muitos FALSOS POSITIVOS: FP={FP} vs FN={FN}")
                elif FN > FP * 3:
                    alerts.append(f"Muitos FALSOS NEGATIVOS: FN={FN} vs FP={FP}")
            
            # Imprimir alertas se houver
            for alert in alerts:
                print(f"{'':>25}   {alert}")
        
        # Sumário estatístico do dataset (usa primeiro modelo)
        first_model_data = list(results.values())[0]
        total_inst = len(first_model_data['y_true_cumulative'])
        total_attacks = sum(first_model_data['y_true_cumulative'])
        total_benign = total_inst - total_attacks
        
        print("="*100)
        print(f"{'ESTATÍSTICAS DO DATASET':^100}")
        print("="*100)
        print(f"Total de instâncias: {total_inst:,}")
        print(f"  - Benign:  {total_benign:,} ({total_benign/total_inst:.2%})")
        print(f"  - Ataques: {total_attacks:,} ({total_attacks/total_inst:.2%})")
        if total_attacks > 0:
            ratio = total_benign / total_attacks
            print(f"  - Razão Benign/Ataque: {ratio:.2f}:1 {'(Desbalanceado!)' if ratio > 10 else ''}")
        print("="*100 + "\n")

    def plot_metrics(self, results, attack_regions=None, title="Métricas Janeladas"):
        # 5 subplots (Kappa comentado, mantém apenas Kappa T)
        fig, axes = plt.subplots(5, 1, figsize=(14, 17), sharex=True)
        (ax1, ax2, ax3, ax4, ax5) = axes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            x_axis = data['instances']
            
            def clean(d_list): return [0.0 if (v is None or np.isnan(v)) else v for v in d_list]

            # Linhas de métricas (Kappa comentado)
            ax1.plot(x_axis, clean(data['accuracy']), label=name, color=color, linewidth=2, zorder=3)
            ax2.plot(x_axis, clean(data['precision']), label=name, color=color, linewidth=2, zorder=3)
            ax3.plot(x_axis, clean(data['recall']), label=name, color=color, linewidth=2, zorder=3)
            ax4.plot(x_axis, clean(data['f1']), label=name, color=color, linewidth=2, zorder=3)
            # ax5.plot(x_axis, clean(data['kappa']), label=name, color=color, linewidth=2, zorder=3)
            ax5.plot(x_axis, clean(data['kappa_t']), label=name, color=color, linewidth=2, zorder=3)

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
        ax2.set_ylabel("Precision (Ataque)", fontsize=12)
        ax3.set_ylabel("Recall (Ataque)", fontsize=12)
        ax4.set_ylabel("F1-Score (Ataque)", fontsize=12)
        # ax5.set_ylabel("Kappa", fontsize=12)
        ax5.set_ylabel("Kappa T", fontsize=12)
        ax5.set_xlabel("Instâncias", fontsize=14)

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

        plt.title("Precision x Recall (Classe Ataque)", fontsize=14, fontweight='bold')
        plt.xlabel("Recall (Ataque)", fontsize=12)
        plt.ylabel("Precision (Ataque)", fontsize=12)
        plt.xlim(-0.05, 1.05)
        # plt.ylim(-0.05, 1.05) ######
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
        #self.plot_precision_recall(results)
        
        return results