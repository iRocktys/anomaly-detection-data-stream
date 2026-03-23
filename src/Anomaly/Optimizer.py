import optuna
import numpy as np
from capymoa.evaluation import ClassificationEvaluator
from src.Anomaly.Models import get_anomaly_models
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import math
from collections import deque
optuna.logging.set_verbosity(optuna.logging.WARNING)

class AnomalyOptunaOptimizer:
    def __init__(self, stream, n_trials=30, discretization_threshold=0.5, target_class=1, target_names=None):
        self.stream = stream
        self.schema = stream.get_schema()
        self.n_trials = n_trials
        self.discretization_threshold = discretization_threshold
        self.target_class = target_class
        self.best_params = {}
        self.target_names = target_names if target_names is not None else ['Normal', 'Ataque']

    def _optuna_callback(self, study, trial):
        f1, prec, rec = trial.user_attrs['metrics']
        scores = trial.user_attrs['scores']
        model_name = trial.user_attrs['model_name']
        attack_regions = trial.user_attrs['attack_regions'] 
        trial_threshold = trial.user_attrs['trial_threshold']
        params = trial.params
        
        print(f"Trial {trial.number + 1}/{self.n_trials} | "
              f"F1: {f1:.2f} | Prec: {prec:.2f} | Rec: {rec:.2f} | "
              f"Params: {params}")
        
        results_dict = {f"{model_name} (Trial {trial.number + 1})": {'scores': scores}}
        title = f"Scores - {model_name} | Trial {trial.number + 1}"
        
        # Passa o trial_threshold para desenhar a linha vermelha no lugar certo
        # self.plot_score(results_dict, attack_regions, title, threshold=trial_threshold)

    def _evaluate_ae(self, model, trial_threshold, warmup_instances=0):
        self.stream.restart()
        
        scores_list = []
        attack_regions = []
        
        y_true_list = []
        y_pred_list = []
        
        in_attack = False
        start_idx = 0
        current_attack_idx = None
        instance_idx = 0 
        
        # Garante um tempo de aprendizado inicial
        min_warmup_required = max(warmup_instances, 0)

        while self.stream.has_more_instances():
            instance = self.stream.next_instance()
            true_label_multiclass = instance.y_index 
            
            if true_label_multiclass != 0: 
                if not in_attack:
                    in_attack = True
                    start_idx = instance_idx
                    current_attack_idx = true_label_multiclass
                elif true_label_multiclass != current_attack_idx:
                    attack_regions.append((start_idx, instance_idx - 1, current_attack_idx))
                    start_idx = instance_idx
                    current_attack_idx = true_label_multiclass
            else:
                if in_attack:
                    attack_regions.append((start_idx, instance_idx - 1, current_attack_idx))
                    in_attack = False
                    current_attack_idx = None
            
            binary_true_label = 1 if true_label_multiclass > 0 else 0
            
            # Define se ainda estamos na fase de aquecimento
            is_warmup_phase = instance_idx < min_warmup_required
            
            raw_score = model.score_instance(instance) 
            scores_list.append(raw_score)
            
            predicted_class = 1 if raw_score > trial_threshold else 0
            
            y_true_list.append(binary_true_label)
            y_pred_list.append(predicted_class)

            try:
                if is_warmup_phase:
                    model.train(instance)
            except ValueError:
                pass
            
            instance_idx += 1

        if in_attack:
            attack_regions.append((start_idx, instance_idx - 1, current_attack_idx))

        # A avaliação das métricas ignora a fase de warmup, como já estava configurado
        eval_start = min_warmup_required
        y_true_eval = y_true_list[eval_start:] if len(y_true_list) > eval_start else y_true_list
        y_pred_eval = y_pred_list[eval_start:] if len(y_pred_list) > eval_start else y_pred_list

        f1_val, prec_val, recall_val = self._get_metric_classifier(
            y_true=y_true_eval, 
            y_pred=y_pred_eval, 
            target_class=self.target_class
        )

        return f1_val, prec_val, recall_val, scores_list, attack_regions
    
    def _evaluate_model(self, model, model_name, trial_threshold, warmup_instances=0):
        self.stream.restart()
        
        scores_list = []
        attack_regions = []
        
        y_true_list = []
        y_pred_list = []
        
        in_attack = False
        start_idx = 0
        current_attack_idx = None
        instance_idx = 0 
        
        while self.stream.has_more_instances():
            instance = self.stream.next_instance()
            true_label_multiclass = instance.y_index 
            
            if true_label_multiclass != 0: 
                if not in_attack:
                    in_attack = True
                    start_idx = instance_idx
                    current_attack_idx = true_label_multiclass
                elif true_label_multiclass != current_attack_idx:
                    attack_regions.append((start_idx, instance_idx - 1, current_attack_idx))
                    start_idx = instance_idx
                    current_attack_idx = true_label_multiclass
            else:
                if in_attack:
                    attack_regions.append((start_idx, instance_idx - 1, current_attack_idx))
                    in_attack = False
                    current_attack_idx = None
            
            binary_true_label = 1 if true_label_multiclass > 0 else 0
            
            score = model.score_instance(instance) 
            scores_list.append(score)
            
            predicted_class = 1 if score > trial_threshold else 0
            
            y_true_list.append(binary_true_label)
            y_pred_list.append(predicted_class)

            try: 
                model.train(instance)
            except ValueError:
                pass
            
            instance_idx += 1

        if in_attack:
            attack_regions.append((start_idx, instance_idx - 1, current_attack_idx))

        # Ignora apenas o período de warm-up explícito
        y_true_eval = y_true_list[warmup_instances:] if len(y_true_list) > warmup_instances else y_true_list
        y_pred_eval = y_pred_list[warmup_instances:] if len(y_pred_list) > warmup_instances else y_pred_list

        # Cálculo das Métricas com Scikit-Learn
        f1_val, prec_val, recall_val = self._get_metric_classifier(
            y_true=y_true_eval, 
            y_pred=y_pred_eval, 
            target_class=self.target_class
        )

        return f1_val, prec_val, recall_val, scores_list, attack_regions

    def optimize(self, model_name, warmup_instances=0):
        tgt_str = f"Classe {self.target_class}" if self.target_class is not None else "Macro"
        print(f"\n[{model_name}] Iniciando otimização focada no F1-Score ({tgt_str}) com {self.n_trials} trials...")
        
        study = optuna.create_study(direction='maximize')
        
        def objective_wrapper(trial):
            # Transforma em minúsculo para facilitar a verificação
            mode = str(self.discretization_threshold).lower()
            
            # Dinâmico - Thresholds totalmente independentes
            if mode in ["dinamico", "dinâmico"]:
                trial_threshold = trial.suggest_float('dynamic_threshold', 0.5, 0.95, step=0.05)
                if model_name == 'AE':
                    model_threshold = trial.suggest_float('threshold', 0.05, 0.9, step=0.05)
                elif model_name == 'HST':
                    model_threshold = trial.suggest_float('anomaly_threshold', 0.1, 0.9, step=0.05)

            # Threshold unificado para discretização e parâmetro interno
            elif mode == "params":
                if model_name == 'AE':
                    shared_threshold = trial.suggest_float('threshold', 0.1, 0.9, step=0.05)
                    trial_threshold = shared_threshold
                    model_threshold = shared_threshold
                elif model_name == 'HST':
                    shared_threshold = trial.suggest_float('anomaly_threshold', 0.1, 0.9, step=0.05)
                    trial_threshold = shared_threshold
                    model_threshold = shared_threshold
                else:
                    # Fallback para modelos que não possuem threshold interno
                    trial_threshold = trial.suggest_float('dynamic_threshold', 0.05, 0.95, step=0.05)

            # Fixo - Discretização travada no valor informado
            else:
                trial_threshold = float(self.discretization_threshold)
                if model_name == 'AE':
                    model_threshold = trial.suggest_float('threshold', 0.1, 0.9, step=0.05)
                elif model_name == 'HST':
                    model_threshold = trial.suggest_float('anomaly_threshold', 0.1, 0.9, step=0.05)

            # Roteamento para os modelos
            if model_name == 'HST':
                f1, prec, rec, scores, attack_regions = self._objective_hst(trial, trial_threshold, model_threshold, warmup_instances)
            elif model_name == 'OIF':
                f1, prec, rec, scores, attack_regions = self._objective_oif(trial, trial_threshold, warmup_instances)
            elif model_name == 'AE':
                f1, prec, rec, scores, attack_regions = self._objective_ae(trial, trial_threshold, model_threshold, warmup_instances)
            elif model_name == 'RRCF':
                f1, prec, rec, scores, attack_regions = self._objective_rrcf(trial, trial_threshold, warmup_instances)
            elif model_name == 'AIF':
                f1, prec, rec, scores, attack_regions = self._objective_aif(trial, trial_threshold, warmup_instances)
            else:
                raise ValueError("Modelo não suportado.")
            
            trial.set_user_attr('metrics', (f1, prec, rec))
            trial.set_user_attr('scores', scores)
            trial.set_user_attr('model_name', model_name)
            trial.set_user_attr('attack_regions', attack_regions)
            trial.set_user_attr('trial_threshold', trial_threshold)
            
            return f1 

        study.optimize(objective_wrapper, n_trials=self.n_trials, callbacks=[self._optuna_callback])
        
        print(f"\n[{model_name}] OTIMIZAÇÃO FINALIZADA")
        best_trial = study.best_trial
        best_f1, best_prec, best_rec = best_trial.user_attrs['metrics']
        
        print(f"Melhor Trial: {best_trial.number + 1}")
        print(f"Melhor Resultado -> F1: {best_f1:.2f} | Prec: {best_prec:.2f} | Rec: {best_rec:.2f}")
        print(f"Melhores Parâmetros: {study.best_params}")
        
        self.best_params[model_name] = study.best_params
        return study.best_params
    
    def extract_attack_regions(targets, normal_class=0):
        regions = []
        in_attack = False
        start_idx = 0
        current_attack = None
        
        for i, label in enumerate(targets):
            # Se não for a classe normal (0), é um ataque
            if label != normal_class:
                if not in_attack:
                    # Começou um ataque
                    in_attack = True
                    start_idx = i
                    current_attack = label
                elif label != current_attack:
                    # Mudou de um ataque direto para outro tipo de ataque
                    regions.append((start_idx, i - 1, current_attack))
                    start_idx = i
                    current_attack = label
            else:
                # Voltou para o normal, fechamos a janela de ataque
                if in_attack:
                    regions.append((start_idx, i - 1, current_attack))
                    in_attack = False
                    current_attack = None
                    
        if in_attack:
            regions.append((start_idx, len(targets) - 1, current_attack))
            
        return regions

    def plot_score(self, results, attack_regions, title, threshold=0.5):
        fig, ax = plt.subplots(figsize=(15, 6)) 
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        bg_colors = ['#F7C5CD', '#C5D9F7', '#C5F7C5', '#F7E6C5', '#E3C5F7', '#F7D9C5', '#C5F7E6']
        
        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
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
            bg_color = bg_colors[attack_idx % len(bg_colors)]
            
            label_to_show = f'{attack_name}' if attack_name not in added_attack_labels else ""
            ax.axvspan(start, end, facecolor=bg_color, alpha=0.3, zorder=1, label=label_to_show)
            
            if label_to_show:
                added_attack_labels.add(attack_name)

        ax.set_title(f"{title} (Média Móvel - Janela {window_size})", fontsize=14, fontweight='bold')
        ax.set_ylabel("Score de Anomalia", fontsize=14)
        ax.set_xlabel("Instâncias", fontsize=14)
        
        # Legendas
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(results) + 2, 
                  fontsize=12, frameon=False)
        for patch in leg.get_patches():
            patch.set_edgecolor('gray')
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)

        ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
        # ax.set_ylim(0.0, 1.1)
        fig.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        plt.show()
    
    def _get_metric_classifier(self, y_true, y_pred, target_class=1):
        if len(y_true) == 0:
            return 0.0, 0.0, 0.0

        if target_class is None:
            # Macro average
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            # Focado em uma classe específica
            f1 = f1_score(y_true, y_pred, pos_label=target_class, average='binary', zero_division=0)
            prec = precision_score(y_true, y_pred, pos_label=target_class, average='binary', zero_division=0)
            rec = recall_score(y_true, y_pred, pos_label=target_class, average='binary', zero_division=0)
            
        return f1 * 100.0, prec * 100.0, rec * 100.0
    
    def _objective_hst(self, trial, trial_threshold, model_threshold, warmup_instances):
        hst_params = {
            'window_size': trial.suggest_int('window_size', 100, 2050, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'anomaly_threshold': model_threshold,
            'size_limit': trial.suggest_float('size_limit', 0.05, 0.5, step=0.05)
        }
        models = get_anomaly_models(self.schema, selected_models=['HST'], hst_params=hst_params)
        
        # Sobrepõe o warmup fixo pelo tamanho da janela gerada neste trial
        dynamic_warmup = hst_params['window_size']
        
        return self._evaluate_model(
            models['HalfSpaceTrees'], 
            'HST', 
            trial_threshold, 
            warmup_instances=dynamic_warmup
        )

    def _objective_oif(self, trial, trial_threshold, warmup_instances):
        oif_params = {
            'num_trees': trial.suggest_int('num_trees', 10, 100, step=10),
            'max_leaf_samples': trial.suggest_int('max_leaf_samples', 16, 128, step=16),
            'growth_criterion': trial.suggest_categorical('growth_criterion', ['fixed', 'adaptive']),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0, step=0.1),
            'window_size': trial.suggest_int('window_size', 512, 4096, step=256),
            'branching_factor': trial.suggest_int('branching_factor', 2, 5)
        }
        models = get_anomaly_models(self.schema, selected_models=['OIF'], oif_params=oif_params)
        return self._evaluate_model(models['OnlineIsolationForest'], 'OIF', trial_threshold, warmup_instances=warmup_instances) 

    def _objective_ae(self, trial, trial_threshold, model_threshold, warmup_instances):
        num_features = self.schema.get_num_attributes()
        max_hidden = max(1, num_features - 1)

        ae_params = {
            'hidden_layer': trial.suggest_int('hidden_layer', 1, max_hidden),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'threshold': model_threshold
        }
        models = get_anomaly_models(self.schema, selected_models=['AE'], ae_params=ae_params)
        
        return self._evaluate_ae(models['Autoencoder'], trial_threshold, warmup_instances=warmup_instances)
    
    def _objective_rrcf(self, trial, trial_threshold, warmup_instances):
        rrcf_params = {
            'tree_size': trial.suggest_int('tree_size', 100, 2000, step=100),
            'n_trees': trial.suggest_int('n_trees', 10, 200, step=10)
        }
        models = get_anomaly_models(self.schema, selected_models=['RRCF'], rrcf_params=rrcf_params)
        return self._evaluate_model(models['RobustRandomCutForest'], 'RRCF', trial_threshold, warmup_instances=warmup_instances) 

    def _objective_aif(self, trial, trial_threshold, warmup_instances):
        aif_params = {
            'window_size': trial.suggest_int('window_size', 128, 1024, step=128),
            'height': trial.suggest_int('height', 5, 20),
        }
        models = get_anomaly_models(self.schema, selected_models=['AIF'], aif_params=aif_params)
        return self._evaluate_model(models['AdaptiveIsolationForest'], 'AIF', trial_threshold, warmup_instances=warmup_instances)