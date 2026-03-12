import optuna
import numpy as np
from capymoa.evaluation import ClassificationEvaluator, AnomalyDetectionEvaluator
from src.Anomaly_Models import get_anomaly_models

# Silencia logs em excesso do Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

class AnomalyOptunaOptimizer:
    def __init__(self, stream, n_trials=30, discretization_threshold=0.5):
        self.stream = stream
        self.schema = stream.get_schema()
        self.n_trials = n_trials
        self.discretization_threshold = discretization_threshold
        self.best_params = {}

    def _get_metric_target_class(self, metrics_dict, metric_name, target_class=1):
        # Tenta pegar a métrica específica da classe de ataque
        val = metrics_dict.get(f'{metric_name}_{target_class}')
        if val is None or np.isnan(val):
            # Fallback para a métrica geral caso não encontre
            val = metrics_dict.get(metric_name, 0.0)
        return float(val) if not np.isnan(val) else 0.0

    def _evaluate_model(self, model):
        self.stream.restart()
        evaluator_class = ClassificationEvaluator(schema=self.schema)
        evaluator_anomaly = AnomalyDetectionEvaluator(schema=self.schema)
        
        while self.stream.has_more_instances():
            instance = self.stream.next_instance()
            true_label = instance.y_index 
            
            score = model.score_instance(instance) 
            predicted_class = 1 if score > self.discretization_threshold else 0
            
            evaluator_class.update(true_label, predicted_class)
            evaluator_anomaly.update(true_label, score)

            try: 
                model.train(instance)
            except ValueError: 
                pass

        class_metrics = evaluator_class.metrics_dict()
        anom_metrics = evaluator_anomaly.metrics_dict()

        # AGORA SIM: Extraindo F1 e Precision focados em detectar os ataques (Classe 1)
        f1_val = self._get_metric_target_class(class_metrics, 'f1_score', target_class=1)
        prec_val = self._get_metric_target_class(class_metrics, 'precision', target_class=1)
        
        recall_val = anom_metrics.get('recall')
        if recall_val is None:
            recall_val = anom_metrics.get('Recall', 0.0)
            
        return f1_val, prec_val, recall_val

    def _optuna_callback(self, study, trial):
        f1, prec, rec = trial.values
        print(f"Trial {trial.number + 1}/{self.n_trials} | "
              f"F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

    def _objective_hst(self, trial):
        hst_params = {
            'window_size': trial.suggest_int('window_size', 100, 2000, step=100),
            'number_of_trees': trial.suggest_int('number_of_trees', 10, 100, step=10),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'anomaly_threshold': trial.suggest_float('anomaly_threshold', 0.1, 0.9, step=0.1),
            'size_limit': trial.suggest_float('size_limit', 0.01, 0.5, step=0.05)
        }
        models = get_anomaly_models(self.schema, selected_models=['HST'], hst_params=hst_params)
        return self._evaluate_model(models['HalfSpaceTrees'])

    def _objective_oif(self, trial):
        oif_params = {
            'num_trees': trial.suggest_int('num_trees', 10, 100, step=10),
            'max_leaf_samples': trial.suggest_int('max_leaf_samples', 16, 128, step=16),
            'growth_criterion': trial.suggest_categorical('growth_criterion', ['fixed', 'adaptive']),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0, step=0.1),
            'window_size': trial.suggest_int('window_size', 512, 4096, step=256),
            'branching_factor': trial.suggest_int('branching_factor', 2, 5)
        }
        models = get_anomaly_models(self.schema, selected_models=['OIF'], oif_params=oif_params)
        return self._evaluate_model(models['OnlineIsolationForest'])

    def _objective_ae(self, trial):
        ae_params = {
            'hidden_layer': trial.suggest_int('hidden_layer', 1, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'threshold': trial.suggest_float('threshold', 0.1, 0.9, step=0.1)
        }
        models = get_anomaly_models(self.schema, selected_models=['AE'], ae_params=ae_params)
        return self._evaluate_model(models['Autoencoder'])

    def _objective_rrcf(self, trial):
        rrcf_params = {
            'tree_size': trial.suggest_int('tree_size', 100, 2000, step=100),
            'n_trees': trial.suggest_int('n_trees', 10, 200, step=10)
        }
        models = get_anomaly_models(self.schema, selected_models=['RRCF'], rrcf_params=rrcf_params)
        return self._evaluate_model(models['RobustRandomCutForest'])

    def _objective_aif(self, trial):
        aif_params = {
            'window_size': trial.suggest_int('window_size', 128, 1024, step=128),
            'n_trees': trial.suggest_int('n_trees', 10, 200, step=10),
            'm_trees': trial.suggest_int('m_trees', 5, 50, step=5),
            'weights': trial.suggest_float('weights', 0.1, 0.9, step=0.1)
        }
        models = get_anomaly_models(self.schema, selected_models=['AIF'], aif_params=aif_params)
        return self._evaluate_model(models['AdaptiveIsolationForest'])

    def optimize(self, model_name):
        print(f"\n[{model_name}] Iniciando otimização MULTIOBJETIVO ({self.n_trials} trials)...")
        study = optuna.create_study(directions=['maximize', 'maximize', 'maximize'])
        
        if model_name == 'HST':
            study.optimize(self._objective_hst, n_trials=self.n_trials, callbacks=[self._optuna_callback])
        elif model_name == 'OIF':
            study.optimize(self._objective_oif, n_trials=self.n_trials, callbacks=[self._optuna_callback])
        elif model_name == 'AE':
            study.optimize(self._objective_ae, n_trials=self.n_trials, callbacks=[self._optuna_callback])
        elif model_name == 'RRCF':
            study.optimize(self._objective_rrcf, n_trials=self.n_trials, callbacks=[self._optuna_callback])
        elif model_name == 'AIF':
            study.optimize(self._objective_aif, n_trials=self.n_trials, callbacks=[self._optuna_callback])
        else:
            raise ValueError(f"Modelo {model_name} não suportado para otimização.")
            
        # O PULO DO GATO: Multiplicando os valores (Produto)! 
        # Se F1, Prec ou Rec for 0.0, a nota inteira vira 0 e o Optuna descarta.
        best_trial = max(study.best_trials, key=lambda t: t.values[0] * t.values[1] * t.values[2])
        
        print(f"\n[{model_name}] --- OTIMIZAÇÃO FINALIZADA ---")
        print(f"Melhor Equilíbrio -> F1: {best_trial.values[0]:.4f} | Prec: {best_trial.values[1]:.4f} | Rec: {best_trial.values[2]:.4f}")
        print(f"Melhores Parâmetros: {best_trial.params}")
        
        self.best_params[model_name] = best_trial.params
        return best_trial.params