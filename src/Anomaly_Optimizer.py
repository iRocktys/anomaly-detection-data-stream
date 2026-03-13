import optuna
import numpy as np
from capymoa.evaluation import ClassificationEvaluator
from src.Anomaly_Models import get_anomaly_models
optuna.logging.set_verbosity(optuna.logging.WARNING)

class AnomalyOptunaOptimizer:
    def __init__(self, stream, n_trials=30, discretization_threshold=0.5, target_class=1):
        self.stream = stream
        self.schema = stream.get_schema()
        self.n_trials = n_trials
        self.discretization_threshold = discretization_threshold
        self.target_class = target_class
        self.best_params = {}

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
    
    def _optuna_callback(self, study, trial):
        f1, prec, rec = trial.user_attrs['metrics']
        params = trial.params
        print(f"Trial {trial.number + 1}/{self.n_trials} | "
              f"F1: {f1:.2f} | Prec: {prec:.2f} | Rec: {rec:.2f} | "
              f"Params: {params}")

    def _evaluate_model(self, model, model_name):
        self.stream.restart()
        evaluator_class = ClassificationEvaluator(schema=self.schema)
        
        while self.stream.has_more_instances():
            instance = self.stream.next_instance()
            true_label = instance.y_index 
            
            score = model.score_instance(instance) 
            predicted_class = 1 if score > self.discretization_threshold else 0
            
            evaluator_class.update(true_label, predicted_class)

            try: 
                if model_name == 'AE':
                    if predicted_class == 0:
                        model.train(instance)
                else:
                    model.train(instance)
            except ValueError: 
                pass

        class_metrics = evaluator_class.metrics_dict()
        f1_val = self._get_metric_classifier(class_metrics, 'f1_score', target_class=self.target_class)
        prec_val = self._get_metric_classifier(class_metrics, 'precision', target_class=self.target_class)
        recall_val = self._get_metric_classifier(class_metrics, 'recall', target_class=self.target_class)

        return f1_val, prec_val, recall_val

    def _objective_hst(self, trial):
        hst_params = {
            'window_size': trial.suggest_int('window_size', 100, 2000, step=100),
            'number_of_trees': trial.suggest_int('number_of_trees', 10, 100, step=10),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'anomaly_threshold': trial.suggest_float('anomaly_threshold', 0.1, 0.9, step=0.1),
            'size_limit': trial.suggest_float('size_limit', 0.01, 0.5, step=0.05)
        }
        models = get_anomaly_models(self.schema, selected_models=['HST'], hst_params=hst_params)
        return self._evaluate_model(models['HalfSpaceTrees'], 'HST') 

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
        return self._evaluate_model(models['OnlineIsolationForest'], 'OIF') 

    def _objective_ae(self, trial):
        ae_params = {
            'hidden_layer': trial.suggest_int('hidden_layer', 1, 5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'threshold': trial.suggest_float('threshold', 0.1, 0.9, step=0.1)
        }
        models = get_anomaly_models(self.schema, selected_models=['AE'], ae_params=ae_params)
        return self._evaluate_model(models['Autoencoder'], 'AE') 

    def _objective_rrcf(self, trial):
        rrcf_params = {
            'tree_size': trial.suggest_int('tree_size', 100, 2000, step=100),
            'n_trees': trial.suggest_int('n_trees', 10, 200, step=10)
        }
        models = get_anomaly_models(self.schema, selected_models=['RRCF'], rrcf_params=rrcf_params)
        return self._evaluate_model(models['RobustRandomCutForest'], 'RRCF') 

    def _objective_aif(self, trial):
        aif_params = {
            'window_size': trial.suggest_int('window_size', 128, 1024, step=128),
            'n_trees': trial.suggest_int('n_trees', 10, 200, step=10),
            'height': trial.suggest_int('height', 5, 20),
            'm_trees': trial.suggest_int('m_trees', 5, 50, step=5),
            'weights': trial.suggest_float('weights', 0.1, 0.9, step=0.1)
        }
        models = get_anomaly_models(self.schema, selected_models=['AIF'], aif_params=aif_params)
        return self._evaluate_model(models['AdaptiveIsolationForest'], 'AIF') 
    
    def optimize(self, model_name):
        tgt_str = f"Classe {self.target_class}" if self.target_class is not None else "Macro"
        print(f"\n[{model_name}] Iniciando otimização focada no F1-Score ({tgt_str}) com {self.n_trials} trials...")
        
        study = optuna.create_study(direction='maximize')
        
        def objective_wrapper(trial):
            if model_name == 'HST':
                f1, prec, rec = self._objective_hst(trial)
            elif model_name == 'OIF':
                f1, prec, rec = self._objective_oif(trial)
            elif model_name == 'AE':
                f1, prec, rec = self._objective_ae(trial)
            elif model_name == 'RRCF':
                f1, prec, rec = self._objective_rrcf(trial)
            elif model_name == 'AIF':
                f1, prec, rec = self._objective_aif(trial)
            else:
                raise ValueError("Modelo não suportado.")
            
            trial.set_user_attr('metrics', (f1, prec, rec))
            return f1 

        study.optimize(objective_wrapper, n_trials=self.n_trials, callbacks=[self._optuna_callback])
        
        print(f"\n[{model_name}] --- OTIMIZAÇÃO FINALIZADA ---")
        best_f1, best_prec, best_rec = study.best_trial.user_attrs['metrics']
        print(f"Melhor Resultado -> F1: {best_f1:.2f} | Prec: {best_prec:.2f} | Rec: {best_rec:.2f}")
        print(f"Melhores Parâmetros: {study.best_params}")
        
        self.best_params[model_name] = study.best_params
        return study.best_params