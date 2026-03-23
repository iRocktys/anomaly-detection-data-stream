from capymoa.evaluation import ClassificationEvaluator
from src.Anomaly.Threshold import DSPOT
from src.Anomaly.Results import Metrics, Plots

class AnomalyExperimentRunner:
    def __init__(self, target_names):
        self.target_names = target_names
        self.normal_class_idx = 0
        for i, name in enumerate(target_names):
            if str(name).strip().upper() in ['BENIGN', 'NORMAL', '0']:
                self.normal_class_idx = i
                break
                
        self.metrics = Metrics()
        self.plots = Plots(target_names)

    def _run_anomaly_evaluation(self, stream, algorithms, window_size, title, warmup_instances=0, target_class=None, threshold=0.5):
        results_metrics = {}
        results_scores = {}
        attack_regions = []
        
        predictions_history = {}
        schema = stream.get_schema()

        for alg_idx, (alg_name, learner) in enumerate(algorithms.items()):
            stream.restart()
            evaluator_class = ClassificationEvaluator(schema=schema, window_size=window_size)
            
            history = {'instances': [], 'f1_score': [], 'precision': [], 'recall': []}
            results_scores[alg_name] = {'scores': []}
            
            alg_true_labels = []
            alg_predicted_classes = []
            
            count = 0
            in_attack = False
            start_attack = 0
            current_attack_label = None

            while stream.has_more_instances():
                instance = stream.next_instance()
                
                true_label_multiclass = instance.y_index 
                true_label_binary = 1 if true_label_multiclass != self.normal_class_idx else 0
                
                if alg_idx == 0:
                    is_attack = (true_label_binary == 1)
                    
                    if is_attack:
                        if not in_attack:
                            in_attack = True
                            start_attack = count
                            current_attack_label = true_label_multiclass
                        elif current_attack_label != true_label_multiclass:
                            attack_regions.append((start_attack, count, current_attack_label))
                            start_attack = count
                            current_attack_label = true_label_multiclass
                    else:
                        if in_attack:
                            in_attack = False
                            attack_regions.append((start_attack, count, current_attack_label))

                score = learner.score_instance(instance) 
                results_scores[alg_name]['scores'].append(score)
                
                predicted_class = 1 if score > threshold else 0
                
                alg_true_labels.append(true_label_binary)
                alg_predicted_classes.append(predicted_class)
                
                if count >= warmup_instances:
                    evaluator_class.update(true_label_binary, predicted_class)
               
                try:
                    learner.train(instance)
                except ValueError:
                    pass

                if count >= warmup_instances and count > 0 and count % window_size == 0:
                    class_metrics = evaluator_class.metrics_dict()
                    
                    # Chamando método da classe Metrics
                    f1_val = self.metrics.get_metric_classifier(class_metrics, 'f1_score', target_class=target_class)
                    prec_val = self.metrics.get_metric_classifier(class_metrics, 'precision', target_class=target_class)
                    recall_val = self.metrics.get_metric_classifier(class_metrics, 'recall', target_class=target_class)

                    history['instances'].append(count)
                    history['f1_score'].append(f1_val)
                    history['precision'].append(prec_val)
                    history['recall'].append(recall_val)
                        
                count += 1
                
            if alg_idx == 0 and in_attack:
                attack_regions.append((start_attack, count, current_attack_label))
                
            results_metrics[alg_name] = history
            
            predictions_history[alg_name] = {
                'true_labels': alg_true_labels,
                'predicted_classes': alg_predicted_classes
            }

        # Chamando métodos das classes instanciadas
        self.metrics.display_cumulative_metrics(predictions_history, warmup_instances=warmup_instances, target_class=target_class)
        self.plots.plot_score(results_scores, attack_regions, title, threshold)
        self.plots.plot_metrics(results_metrics, attack_regions, title, window_size)

    def _run_anomaly_DSPOT(self, stream, algorithms, window_size, title, warmup_instances=0, target_class=None, dspot_q=1e-3, dspot_depth=50, dspot_t_quantile=0.98):
        results_metrics = {}
        results_scores = {}
        attack_regions = []
        
        predictions_history = {}
        schema = stream.get_schema()

        for alg_idx, (alg_name, learner) in enumerate(algorithms.items()):
            stream.restart()
            evaluator_class = ClassificationEvaluator(schema=schema, window_size=window_size)
            
            # Instanciação do DSPOT atualizada com o novo parâmetro dspot_t_quantile
            dspot = DSPOT(q=dspot_q, depth=dspot_depth, t_quantile=dspot_t_quantile)
            
            history = {'instances': [], 'f1_score': [], 'precision': [], 'recall': []}
            results_scores[alg_name] = {'scores': [], 'thresholds': [], 'trends': []}
            
            alg_true_labels = []
            alg_predicted_classes = []
            
            count = 0
            in_attack = False
            start_attack = 0
            current_attack_label = None

            while stream.has_more_instances():
                instance = stream.next_instance()
                
                true_label_multiclass = instance.y_index 
                true_label_binary = 1 if true_label_multiclass != self.normal_class_idx else 0
                
                if alg_idx == 0:
                    is_attack = (true_label_binary == 1)
                    
                    if is_attack:
                        if not in_attack:
                            in_attack = True
                            start_attack = count
                            current_attack_label = true_label_multiclass
                        elif current_attack_label != true_label_multiclass:
                            attack_regions.append((start_attack, count, current_attack_label))
                            start_attack = count
                            current_attack_label = true_label_multiclass
                    else:
                        if in_attack:
                            in_attack = False
                            attack_regions.append((start_attack, count, current_attack_label))

                score = learner.score_instance(instance) 
                
                predicted_class, dyn_thresh, local_trend = dspot.update_and_predict(score, warmup_instances)
                
                results_scores[alg_name]['scores'].append(score)
                results_scores[alg_name]['thresholds'].append(dyn_thresh)
                results_scores[alg_name]['trends'].append(local_trend)
                
                alg_true_labels.append(true_label_binary)
                alg_predicted_classes.append(predicted_class)
                
                if count >= warmup_instances:
                    evaluator_class.update(true_label_binary, predicted_class)
               
                try:
                    learner.train(instance)
                except ValueError:
                    pass

                if count >= warmup_instances and count > 0 and count % window_size == 0:
                    class_metrics = evaluator_class.metrics_dict()
                    
                    # Chamando método da classe Metrics
                    f1_val = self.metrics.get_metric_classifier(class_metrics, 'f1_score', target_class=target_class)
                    prec_val = self.metrics.get_metric_classifier(class_metrics, 'precision', target_class=target_class)
                    recall_val = self.metrics.get_metric_classifier(class_metrics, 'recall', target_class=target_class)

                    history['instances'].append(count)
                    history['f1_score'].append(f1_val)
                    history['precision'].append(prec_val)
                    history['recall'].append(recall_val)
                        
                count += 1
                
            if alg_idx == 0 and in_attack:
                attack_regions.append((start_attack, count, current_attack_label))
                
            results_metrics[alg_name] = history
            
            predictions_history[alg_name] = {
                'true_labels': alg_true_labels,
                'predicted_classes': alg_predicted_classes
            }

        # Chamando métodos das classes instanciadas
        # self.metrics.display_cumulative_metrics(predictions_history, warmup_instances=warmup_instances, target_class=target_class)
        # self.plots.plot_dspot_score(results_scores, attack_regions, title)
        # self.plots.plot_metrics(results_metrics, attack_regions, title, window_size)

        primeiro_algoritmo = list(algorithms.keys())[0]
        dados_finais = predictions_history[primeiro_algoritmo]
        
        y_true_final = dados_finais['true_labels'][warmup_instances:] if len(dados_finais['true_labels']) > warmup_instances else dados_finais['true_labels']
        y_pred_final = dados_finais['predicted_classes'][warmup_instances:] if len(dados_finais['predicted_classes']) > warmup_instances else dados_finais['predicted_classes']
        
        f1_final, prec_final, recall_final = self.metrics.calc_sklearn_metrics(y_true_final, y_pred_final, target_class)
        
        return {
            'f1_score': f1_final,
            'precision': prec_final,
            'recall': recall_final
        }

    def _run_poisoning_evolution(self, scenarios_dict, threshold_type='dspot', fixed_threshold=0.5, warmup_instances=0, dspot_q=1e-3, dspot_depth=50):
        results_scores = {}
        
        for ds_name, setup in scenarios_dict.items():
            stream = setup['stream']
            learner = setup['learner']
            stream.restart()
            
            if threshold_type == 'dspot':
                dspot = DSPOT(q=dspot_q, depth=dspot_depth)
                
            results_scores[ds_name] = {'scores': [], 'thresholds': [], 'trends': [], 'attack_regions': []}
            
            count = 0
            in_attack = False
            start_attack = 0
            current_attack_label = None
            
            while stream.has_more_instances():
                instance = stream.next_instance()
                true_label_multiclass = instance.y_index 
                true_label_binary = 1 if true_label_multiclass != self.normal_class_idx else 0
                
                is_attack = (true_label_binary == 1)
                
                if is_attack:
                    if not in_attack:
                        in_attack = True
                        start_attack = count
                        current_attack_label = true_label_multiclass
                    elif current_attack_label != true_label_multiclass:
                        results_scores[ds_name]['attack_regions'].append((start_attack, count, current_attack_label))
                        start_attack = count
                        current_attack_label = true_label_multiclass
                else:
                    if in_attack:
                        in_attack = False
                        results_scores[ds_name]['attack_regions'].append((start_attack, count, current_attack_label))

                score = learner.score_instance(instance) 
                
                if threshold_type == 'dspot':
                    predicted_class, dyn_thresh, local_trend = dspot.update_and_predict(score, warmup_instances)
                    results_scores[ds_name]['thresholds'].append(dyn_thresh)
                    results_scores[ds_name]['trends'].append(local_trend)
                else:
                    predicted_class = 1 if score > fixed_threshold else 0

                results_scores[ds_name]['scores'].append(score)
                
                try:
                    learner.train(instance)
                except ValueError:
                    pass
                        
                count += 1
                
            if in_attack:
                results_scores[ds_name]['attack_regions'].append((start_attack, count, current_attack_label))
                
        return results_scores
    
    def _run_anomaly_ae(self, stream, algorithms, window_size, title, warmup_instances=0, target_class=None, threshold=0.5):
        results_metrics = {}
        results_scores = {}
        attack_regions = []
        
        predictions_history = {}
        schema = stream.get_schema()
        
        # Garante que o warmup seja pelo menos 0
        min_warmup_required = max(warmup_instances, 0)

        for alg_idx, (alg_name, learner) in enumerate(algorithms.items()):
            stream.restart()
            evaluator_class = ClassificationEvaluator(schema=schema, window_size=window_size)
            
            history = {'instances': [], 'f1_score': [], 'precision': [], 'recall': []}
            results_scores[alg_name] = {'scores': []}
            
            alg_true_labels = []
            alg_predicted_classes = []
            
            count = 0
            in_attack = False
            start_attack = 0
            current_attack_label = None

            while stream.has_more_instances():
                instance = stream.next_instance()
                
                true_label_multiclass = instance.y_index 
                true_label_binary = 1 if true_label_multiclass != self.normal_class_idx else 0
                
                if alg_idx == 0:
                    is_attack = (true_label_binary == 1)
                    
                    if is_attack:
                        if not in_attack:
                            in_attack = True
                            start_attack = count
                            current_attack_label = true_label_multiclass
                        elif current_attack_label != true_label_multiclass:
                            attack_regions.append((start_attack, count - 1, current_attack_label))
                            start_attack = count
                            current_attack_label = true_label_multiclass
                    else:
                        if in_attack:
                            in_attack = False
                            attack_regions.append((start_attack, count - 1, current_attack_label))

                is_warmup_phase = count < min_warmup_required

                score = learner.score_instance(instance) 
                results_scores[alg_name]['scores'].append(score)
                
                predicted_class = 1 if score > threshold else 0
                
                alg_true_labels.append(true_label_binary)
                alg_predicted_classes.append(predicted_class)
                
                if count >= min_warmup_required:
                    evaluator_class.update(true_label_binary, predicted_class)
               
                try:
                    if is_warmup_phase:
                        learner.train(instance)
                except ValueError:
                    pass

                if count >= min_warmup_required and count > 0 and count % window_size == 0:
                    class_metrics = evaluator_class.metrics_dict()
                    
                    # Chamando método da classe Metrics
                    f1_val = self.metrics.get_metric_classifier(class_metrics, 'f1_score', target_class=target_class)
                    prec_val = self.metrics.get_metric_classifier(class_metrics, 'precision', target_class=target_class)
                    recall_val = self.metrics.get_metric_classifier(class_metrics, 'recall', target_class=target_class)

                    history['instances'].append(count)
                    history['f1_score'].append(f1_val)
                    history['precision'].append(prec_val)
                    history['recall'].append(recall_val)
                        
                count += 1
                
            if alg_idx == 0 and in_attack:
                attack_regions.append((start_attack, count - 1, current_attack_label))
                
            results_metrics[alg_name] = history
            
            predictions_history[alg_name] = {
                'true_labels': alg_true_labels,
                'predicted_classes': alg_predicted_classes
            }

        # Chamando métodos das classes instanciadas
        self.metrics.display_cumulative_metrics(predictions_history, warmup_instances=min_warmup_required, target_class=target_class)
        self.plots.plot_score(results_scores, attack_regions, title, threshold)
        self.plots.plot_metrics(results_metrics, attack_regions, title, window_size)