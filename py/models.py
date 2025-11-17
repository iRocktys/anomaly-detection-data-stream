from capymoa.classifier import (
    LeveragingBagging,
    HoeffdingTree,
    HoeffdingAdaptiveTree,
    AdaptiveRandomForestClassifier,
)
import numpy as np
from collections import deque
from capymoa.evaluation import ClassificationEvaluator
from capymoa.drift.detectors import DDM, ADWIN, ABCD

def get_models(schema, window_size, delay_length=None):
    models_to_test = {
        "LeveragingBagging": {
            "model_instance": LeveragingBagging(
                schema=schema, 
                CLI=None, # (None)
                random_seed=1, # (1)
                base_learner=None, # (None)
                ensemble_size=100, # (100)
                minibatch_size=None, # (None)
                number_of_jobs=None # (None)
            ),
            "evaluator": ClassificationEvaluator(
                schema=schema, 
                window_size=window_size,
                allow_abstaining=True,
                moa_evaluator=None
            ),
            "drift_ddm": DDM(
                min_n_instances=30,
                warning_level=2.0,
                out_control_level=3.0,
                CLI=None
            ),
            "drift_adwin": ADWIN(
                delta=0.002, # 0.002
                CLI=None # None
            ),
            "drift_ABCD": ABCD(
                delta_drift=0.002, # 0.002 - O nível de confiança desejado para a detecção de uma deriva.
                delta_warn=0.01, # 0.01 - O nível de confiança desejado para a detecção de um aviso.
                model_id='ae', # 'ae'
                split_type='ed', # 'ed' 
                encoding_factor=0.5, # 0.5
                update_epochs=50, # 50 
                num_splits=20, # 20
                max_size=np.inf, # np.inf 
                subspace_threshold=2.5, # 2.5 
                n_min=100, # 100 
                maximum_absolute_value=1.0, # 1.0
                bonferroni=False # False 
            ),
            "window_errors": [],
            "results_accuracy": [],
            "results_drift_ddm": [],
            "results_drift_adwin": [],
            "results_drift_ABCD": []
        },
# -----------------------------------------------------------------------------------------------------        
        "HoeffdingAdaptiveTree": {
            "model_instance": HoeffdingAdaptiveTree(
                schema=schema,
                random_seed=0, # (0)
                grace_period=200, # (200)
                split_criterion='InfoGainSplitCriterion', # ('InfoGainSplitCriterion')
                confidence=0.01, # (0.01)
                tie_threshold=0.05, # (0.05)
                leaf_prediction='NaiveBayesAdaptive',
                nb_threshold=0, # (0)
                numeric_attribute_observer='GaussianNumericAttributeClassObserver', # ('GaussianNumericAttributeClassObserver')
                binary_split=False, # (False)
                max_byte_size=33554432, # (32MB)
                memory_estimate_period=1000000, # (1.000.000)
                stop_mem_management=True, # (True)
                remove_poor_attrs=False, # (False)
                disable_prepruning=True # (True)
            ),
            "evaluator": ClassificationEvaluator(
                schema=schema, 
                window_size=window_size,
                allow_abstaining=True,
                moa_evaluator=None
            ),
            "drift_ddm": DDM(
                min_n_instances=30,
                warning_level=2.0,
                out_control_level=3.0,
                CLI=None
            ),
            "drift_adwin": ADWIN(
                delta=0.002, # 0.002
                CLI=None # None
            ),
            "drift_ABCD": ABCD(
                delta_drift=0.002, # 0.002 - O nível de confiança desejado para a detecção de uma deriva.
                delta_warn=0.01, # 0.01 - O nível de confiança desejado para a detecção de um aviso.
                model_id='ae', # 'ae'
                split_type='ed', # 'ed' 
                encoding_factor=0.5, # 0.5
                update_epochs=50, # 50 
                num_splits=20, # 20
                max_size=np.inf, # np.inf 
                subspace_threshold=2.5, # 2.5 
                n_min=100, # 100 
                maximum_absolute_value=1.0, # 1.0
                bonferroni=False # False 
            ),
            "window_errors": [],
            "results_accuracy": [],
            "results_drift_ddm": [],
            "results_drift_adwin": [],
            "results_drift_ABCD": []
        },
# -----------------------------------------------------------------------------------------------------
        "AdaptiveRandomForest": {
            "model_instance": AdaptiveRandomForestClassifier(
                schema=schema,
                CLI=None, # (None)
                random_seed=1, # (1)
                base_learner=None, # (None)
                ensemble_size=100, # (100)
                max_features=0.6, # (0.6)
                lambda_param=6.0, # (6.0)
                minibatch_size=None, # (None)
                number_of_jobs=1, # (1)
                drift_detection_method=None, # (None)
                warning_detection_method=None, # (None)
                disable_weighted_vote=False, # (False)
                disable_drift_detection=False, # (False)
                disable_background_learner=False # (False) 
            ),
            "evaluator": ClassificationEvaluator(
                schema=schema, 
                window_size=window_size,
                allow_abstaining=True,
                moa_evaluator=None
            ),
            "drift_ddm": DDM(
                min_n_instances=30,
                warning_level=2.0,
                out_control_level=3.0,
                CLI=None
            ),
            "drift_adwin": ADWIN(
                delta=0.002, # 0.002
                CLI=None # None
            ),
            "drift_ABCD": ABCD(
                delta_drift=0.002, # 0.002 - O nível de confiança desejado para a detecção de uma deriva.
                delta_warn=0.01, # 0.01 - O nível de confiança desejado para a detecção de um aviso.
                model_id='ae', # 'ae'
                split_type='ed', # 'ed' 
                encoding_factor=0.5, # 0.5
                update_epochs=50, # 50 
                num_splits=20, # 20
                max_size=np.inf, # np.inf 
                subspace_threshold=2.5, # 2.5 
                n_min=100, # 100 
                maximum_absolute_value=1.0, # 1.0
                bonferroni=False # False 
            ),
            "window_errors": [],
            "results_accuracy": [],
            "results_drift_ddm": [],
            "results_drift_adwin": [],
            "results_drift_ABCD": []
        },
# -----------------------------------------------------------------------------------------------------
        "HoeffdingTree": {
            "model_instance": HoeffdingTree(
                schema=schema, 
                random_seed=1
            ),
            "evaluator": ClassificationEvaluator(
                schema=schema, 
                window_size=window_size,
                allow_abstaining=True,
                moa_evaluator=None
            ),
            "drift_ddm": DDM(
                min_n_instances=30,
                warning_level=2.0,
                out_control_level=3.0,
                CLI=None
            ),
            "drift_adwin": ADWIN(
                delta=0.002, # 0.002
                CLI=None # None
            ),
            "drift_ABCD": ABCD(
                delta_drift=0.002, # 0.002 - O nível de confiança desejado para a detecção de uma deriva.
                delta_warn=0.01, # 0.01 - O nível de confiança desejado para a detecção de um aviso.
                model_id='ae', # 'ae'
                split_type='ed', # 'ed' 
                encoding_factor=0.5, # 0.5
                update_epochs=50, # 50 
                num_splits=20, # 20
                max_size=np.inf, # np.inf 
                subspace_threshold=2.5, # 2.5 
                n_min=100, # 100 
                maximum_absolute_value=1.0, # 1.0
                bonferroni=False # False 
            ),
            "window_errors": [],
            "results_accuracy": [],
            "results_drift_ddm": [],
            "results_drift_adwin": [],
            "results_drift_ABCD": []
        }
    }

    if delay_length is not None:
        for model_name in models_to_test:
            models_to_test[model_name]["prediction_queue"] = deque(maxlen=delay_length)
            
    return models_to_test