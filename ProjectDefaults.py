from copy import deepcopy
from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path("output")
DEFAULT_OPTIMIZATION_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "Optimization"
DEFAULT_SIMILARITY_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "Similarity"
DEFAULT_DSPOT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "DSPOT"

DEFAULT_TARGET_COLUMN = "Label"
DEFAULT_SELECTED_FEATURES = None
DEFAULT_REMOVED_FEATURES = None
DEFAULT_BINARY_LABEL = True
DEFAULT_NORMAL_CLASS_INDEX = 0
DEFAULT_NORMAL_LABELS = ("BENIGN", "NORMAL", "0")

DEFAULT_MODEL_CODE = "AIF"
DEFAULT_AIF_PARAMETERS = {
    "window_size": 256,
    "n_trees": 100,
    "height": 11,
    "m_trees": 10,
    "weights": 0.3722,
}
DEFAULT_MODEL_PARAMETERS = {
    "AIF": DEFAULT_AIF_PARAMETERS,
    "HST": {
        "CLI": None,
        "window_size": 250,
        "number_of_trees": 25,
        "max_depth": 15,
        "anomaly_threshold": 0.5,
        "size_limit": 0.1,
    },
    "AE": {
        "hidden_layer": 2,
        "learning_rate": 0.5,
        "threshold": 0.6,
    },
    "SRHF": {
        "max_height": 5,
        "num_trees": 100,
        "window_size": 20,
        "random_seed": 0,
    },
}

DEFAULT_CLASSIFIER_CODE = "HAT"
DEFAULT_CLASSIFIER_PARAMETERS = {
    "ARF": {
        "ensemble_size": 100,
        "max_features": 0.6,
        "lambda_param": 6.0,
        "number_of_jobs": 1,
    },
    "HAT": {
        "grace_period": 200,
        "confidence": 1e-3,
        "tie_threshold": 0.05,
        "leaf_prediction": "NaiveBayesAdaptive",
    },
    "HT": {
        "grace_period": 200,
        "confidence": 1e-3,
        "tie_threshold": 0.05,
        "leaf_prediction": "NaiveBayesAdaptive",
    },
    "NB": {},
}

DEFAULT_IMPUTER_NAME = "incrementalMean"
DEFAULT_IMPUTER_PARAMETERS = {}
DEFAULT_NORMALIZER_NAME = "incrementalZScore"
DEFAULT_NORMALIZER_PARAMETERS = {
    "epsilon": 1e-8,
    "clip": None,
}
DEFAULT_TRAINING_STRATEGY = "all"
DEFAULT_TRAINING_PARAMETERS = {}

DEFAULT_AIF_WARMUP_SIZE = 500
DEFAULT_DSPOT_CALIBRATION_WINDOW = 500
DEFAULT_INITIAL_WARMUP_SIZE = DEFAULT_AIF_WARMUP_SIZE + DEFAULT_DSPOT_CALIBRATION_WINDOW
DEFAULT_CLASSIFICATION_WARMUP_SIZE = 0

DEFAULT_THRESHOLD_NAME = "dspot"
DEFAULT_SCORE_WINDOW_SIZES = (10,)
DEFAULT_THRESHOLD_SCORE_SOURCE = "ma10"
DEFAULT_DSPOT_DRIFT_DEPTH = 50
DEFAULT_DSPOT_PARAMETERS = {
    "driftDepth": DEFAULT_DSPOT_DRIFT_DEPTH,
    "calibrationSize": DEFAULT_DSPOT_CALIBRATION_WINDOW - DEFAULT_DSPOT_DRIFT_DEPTH,
    "initialQuantile": 0.95,
    "risk": 0.001,
    "refitEvery": 10,
    "optimizationStarts": 10,
    "tolerance": 1e-8,
}

DEFAULT_METRICS_WINDOW_SIZE = 100
DEFAULT_SEED = 42
DEFAULT_GENERATE_PLOTS = True
DEFAULT_STRICT_TRAINING_ERRORS = False

DEFAULT_N_TRIALS = 100
DEFAULT_OPTIMIZATION_METRICS_WINDOW_SIZE = 100
DEFAULT_OPTUNA_SEED = 42
DEFAULT_OPTIMIZE_MODEL_PARAMETERS = True
DEFAULT_MAX_TRIAL_ATTEMPTS_MULTIPLIER = 10


def getModelDefaults(code=DEFAULT_MODEL_CODE):
    # Retorna uma cópia dos parâmetros padrão do detector de anomalias solicitado.
    resolved_code = str(code).strip().upper()
    if resolved_code not in DEFAULT_MODEL_PARAMETERS:
        raise ValueError(f"Unknown anomaly detector defaults: {resolved_code}.")
    return deepcopy(DEFAULT_MODEL_PARAMETERS[resolved_code])


def getClassifierDefaults(code=DEFAULT_CLASSIFIER_CODE):
    # Retorna uma cópia dos parâmetros padrão do classificador solicitado.
    resolved_code = str(code).strip().upper()
    if resolved_code not in DEFAULT_CLASSIFIER_PARAMETERS:
        raise ValueError(f"Unknown classifier defaults: {resolved_code}.")
    return deepcopy(DEFAULT_CLASSIFIER_PARAMETERS[resolved_code])


def getDefaultParameters():
    # Retorna a configuração padrão do experimento de detecção preservando o contrato original do projeto.
    return {
        "dataset": {
            "targetColumn": DEFAULT_TARGET_COLUMN,
            "selectedFeatures": DEFAULT_SELECTED_FEATURES,
            "removedFeatures": DEFAULT_REMOVED_FEATURES,
            "binaryLabel": DEFAULT_BINARY_LABEL,
        },
        "model": {
            "code": DEFAULT_MODEL_CODE,
            "parameters": getModelDefaults(DEFAULT_MODEL_CODE),
        },
        "preprocessing": {
            "imputer": DEFAULT_IMPUTER_NAME,
            "imputerParameters": deepcopy(DEFAULT_IMPUTER_PARAMETERS),
            "normalizer": DEFAULT_NORMALIZER_NAME,
            "normalizerParameters": deepcopy(DEFAULT_NORMALIZER_PARAMETERS),
        },
        "training": {
            "strategy": DEFAULT_TRAINING_STRATEGY,
            "parameters": deepcopy(DEFAULT_TRAINING_PARAMETERS),
        },
        "threshold": {
            "name": DEFAULT_THRESHOLD_NAME,
            "scoreSource": DEFAULT_THRESHOLD_SCORE_SOURCE,
            "scoreWindowSizes": tuple(DEFAULT_SCORE_WINDOW_SIZES),
            "parameters": deepcopy(DEFAULT_DSPOT_PARAMETERS),
        },
        "execution": {
            "initialWarmupSize": DEFAULT_INITIAL_WARMUP_SIZE,
            "metricsWindowSize": DEFAULT_METRICS_WINDOW_SIZE,
            "seed": DEFAULT_SEED,
            "generatePlots": DEFAULT_GENERATE_PLOTS,
            "strictTrainingErrors": DEFAULT_STRICT_TRAINING_ERRORS,
        },
    }


def getClassificationParameters():
    # Retorna a configuração padrão do experimento supervisionado binário sem alterar os defaults de detecção.
    parameters = getDefaultParameters()
    parameters["model"] = {
        "code": DEFAULT_CLASSIFIER_CODE,
        "parameters": getClassifierDefaults(DEFAULT_CLASSIFIER_CODE),
    }
    parameters["execution"]["initialWarmupSize"] = DEFAULT_CLASSIFICATION_WARMUP_SIZE
    parameters.pop("threshold", None)
    parameters.pop("training", None)
    return parameters
