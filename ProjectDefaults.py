from pathlib import Path
from copy import deepcopy

DEFAULT_SELECTED_FEATURES = (
    "Max Packet Length",
    "Average Packet Size",
    "Fwd Packet Length Min",
    "Min Packet Length",
    "Fwd Packet Length Max",
    "Packet Length Mean",
    "Fwd Packet Length Mean",
    "Avg Fwd Segment Size",
    "min_seg_size_forward",
    "ACK Flag Count",
    "Flow Duration",
    "Fwd IAT Total",
    "Flow IAT Max",
    "Fwd IAT Max",
    "Flow IAT Std",
    "Fwd IAT Std",
    "Fwd IAT Mean",
    "Flow IAT Mean",
    "Total Length of Fwd Packets",
    "Subflow Fwd Bytes",
    "act_data_pkt_fwd",
    "Subflow Fwd Packets",
    "Total Fwd Packets",
    "Down/Up Ratio",
    "Init_Win_bytes_backward",
    "Total Length of Bwd Packets",
    "Subflow Bwd Bytes",
    "Flow IAT Min",
    "Bwd Packet Length Max",
    "URG Flag Count",
    "Bwd IAT Total",
    "Bwd Packets/s",
    "Init_Win_bytes_forward",
)

DEFAULT_REMOVED_FEATURES = (
    "Flow ID",
    "Timestamp",
    "SimillarHTTP",
    "Unnamed: 0",
)

DEFAULT_AIF_PARAMETERS = {
    "window_size": 1024,
    "n_trees": 100,
    "height": 11,
    "m_trees": 10,
    "weights": 0.3722,
}



DEFAULT_MODEL_CODE = "AIF"
DEFAULT_MODEL_PARAMETERS = {
    "AIF": DEFAULT_AIF_PARAMETERS,
    'HST': {'CLI': None, 'window_size': 250, 'number_of_trees': 25, 'max_depth': 15, 'anomaly_threshold': 0.5, 'size_limit': 0.1},
    'AE': {'hidden_layer': 2, 'learning_rate': 0.5, 'threshold': 0.6},
    'SRHF': {'max_height': 5, 'num_trees': 100, 'window_size': 20, 'random_seed': 0},
}
DEFAULT_TARGET_COLUMN = "Label"
DEFAULT_BINARY_LABEL = True
DEFAULT_NORMAL_CLASS_INDEX = 0
DEFAULT_DATA_ROOT = Path("data/15k")
DEFAULT_OUTPUT_ROOT = Path("output")
DEFAULT_OPTIMIZATION_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "Optimization"
DEFAULT_SCENARIOS = ("Adaptation", "Consistency", "Generalization", "Recurrence")
DEFAULT_DATASET_SCENARIOS = ("Adaptação", "Consistência", "Generalização", "Recorrência")
DEFAULT_ATTACK_SCALES = (25, 200, 1000)
DEFAULT_OPTIMIZATION_BLOCK_SIZE = 200
DEFAULT_INITIAL_WARMUP_SIZE = 2000
DEFAULT_SEED = 42
DEFAULT_OPTUNA_SEED = 42
DEFAULT_METRICS_WINDOW_SIZE = 1000
DEFAULT_OPTIMIZATION_METRICS_WINDOW_SIZE = 100
DEFAULT_N_TRIALS = 100
DEFAULT_TOP_K = 10
DEFAULT_OPTIMIZE_MODEL_PARAMETERS = True
DEFAULT_IMPUTER_NAMES = ("zero", "incrementalMean")
DEFAULT_IMPUTER_NAME = "zero"
DEFAULT_NORMALIZER_NAME = "incrementalZScore"
DEFAULT_NORMALIZER_PARAMETERS = {"epsilon": 1e-8, "clip": None}
DEFAULT_SCORE_WINDOW_SIZES = (10, 50, 100)
DEFAULT_DSPOT_SCORE_SOURCES = ("raw", "ma10", "ma50", "ma100")
DEFAULT_DSPOT_CALIBRATION_WINDOW = 1000
DEFAULT_DSPOT_DRIFT_DEPTH = 50
DEFAULT_DSPOT_PARAMETERS = {
    "driftDepth": DEFAULT_DSPOT_DRIFT_DEPTH,
    "calibrationSize": DEFAULT_DSPOT_CALIBRATION_WINDOW - DEFAULT_DSPOT_DRIFT_DEPTH,
    "initialQuantile": 0.98,
    "risk": 0.001,
    "refitEvery": 1,
    "optimizationStarts": 10,
    "tolerance": 1e-8,
}


def getModelDefaults(code=DEFAULT_MODEL_CODE):
    resolved = str(code).strip().upper()
    if resolved not in DEFAULT_MODEL_PARAMETERS:
        raise ValueError(f"Modelo sem defaults no projeto: {resolved}.")
    return deepcopy(DEFAULT_MODEL_PARAMETERS[resolved])
