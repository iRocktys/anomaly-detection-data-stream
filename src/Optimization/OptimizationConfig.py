from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ProjectDefaults as defaults
from src.Experiment.Configuration import resolveDatasets, resolveParameters
from src.Experiment.Task import normalizeTask
from src.Optimization.AifSearchSpace import AifSearchSpaceConfig
from src.Optimization.DspotSearchSpace import DspotSearchSpaceConfig


@dataclass(frozen=True)
class OptimizationConfig:
    experimentName: str
    datasets: dict
    task: str = "anomaly"
    parameters: dict = field(default_factory=dict)
    outputRoot: Path = defaults.DEFAULT_OPTIMIZATION_OUTPUT_ROOT
    nTrials: int = defaults.DEFAULT_N_TRIALS
    optunaSeed: int = defaults.DEFAULT_OPTUNA_SEED
    optimizeModelParameters: bool = defaults.DEFAULT_OPTIMIZE_MODEL_PARAMETERS
    maxTrialAttemptsMultiplier: int = defaults.DEFAULT_MAX_TRIAL_ATTEMPTS_MULTIPLIER
    fixedModelParameters: dict[str, Any] = field(default_factory=dict)
    aifSearchSpace: AifSearchSpaceConfig = field(default_factory=AifSearchSpaceConfig)
    dspotSearchSpace: DspotSearchSpaceConfig = field(default_factory=DspotSearchSpaceConfig)

    def __post_init__(self):
        # Valida a configuração comum da otimização binária para detecção ou classificação.
        experiment_name = str(self.experimentName).strip()
        task = normalizeTask(self.task)
        if not experiment_name:
            raise ValueError("experimentName must be provided explicitly.")
        if not isinstance(self.datasets, dict) or not self.datasets:
            raise ValueError("datasets must be a non-empty dictionary.")
        if int(self.nTrials) < 1 or int(self.maxTrialAttemptsMultiplier) < 1:
            raise ValueError("nTrials and maxTrialAttemptsMultiplier must be positive.")
        object.__setattr__(self, "experimentName", experiment_name)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "outputRoot", Path(self.outputRoot))
        object.__setattr__(self, "parameters", dict(self.parameters or {}))
        object.__setattr__(self, "fixedModelParameters", dict(self.fixedModelParameters or {}))
        if task == "anomaly":
            self.validateAnomalyWarmup()

    @property
    def effectiveParameters(self):
        # Resolve os parâmetros fornecidos sobre os defaults específicos da tarefa.
        return resolveParameters(self.parameters, task=self.task)

    @property
    def datasetConfigs(self):
        # Converte o dicionário de datasets no mesmo contrato utilizado pelo treinamento normal.
        return resolveDatasets(self.datasets, self.effectiveParameters, task=self.task)

    def validateAnomalyWarmup(self):
        # Garante o protocolo fixo de 500 amostras para o detector e 500 para o DSPOT.
        warmup = int(self.effectiveParameters["execution"]["initialWarmupSize"])
        calibration = int(self.dspotSearchSpace.calibrationWindow)
        if warmup != defaults.DEFAULT_ANOMALY_WARMUP_SIZE or calibration != defaults.DEFAULT_DSPOT_CALIBRATION_WINDOW:
            raise ValueError("Anomaly optimization requires a total warm-up of 1000 and a DSPOT calibration of 500.")

