from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ProjectDefaults as defaults
from src.Experiment.Configuration import resolveDatasets, resolveParameters
from src.Optimization.AifSearchSpace import AifSearchSpaceConfig
from src.Optimization.DspotSearchSpace import DspotSearchSpaceConfig


@dataclass(frozen=True)
class OptimizationConfig:
    experimentName: str
    datasets: dict
    parameters: dict = field(default_factory=dict)
    task: str = "anomaly"
    outputRoot: Path = defaults.DEFAULT_OPTIMIZATION_OUTPUT_ROOT
    nTrials: int = defaults.DEFAULT_N_TRIALS
    optunaSeed: int = defaults.DEFAULT_OPTUNA_SEED
    optimizeModelParameters: bool = defaults.DEFAULT_OPTIMIZE_MODEL_PARAMETERS
    maxTrialAttemptsMultiplier: int = defaults.DEFAULT_MAX_TRIAL_ATTEMPTS_MULTIPLIER
    fixedModelParameters: dict[str, Any] = field(default_factory=dict)
    aifSearchSpace: AifSearchSpaceConfig = field(default_factory=AifSearchSpaceConfig)
    dspotSearchSpace: DspotSearchSpaceConfig = field(default_factory=DspotSearchSpaceConfig)

    def __post_init__(self):
        # Valida a configuração mantendo compatibilidade com o formato original do notebook de otimização.
        experiment_name = str(self.experimentName).strip()
        task = str(self.task).strip().lower()
        if not experiment_name:
            raise ValueError("experimentName must be provided explicitly.")
        if task not in {"anomaly", "classification"}:
            raise ValueError("Task must be 'anomaly' or 'classification'.")
        if not isinstance(self.datasets, dict) or not self.datasets:
            raise ValueError("datasets must be a non-empty dictionary.")
        if int(self.nTrials) < 1:
            raise ValueError("nTrials must be greater than zero.")
        if int(self.maxTrialAttemptsMultiplier) < 1:
            raise ValueError("maxTrialAttemptsMultiplier must be greater than zero.")
        object.__setattr__(self, "experimentName", experiment_name)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "outputRoot", Path(self.outputRoot))
        object.__setattr__(self, "parameters", dict(self.parameters or {}))
        object.__setattr__(self, "fixedModelParameters", dict(self.fixedModelParameters or {}))
        if task == "anomaly":
            self._validateWarmupProtocol()

    def _validateWarmupProtocol(self):
        # Garante o protocolo fixo de 500 instâncias do AIF e 500 scores de calibração do DSPOT.
        parameters = self.effectiveParameters
        total = int(parameters["execution"]["initialWarmupSize"])
        dspot = int(self.dspotSearchSpace.calibrationWindow)
        model = total - dspot
        if (
            total != defaults.DEFAULT_INITIAL_WARMUP_SIZE
            or dspot != defaults.DEFAULT_DSPOT_CALIBRATION_WINDOW
            or model != defaults.DEFAULT_AIF_WARMUP_SIZE
        ):
            raise ValueError("Anomaly optimization requires 500 model instances and 500 DSPOT calibration scores.")

    @property
    def effectiveParameters(self):
        # Resolve os parâmetros informados sobre os defaults específicos da tarefa escolhida.
        return resolveParameters(self.parameters, task=self.task)

    @property
    def datasetConfigs(self):
        # Converte os datasets do notebook no mesmo contrato utilizado pelo treinamento normal.
        return resolveDatasets(self.datasets, self.effectiveParameters, task=self.task)


@dataclass(frozen=True)
class TrialConfiguration:
    imputerName: str
    thresholdScoreSource: str
    scoreWindowSizes: tuple[int, ...]
    driftDepth: int
    calibrationSize: int
    initialQuantile: float
    risk: float
    refitEvery: int
    optimizationStarts: int
    tolerance: float
    modelParameters: dict[str, Any] = field(default_factory=dict)

    @property
    def calibrationWindow(self):
        # Retorna o total de scores usados pela etapa de calibração do DSPOT.
        return self.driftDepth + self.calibrationSize


@dataclass(frozen=True)
class PreparedDataset:
    name: str
    datasetPath: Path
    stream: object
    targetNames: tuple[str, ...]
    featureNames: tuple[str, ...]
    labelNames: tuple[str, ...]
    totalInstances: int
    attackInstances: int
    attackRatioPercent: float
    parameters: dict[str, Any] = field(default_factory=dict)
