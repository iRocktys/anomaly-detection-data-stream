"""Contratos e opções gerais; intervalos específicos vivem nos espaços de busca."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from numbers import Integral

import ProjectDefaults as defaults
# Reexports preservam os imports antigos.
from ProjectDefaults import DEFAULT_SELECTED_FEATURES, DEFAULT_REMOVED_FEATURES, DEFAULT_AIF_PARAMETERS
from src.Optimization.AifSearchSpace import AifSearchSpaceConfig
from src.Optimization.DspotSearchSpace import DspotSearchSpaceConfig


@dataclass(frozen=True)
class DatasetProfile:
    targetColumn: str = defaults.DEFAULT_TARGET_COLUMN
    selectedFeatures: tuple[str, ...] | None = defaults.DEFAULT_SELECTED_FEATURES
    removedFeatures: tuple[str, ...] | None = defaults.DEFAULT_REMOVED_FEATURES
    binaryLabel: bool = defaults.DEFAULT_BINARY_LABEL

    def __post_init__(self):
        if not str(self.targetColumn).strip():
            raise ValueError("targetColumn não pode ser vazio.")
        object.__setattr__(self, "targetColumn", str(self.targetColumn).strip())
        for name in ("selectedFeatures", "removedFeatures"):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, tuple(str(v).strip() for v in values))


@dataclass(frozen=True)
class ModelProfile:
    code: str = defaults.DEFAULT_MODEL_CODE
    parameters: dict[str, Any] | None = None

    def __post_init__(self):
        code = str(self.code).strip().upper()
        parameters = defaults.getModelDefaults(code)
        parameters.update(dict(self.parameters or {}))
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "parameters", parameters)


@dataclass(frozen=True)
class OptimizationConfig:
    dataRoot: Path = defaults.DEFAULT_DATA_ROOT
    outputRoot: Path = defaults.DEFAULT_OPTIMIZATION_OUTPUT_ROOT
    scenarios: tuple[str, ...] = defaults.DEFAULT_SCENARIOS
    blockSize: int = defaults.DEFAULT_OPTIMIZATION_BLOCK_SIZE
    nTrials: int = defaults.DEFAULT_N_TRIALS
    topK: int = defaults.DEFAULT_TOP_K
    metricsWindowSize: int = defaults.DEFAULT_OPTIMIZATION_METRICS_WINDOW_SIZE
    initialWarmupSize: int = defaults.DEFAULT_INITIAL_WARMUP_SIZE
    seed: int = defaults.DEFAULT_SEED
    optunaSeed: int = defaults.DEFAULT_OPTUNA_SEED
    optimizeModelParameters: bool = defaults.DEFAULT_OPTIMIZE_MODEL_PARAMETERS
    fixedModelParameters: dict[str, Any] = field(default_factory=dict)
    imputerNames: tuple[str, ...] = defaults.DEFAULT_IMPUTER_NAMES
    normalizerName: str = defaults.DEFAULT_NORMALIZER_NAME
    normalizerParameters: dict[str, Any] = field(default_factory=lambda: dict(defaults.DEFAULT_NORMALIZER_PARAMETERS))
    aifSearchSpace: AifSearchSpaceConfig = field(default_factory=AifSearchSpaceConfig)
    dspotSearchSpace: DspotSearchSpaceConfig = field(default_factory=DspotSearchSpaceConfig)

    def __post_init__(self):
        object.__setattr__(self, "dataRoot", Path(self.dataRoot))
        object.__setattr__(self, "outputRoot", Path(self.outputRoot))
        object.__setattr__(self, "scenarios", tuple(str(v) for v in self.scenarios))
        object.__setattr__(self, "imputerNames", tuple(self.imputerNames))
        object.__setattr__(self, "fixedModelParameters", dict(self.fixedModelParameters))
        object.__setattr__(self, "normalizerParameters", dict(self.normalizerParameters))
        for name in ("blockSize", "nTrials", "topK", "metricsWindowSize", "initialWarmupSize"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} deve ser inteiro maior que zero.")
        if not self.scenarios or len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("Informe cenários distintos para otimização.")
        if not self.imputerNames or not set(self.imputerNames) <= {"zero", "incrementalMean"}:
            raise ValueError("Imputadores aceitos: zero e incrementalMean.")
        if self.normalizerName not in ("incrementalZScore", "none"):
            raise ValueError("Normalizador aceito: incrementalZScore ou none.")
        if not isinstance(self.aifSearchSpace, AifSearchSpaceConfig) or not isinstance(self.dspotSearchSpace, DspotSearchSpaceConfig):
            raise TypeError("Utilize AifSearchSpaceConfig e DspotSearchSpaceConfig nos espaços de busca.")
        if self.dspotSearchSpace.calibrationWindowMaximum > self.initialWarmupSize:
            raise ValueError("A maior calibração do DSPOT deve caber no warm-up.")


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
        return self.driftDepth + self.calibrationSize


@dataclass(frozen=True)
class PreparedScenario:
    name: str
    datasetPath: Path
    datasetName: str
    stream: object
    targetNames: tuple[str, ...]
    featureNames: tuple[str, ...]
    labelNames: tuple[str, ...]
    totalInstances: int
    attackInstances: int
    attackRatioPercent: float


class SearchSpaceProtocol(Protocol):
    def suggest(self, trial: Any) -> TrialConfiguration:
        ...
