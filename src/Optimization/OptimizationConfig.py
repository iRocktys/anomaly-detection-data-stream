from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ProjectDefaults as defaults
from src.Experiment.Configuration import (
    resolveDatasets,
    resolveParameters,
)
from src.Optimization.AifSearchSpace import AifSearchSpaceConfig
from src.Optimization.DspotSearchSpace import DspotSearchSpaceConfig


@dataclass(frozen=True)
class OptimizationConfig:
    experimentName: str
    datasets: dict
    parameters: dict = field(
        default_factory=dict
    )
    outputRoot: Path = defaults.DEFAULT_OPTIMIZATION_OUTPUT_ROOT
    nTrials: int = defaults.DEFAULT_N_TRIALS
    topK: int = defaults.DEFAULT_TOP_K
    optunaSeed: int = defaults.DEFAULT_OPTUNA_SEED
    studyVersion: str = defaults.DEFAULT_OPTUNA_STUDY_VERSION
    optimizeModelParameters: bool = defaults.DEFAULT_OPTIMIZE_MODEL_PARAMETERS
    maxTrialAttemptsMultiplier: int = defaults.DEFAULT_MAX_TRIAL_ATTEMPTS_MULTIPLIER
    fixedModelParameters: dict[str, Any] = field(
        default_factory=dict
    )
    aifSearchSpace: AifSearchSpaceConfig = field(
        default_factory=AifSearchSpaceConfig
    )
    dspotSearchSpace: DspotSearchSpaceConfig = field(
        default_factory=DspotSearchSpaceConfig
    )

    def __post_init__(self):
        # Valida a otimização genérica e mantém nome do experimento e versão explicitamente controlados.
        if not str(
            self.experimentName
        ).strip():
            raise ValueError(
                "experimentName deve ser informado explicitamente."
            )

        if (
            not isinstance(
                self.datasets,
                dict,
            )
            or not self.datasets
        ):
            raise ValueError(
                "datasets deve ser um dicionário não vazio."
            )

        if (
            int(
                self.nTrials
            )
            < 1
            or int(
                self.topK
            )
            < 1
        ):
            raise ValueError(
                "nTrials e topK devem ser maiores que zero."
            )

        if int(
            self.maxTrialAttemptsMultiplier
        ) < 1:
            raise ValueError(
                "maxTrialAttemptsMultiplier deve ser maior que zero."
            )

        if not str(
            self.studyVersion
        ).strip():
            raise ValueError(
                "studyVersion deve ser informado."
            )

        object.__setattr__(
            self,
            "experimentName",
            str(
                self.experimentName
            ).strip(),
        )

        object.__setattr__(
            self,
            "studyVersion",
            str(
                self.studyVersion
            ).strip(),
        )

        object.__setattr__(
            self,
            "outputRoot",
            Path(
                self.outputRoot
            ),
        )

        object.__setattr__(
            self,
            "parameters",
            dict(
                self.parameters
                or {}
            ),
        )

        object.__setattr__(
            self,
            "fixedModelParameters",
            dict(
                self.fixedModelParameters
                or {}
            ),
        )

        self._validateWarmupProtocol()

    def _validateWarmupProtocol(self):
        # Garante que a otimização use exatamente 500 instâncias para AIF e 500 para DSPOT.
        parameters = self.effectiveParameters

        total = int(
            parameters[
                "execution"
            ][
                "initialWarmupSize"
            ]
        )

        dspot = int(
            self.dspotSearchSpace.calibrationWindow
        )

        model = (
            total
            - dspot
        )

        if (
            total
            != defaults.DEFAULT_INITIAL_WARMUP_SIZE
            or dspot
            != defaults.DEFAULT_DSPOT_CALIBRATION_WINDOW
            or model
            != defaults.DEFAULT_AIF_WARMUP_SIZE
        ):
            raise ValueError(
                "O protocolo de otimização exige 500 instâncias para o AIF "
                "e 500 para o DSPOT, totalizando initialWarmupSize=1000."
            )

    @property
    def effectiveParameters(self):
        # Retorna os parâmetros resolvidos sobre os defaults técnicos do projeto.
        return resolveParameters(
            self.parameters
        )

    @property
    def datasetConfigs(self):
        # Converte os datasets explícitos em contratos usados pela otimização.
        return resolveDatasets(
            self.datasets,
            self.effectiveParameters,
        )


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
    modelParameters: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def calibrationWindow(self):
        # Retorna a janela total do DSPOT formada por driftDepth e calibrationSize.
        return (
            self.driftDepth
            + self.calibrationSize
        )


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
