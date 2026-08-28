from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True)
class DatasetProfile:
    targetColumn: str = "Label"
    selectedFeatures: tuple[str, ...] | None = DEFAULT_SELECTED_FEATURES
    removedFeatures: tuple[str, ...] | None = DEFAULT_REMOVED_FEATURES
    binaryLabel: bool = True


@dataclass(frozen=True)
class ModelProfile:
    code: str = "AIF"
    parameters: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_AIF_PARAMETERS)
    )


@dataclass(frozen=True)
class OptimizationConfig:
    dataRoot: Path = Path("data/15k")
    outputRoot: Path = Path("output/Optimization")
    scenarios: tuple[str, ...] = (
        "Adaptation",
        "Consistency",
        "Generalization",
        "Recurrence",
    )
    blockSize: int = 200
    nTrials: int = 100
    topK: int = 10
    metricsWindowSize: int = 100
    initialWarmupSize: int = 2000
    seed: int = 42
    optunaSeed: int = 42

    def __post_init__(self):
        object.__setattr__(self, "dataRoot", Path(self.dataRoot))
        object.__setattr__(self, "outputRoot", Path(self.outputRoot))
        object.__setattr__(
            self,
            "scenarios",
            tuple(str(name) for name in self.scenarios),
        )
        if self.blockSize < 1:
            raise ValueError("blockSize deve ser maior que zero.")
        if self.nTrials < 1:
            raise ValueError("nTrials deve ser maior que zero.")
        if self.topK < 1:
            raise ValueError("topK deve ser maior que zero.")
        if self.metricsWindowSize < 1:
            raise ValueError("metricsWindowSize deve ser maior que zero.")
        if self.initialWarmupSize < 1:
            raise ValueError("initialWarmupSize deve ser maior que zero.")
        if not self.scenarios:
            raise ValueError("Informe ao menos um cenário para otimização.")


@dataclass(frozen=True)
class DspotSearchSpaceConfig:
    imputerNames: tuple[str, ...] = ("zero", "incrementalMean")
    scoreModes: tuple[str, ...] = ("raw", "movingAverage")
    movingAverageMinimum: int = 2
    movingAverageMaximum: int = 200
    riskMinimum: float = 1e-5
    riskMaximum: float = 1e-2
    initialQuantileMinimum: float = 0.90
    initialQuantileMaximum: float = 0.99
    calibrationWindowMinimum: int = 500
    calibrationWindowMaximum: int = 1500
    calibrationWindowStep: int = 50
    driftDepthMinimum: int = 20
    driftDepthMaximum: int = 200
    driftDepthStep: int = 5
    refitEveryMinimum: int = 1
    refitEveryMaximum: int = 25
    optimizationStarts: int = 10
    tolerance: float = 1e-8

    def __post_init__(self):
        if not self.imputerNames or not self.scoreModes:
            raise ValueError(
                "O espaço de busca deve possuir imputadores e fontes de score."
            )
        if self.movingAverageMinimum < 1 or (
            self.movingAverageMaximum < self.movingAverageMinimum
        ):
            raise ValueError("Intervalo de média móvel inválido.")
        if not (
            0.0 < self.riskMinimum <= self.riskMaximum < 1.0
        ):
            raise ValueError("Intervalo de risk inválido.")
        if not (
            0.5
            < self.initialQuantileMinimum
            <= self.initialQuantileMaximum
            < 1.0
        ):
            raise ValueError("Intervalo de initialQuantile inválido.")
        if self.calibrationWindowStep < 1 or self.driftDepthStep < 1:
            raise ValueError("Os passos das janelas devem ser positivos.")
        if (
            self.calibrationWindowMaximum
            < self.calibrationWindowMinimum
            or self.driftDepthMaximum < self.driftDepthMinimum
        ):
            raise ValueError("Intervalos de calibração ou drift inválidos.")
        if self.calibrationWindowMinimum - self.driftDepthMaximum < 20:
            raise ValueError(
                "A menor janela de calibração deve preservar ao menos 20 "
                "valores após o driftDepth máximo."
            )
        if self.refitEveryMinimum < 1 or (
            self.refitEveryMaximum < self.refitEveryMinimum
        ):
            raise ValueError("Intervalo de refitEvery inválido.")
        if self.optimizationStarts < 2 or self.tolerance <= 0:
            raise ValueError(
                "optimizationStarts e tolerance possuem valores inválidos."
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

    @property
    def calibrationWindow(self):
        return self.driftDepth + self.calibrationSize
