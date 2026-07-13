from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComponentConfig:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def validate(self, fieldName="component"):
        if not str(self.name).strip():
            raise ValueError(f"{fieldName}.name não pode ser vazio.")

    def toDict(self):
        return {"name": self.name, "parameters": dict(self.parameters)}


@dataclass(frozen=True)
class DatasetConfig:
    path: str
    selectedFeatures: list[str]
    name: str = "dataset"
    targetLabelColumn: str = "Label"
    ignoredColumns: list[str] = field(default_factory=lambda: [
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Protocol",
        "Inbound",
    ])
    imputationMethod: str = "0"
    normalClassIndex: int = 0

    def validate(self):
        if not str(self.path).strip():
            raise ValueError("DatasetConfig.path não pode ser vazio.")
        if not self.selectedFeatures:
            raise ValueError("DatasetConfig.selectedFeatures deve possuir ao menos uma feature.")
        if not str(self.targetLabelColumn).strip():
            raise ValueError("DatasetConfig.targetLabelColumn não pode ser vazio.")

    def resolvedPath(self):
        return str(Path(self.path).expanduser())

    def toDict(self):
        value = asdict(self)
        value["path"] = self.resolvedPath()
        return value


@dataclass(frozen=True)
class ModelConfig:
    code: str
    parameters: dict[str, Any] = field(default_factory=dict)
    name: str | None = None

    def validate(self):
        if not str(self.code).strip():
            raise ValueError("ModelConfig.code não pode ser vazio.")

    def resolvedCode(self):
        return str(self.code).strip().upper()

    def resolvedName(self):
        return str(self.name).strip() if self.name else self.resolvedCode()

    def toDict(self):
        return {
            "code": self.resolvedCode(),
            "name": self.resolvedName(),
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class ThresholdEvaluationConfig:
    name: str
    threshold: ComponentConfig
    scoreSmoother: ComponentConfig = field(default_factory=lambda: ComponentConfig("none"))
    decisionStrategy: ComponentConfig = field(default_factory=lambda: ComponentConfig("threshold"))
    scoreColumn: str = "rawScore"
    metricsWindow: int = 1000
    evaluateOnlyReady: bool = True
    saveThresholdState: bool = False

    def validate(self):
        if not str(self.name).strip():
            raise ValueError("ThresholdEvaluationConfig.name não pode ser vazio.")
        self.threshold.validate("ThresholdEvaluationConfig.threshold")
        self.scoreSmoother.validate("ThresholdEvaluationConfig.scoreSmoother")
        self.decisionStrategy.validate("ThresholdEvaluationConfig.decisionStrategy")
        if not str(self.scoreColumn).strip():
            raise ValueError("ThresholdEvaluationConfig.scoreColumn não pode ser vazio.")
        if int(self.metricsWindow) < 1:
            raise ValueError("ThresholdEvaluationConfig.metricsWindow deve ser maior que zero.")

    def toDict(self):
        return {
            "name": self.name,
            "threshold": self.threshold.toDict(),
            "scoreSmoother": self.scoreSmoother.toDict(),
            "decisionStrategy": self.decisionStrategy.toDict(),
            "scoreColumn": self.scoreColumn,
            "metricsWindow": int(self.metricsWindow),
            "evaluateOnlyReady": bool(self.evaluateOnlyReady),
            "saveThresholdState": bool(self.saveThresholdState),
        }


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "output/Experiments"
    saveNormalizedFeatures: bool = False
    saveWindowMetrics: bool = True
    printSummary: bool = True

    def validate(self):
        if not str(self.directory).strip():
            raise ValueError("OutputConfig.directory não pode ser vazio.")

    def resolvedDirectory(self):
        return str(Path(self.directory).expanduser())

    def toDict(self):
        return {
            "directory": self.resolvedDirectory(),
            "saveNormalizedFeatures": bool(self.saveNormalizedFeatures),
            "saveWindowMetrics": bool(self.saveWindowMetrics),
            "printSummary": bool(self.printSummary),
        }


@dataclass(frozen=True)
class ExperimentPlan:
    datasets: list[DatasetConfig]
    models: list[ModelConfig]
    normalizers: list[ComponentConfig]
    thresholdEvaluations: list[ThresholdEvaluationConfig] = field(default_factory=list)
    featureExtractor: ComponentConfig = field(default_factory=lambda: ComponentConfig("none"))
    featureSmoother: ComponentConfig = field(default_factory=lambda: ComponentConfig("none"))
    trainingStrategy: ComponentConfig = field(default_factory=lambda: ComponentConfig("all"))
    normalizerUpdatePolicy: str = "all"
    movingAverageWindows: list[int] = field(default_factory=lambda: [3, 5, 10, 50, 100])
    runSeeds: list[int] = field(default_factory=lambda: [1])
    output: OutputConfig = field(default_factory=OutputConfig)

    def withChanges(self, **changes):
        return replace(self, **changes)

    def validate(self):
        if not self.datasets:
            raise ValueError("ExperimentPlan.datasets deve possuir ao menos um dataset.")
        if not self.models:
            raise ValueError("ExperimentPlan.models deve possuir ao menos um modelo.")
        if not self.normalizers:
            raise ValueError("ExperimentPlan.normalizers deve possuir ao menos um normalizador.")
        if not self.runSeeds:
            raise ValueError("ExperimentPlan.runSeeds deve possuir ao menos uma seed.")
        if self.normalizerUpdatePolicy not in {"all", "oracleNormal", "none"}:
            raise ValueError(
                "normalizerUpdatePolicy deve ser all, oracleNormal ou none. "
                "oracleNormal usa o rótulo verdadeiro e deve ser tratado como experimento controlado."
            )
        for dataset in self.datasets:
            dataset.validate()
        for model in self.models:
            model.validate()
        for normalizer in self.normalizers:
            normalizer.validate("normalizer")
        for evaluation in self.thresholdEvaluations:
            evaluation.validate()
        self.featureExtractor.validate("featureExtractor")
        self.featureSmoother.validate("featureSmoother")
        self.trainingStrategy.validate("trainingStrategy")
        self.output.validate()
        if any(int(window) < 1 for window in self.movingAverageWindows):
            raise ValueError("movingAverageWindows aceita somente valores maiores que zero.")

    def scorePlan(self):
        return replace(self, thresholdEvaluations=[])

    def toDict(self):
        return {
            "datasets": [dataset.toDict() for dataset in self.datasets],
            "models": [model.toDict() for model in self.models],
            "normalizers": [normalizer.toDict() for normalizer in self.normalizers],
            "thresholdEvaluations": [evaluation.toDict() for evaluation in self.thresholdEvaluations],
            "featureExtractor": self.featureExtractor.toDict(),
            "featureSmoother": self.featureSmoother.toDict(),
            "trainingStrategy": self.trainingStrategy.toDict(),
            "normalizerUpdatePolicy": self.normalizerUpdatePolicy,
            "movingAverageWindows": [int(window) for window in self.movingAverageWindows],
            "runSeeds": [int(seed) for seed in self.runSeeds],
            "output": self.output.toDict(),
        }


# Alias temporário para imports antigos. Novos experimentos devem usar ExperimentPlan.
ExperimentConfig = ExperimentPlan
