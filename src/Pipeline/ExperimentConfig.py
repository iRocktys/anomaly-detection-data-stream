from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from src.Anomaly.Thresholds.ThresholdRegistry import ThresholdRegistry


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
        ThresholdRegistry.validateConfig(self.threshold)
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
    warmup: int = 200
    movingAverageWindows: list[int] = field(default_factory=lambda: [3, 5, 10, 50, 100])
    runSeeds: list[int] = field(default_factory=lambda: [1])
    output: OutputConfig = field(default_factory=OutputConfig)

    trainingAliases = {
        "all": "all",
        "trainall": "all",
        "predictednormal": "predictedNormal",
        "normalprediction": "predictedNormal",
        "predictednormalonly": "predictedNormal",
    }

    def withChanges(self, **changes):
        return replace(self, **changes)

    @staticmethod
    def normalizeComponentName(name):
        return "".join(
            character
            for character in str(name or "").lower()
            if character.isalnum()
        )

    def resolvedTrainingName(self):
        normalizedName = self.normalizeComponentName(self.trainingStrategy.name)
        if normalizedName not in self.trainingAliases:
            available = "all, predictedNormal"
            raise ValueError(
                f"Estratégia de treinamento desconhecida: {self.trainingStrategy.name}. "
                f"Disponíveis: {available}."
            )
        return self.trainingAliases[normalizedName]

    def trainingRequiresPrediction(self):
        return self.resolvedTrainingName() == "predictedNormal"

    def resolveTrainingEvaluation(self):
        if not self.trainingRequiresPrediction():
            return None
        if not self.thresholdEvaluations:
            raise ValueError(
                "O treinamento predictedNormal requer ao menos uma avaliação de threshold."
            )

        requestedName = self.trainingStrategy.parameters.get("evaluationName")
        if requestedName is not None:
            matches = [
                evaluation
                for evaluation in self.thresholdEvaluations
                if str(evaluation.name) == str(requestedName)
            ]
            if not matches:
                available = ", ".join(
                    evaluation.name for evaluation in self.thresholdEvaluations
                )
                raise ValueError(
                    f"Avaliação de treinamento não encontrada: {requestedName}. "
                    f"Disponíveis: {available}."
                )
            return matches[0]

        if len(self.thresholdEvaluations) != 1:
            raise ValueError(
                "Com múltiplas avaliações, predictedNormal requer o parâmetro "
                "evaluationName para indicar qual threshold controla o treinamento."
            )
        return self.thresholdEvaluations[0]

    def validate(self):
        if not self.datasets:
            raise ValueError("ExperimentPlan.datasets deve possuir ao menos um dataset.")
        if not self.models:
            raise ValueError("ExperimentPlan.models deve possuir ao menos um modelo.")
        if not self.normalizers:
            raise ValueError("ExperimentPlan.normalizers deve possuir ao menos um normalizador.")
        if not self.runSeeds:
            raise ValueError("ExperimentPlan.runSeeds deve possuir ao menos uma seed.")
        if int(self.warmup) < 20:
            raise ValueError("ExperimentPlan.warmup deve ser maior ou igual a 20.")
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
        self.resolvedTrainingName()
        if self.trainingRequiresPrediction():
            self.resolveTrainingEvaluation()
        self.output.validate()
        if any(int(window) < 1 for window in self.movingAverageWindows):
            raise ValueError("movingAverageWindows aceita somente valores maiores que zero.")

    def scorePlan(self):
        if self.trainingRequiresPrediction():
            return self
        return replace(self, thresholdEvaluations=[])

    def toDict(self):
        feedbackEvaluation = self.resolveTrainingEvaluation()
        return {
            "datasets": [dataset.toDict() for dataset in self.datasets],
            "models": [model.toDict() for model in self.models],
            "normalizers": [normalizer.toDict() for normalizer in self.normalizers],
            "thresholdEvaluations": [evaluation.toDict() for evaluation in self.thresholdEvaluations],
            "featureExtractor": self.featureExtractor.toDict(),
            "featureSmoother": self.featureSmoother.toDict(),
            "trainingStrategy": self.trainingStrategy.toDict(),
            "trainingFeedbackEvaluation": (
                feedbackEvaluation.name if feedbackEvaluation is not None else None
            ),
            "normalizerUpdatePolicy": self.normalizerUpdatePolicy,
            "warmup": int(self.warmup),
            "movingAverageWindows": [int(window) for window in self.movingAverageWindows],
            "runSeeds": [int(seed) for seed in self.runSeeds],
            "output": self.output.toDict(),
        }


ExperimentConfig = ExperimentPlan
