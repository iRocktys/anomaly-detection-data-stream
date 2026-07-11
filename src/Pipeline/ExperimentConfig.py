from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class ComponentConfig:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    datasetPath: str
    selectedFeatures: list[str]
    datasetName: str = "dataset"
    outputDirectory: str = "output/ExpNormalizers"
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
    modelCodes: list[str] = field(default_factory=lambda: ["AIF", "HST"])
    normalizers: list[ComponentConfig] = field(default_factory=list)
    featureSelector: ComponentConfig = field(default_factory=lambda: ComponentConfig("selected"))
    featureExtractor: ComponentConfig = field(default_factory=lambda: ComponentConfig("none"))
    featureSmoother: ComponentConfig = field(default_factory=lambda: ComponentConfig("none"))
    scoreSmoother: ComponentConfig = field(default_factory=lambda: ComponentConfig("none"))
    thresholdStrategy: ComponentConfig = field(default_factory=lambda: ComponentConfig("fixed", {"value": 0.5}))
    decisionStrategy: ComponentConfig = field(default_factory=lambda: ComponentConfig("threshold"))
    trainingStrategy: ComponentConfig = field(default_factory=lambda: ComponentConfig("all"))
    normalizerUpdatePolicy: str = "all"
    movingAverageWindows: list[int] = field(default_factory=lambda: [3, 5, 10, 50, 100])
    modelParameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    runSeed: int = 1
    saveNormalizedFeatures: bool = True
    printSummary: bool = True


    def withChanges(self, **changes):
        return replace(self, **changes)

    def validate(self):
        if not self.datasetPath:
            raise ValueError("datasetPath não pode ser vazio.")
        if not self.selectedFeatures:
            raise ValueError("selectedFeatures deve possuir pelo menos uma feature.")
        if not self.modelCodes:
            raise ValueError("modelCodes deve possuir pelo menos um modelo.")
        if not self.normalizers:
            raise ValueError("normalizers deve possuir pelo menos um normalizador.")
        if self.normalizerUpdatePolicy not in {"all", "normalOnly", "none"}:
            raise ValueError("normalizerUpdatePolicy deve ser all, normalOnly ou none.")
