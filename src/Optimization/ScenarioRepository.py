from dataclasses import dataclass
from pathlib import Path
import unicodedata

import pandas as pd

from src.Data.Processor import DataStreamProcessor
from src.Optimization.OptimizationConfig import DatasetProfile


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


class ScenarioRepository:
    scenarioAliases = {
        "adaptacao": "Adaptation",
        "adaptation": "Adaptation",
        "consistencia": "Consistency",
        "consistency": "Consistency",
        "generalizacao": "Generalization",
        "generalization": "Generalization",
        "recorrencia": "Recurrence",
        "recurrence": "Recurrence",
    }

    def __init__(self, dataRoot, datasetProfile=None):
        self.dataRoot = Path(dataRoot)
        self.datasetProfile = datasetProfile or DatasetProfile()

    def prepare(self, scenarios, blockSize):
        paths = self._discover(scenarios, blockSize)
        return tuple(
            self._prepareScenario(name, paths[name]) for name in scenarios
        )

    def _discover(self, scenarios, blockSize):
        if not self.dataRoot.exists():
            raise FileNotFoundError(
                f"Diretório de datasets não encontrado: {self.dataRoot.resolve()}"
            )

        expected = {str(name): [] for name in scenarios}
        suffix = f"_{int(blockSize)}"
        for path in sorted(self.dataRoot.rglob("*.csv")):
            normalizedStem = self._normalize(path.stem)
            if not normalizedStem.endswith(suffix):
                continue
            scenarioKey = normalizedStem[: -len(suffix)].rstrip("_")
            scenarioName = self.scenarioAliases.get(scenarioKey)
            if scenarioName in expected:
                expected[scenarioName].append(path)

        resolved = {}
        for name, matches in expected.items():
            if not matches:
                raise FileNotFoundError(
                    f"Dataset do cenário {name}_{blockSize} não encontrado "
                    f"em {self.dataRoot.resolve()}."
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Mais de um dataset encontrado para {name}_{blockSize}: "
                    f"{[str(path) for path in matches]}"
                )
            resolved[name] = matches[0]
        return resolved

    def _prepareScenario(self, name, path):
        dataframe = pd.read_csv(path)
        processor = DataStreamProcessor(
            logging=False,
            selected_features=(
                list(self.datasetProfile.selectedFeatures)
                if self.datasetProfile.selectedFeatures is not None
                else None
            ),
            removed_features=(
                list(self.datasetProfile.removedFeatures)
                if self.datasetProfile.removedFeatures is not None
                else None
            ),
        )
        stream, targetNames, featureNames, labelNames = processor.create_stream(
            dataframe,
            target_label_col=self.datasetProfile.targetColumn,
            binary_label=self.datasetProfile.binaryLabel,
        )
        normalizedLabels = pd.Series(labelNames).astype(str).str.upper()
        attackInstances = int(
            (~normalizedLabels.isin(["BENIGN", "NORMAL"])).sum()
        )
        totalInstances = int(len(labelNames))
        attackRatioPercent = (
            100.0 * attackInstances / totalInstances if totalInstances else 0.0
        )
        return PreparedScenario(
            name=name,
            datasetPath=path,
            datasetName=path.stem,
            stream=stream,
            targetNames=tuple(targetNames),
            featureNames=tuple(featureNames),
            labelNames=tuple(labelNames),
            totalInstances=totalInstances,
            attackInstances=attackInstances,
            attackRatioPercent=attackRatioPercent,
        )

    def _normalize(self, value):
        return (
            unicodedata.normalize("NFKD", str(value))
            .encode("ascii", "ignore")
            .decode("ascii")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
