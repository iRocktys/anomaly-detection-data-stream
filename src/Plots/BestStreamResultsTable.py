import re
import unicodedata
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

class BestStreamResultsTable:
    REQUIRED_COLUMNS = {
        "dataset",
        "model",
        "imputer",
        "normalizer",
        "trainingStrategy",
        "thresholdStrategy",
        "thresholdScoreSource",
        "thresholdScoreLabel",
        "attackRatioPercent",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    }

    SCENARIO_ORDER = {
        "Consistency": 0,
        "Generalization": 1,
        "Adaptation": 2,
        "Recurrence": 3,
    }

    BLOCK_ORDER = {
        25: 0,
        200: 1,
        1000: 2,
    }

    def __init__(
        self,
        resultsDirectory="output",
        outputDirectory="output/Best Results",
        model="AIF",
        trainingStrategy="all",
        featureCount=33,
        filename="Best_Stream_Results.png",
        title=None,
        dpi=300,
    ):
        self.resultsDirectory = Path(resultsDirectory)
        self.outputDirectory = Path(outputDirectory)
        self.model = (
            str(model).strip().upper()
            if model is not None
            else None
        )
        self.trainingStrategy = (
            self.normalizeToken(trainingStrategy)
            if trainingStrategy is not None
            else None
        )
        self.featureCount = (
            int(featureCount)
            if featureCount is not None
            else None
        )
        self.filename = str(filename)
        self.title = title or self.buildDefaultTitle()
        self.dpi = int(dpi)

        self.headerColor = "#3f456c"
        self.evenRowColor = "#f7f7f7"
        self.gridColor = "#777777"

    def buildDefaultTitle(self):
        modelName = self.model or "Anomaly Detection"
        return f"Best {modelName} Stream Results by Scenario"

    def discoverMetricFiles(self):
        if not self.resultsDirectory.exists():
            raise FileNotFoundError(
                "Diretório de resultados não encontrado: "
                f"{self.resultsDirectory.resolve()}"
            )

        metricFiles = []

        for path in self.resultsDirectory.rglob(
            "stream_metrics.csv"
        ):
            if not path.is_file():
                continue

            if self.model is not None:
                try:
                    pathModel = path.parent.parent.parent.name.upper()
                except IndexError:
                    continue

                if pathModel != self.model:
                    continue

            metricFiles.append(path)

        return sorted(
            metricFiles,
            key=lambda path: str(path).lower(),
        )

    def readCandidate(self, path):
        frame = pd.read_csv(path)

        if len(frame) != 1:
            raise ValueError(
                f"{path} deve possuir exatamente uma linha de métricas, "
                f"mas possui {len(frame)}."
            )

        missingColumns = sorted(
            self.REQUIRED_COLUMNS
            - set(frame.columns)
        )

        if missingColumns:
            raise ValueError(
                f"{path} não contém as colunas obrigatórias: "
                f"{missingColumns}"
            )

        row = frame.iloc[0]
        runDirectory = path.parent
        scenarioDirectory = runDirectory.parent
        modelDirectory = scenarioDirectory.parent

        scenarioName = self.normalizeScenarioName(
            scenarioDirectory.name
        )
        datasetScenario = self.normalizeScenarioName(
            row["dataset"]
        )

        if scenarioName != datasetScenario:
            raise ValueError(
                "O cenário registrado no CSV não corresponde à pasta: "
                f"CSV={datasetScenario}, pasta={scenarioName}, arquivo={path}"
            )

        csvModel = str(row["model"]).strip().upper()
        pathModel = modelDirectory.name.strip().upper()

        if csvModel != pathModel:
            raise ValueError(
                "O modelo registrado no CSV não corresponde à pasta: "
                f"CSV={csvModel}, pasta={pathModel}, arquivo={path}"
            )

        normalizedTraining = self.normalizeToken(
            row["trainingStrategy"]
        )

        if (
            self.trainingStrategy is not None
            and normalizedTraining
            != self.trainingStrategy
        ):
            return None

        runMatch = re.match(
            r"^(\d+)(?:-|$)",
            runDirectory.name,
        )
        runNumber = (
            int(runMatch.group(1))
            if runMatch
            else -1
        )
        runLabel = (
            f"{runNumber:03d}"
            if runNumber >= 0
            else runDirectory.name
        )

        numericValues = {}

        for column in [
            "attackRatioPercent",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ]:
            numericValue = pd.to_numeric(
                pd.Series([row[column]]),
                errors="coerce",
            ).iloc[0]

            if pd.isna(numericValue):
                raise ValueError(
                    f"Valor inválido na coluna '{column}' de {path}."
                )

            numericValues[column] = float(
                numericValue
            )

        return {
            "Scenario": scenarioName,
            "Attack Ratio (%)": numericValues[
                "attackRatioPercent"
            ],
            "Model": csvModel,
            "Training": self.formatTraining(
                row["trainingStrategy"]
            ),
            "Threshold": str(
                row["thresholdStrategy"]
            ).strip().upper(),
            "Score Source": self.formatScoreSource(
                row["thresholdScoreLabel"],
                row["thresholdScoreSource"],
            ),
            "Imputer": self.formatImputer(
                row["imputer"]
            ),
            "F1-Score": numericValues["f1"],
            "Precision": numericValues[
                "precision"
            ],
            "Recall": numericValues["recall"],
            "FN": int(round(numericValues["fn"])),
            "FP": int(round(numericValues["fp"])),
            "Run": runLabel,
            "_runNumber": runNumber,
            "_trainingKey": normalizedTraining,
            "_thresholdKey": self.normalizeToken(
                row["thresholdStrategy"]
            ),
            "_scoreSourceKey": self.normalizeToken(
                row["thresholdScoreSource"]
            ),
            "_imputerKey": self.normalizeToken(
                row["imputer"]
            ),
            "_normalizerKey": self.normalizeToken(
                row["normalizer"]
            ),
            "_sourcePath": str(path),
        }

    def loadCandidates(self):
        metricFiles = self.discoverMetricFiles()

        if not metricFiles:
            modelMessage = (
                f" para o modelo {self.model}"
                if self.model
                else ""
            )
            raise FileNotFoundError(
                "Nenhum stream_metrics.csv foi encontrado"
                f"{modelMessage} em "
                f"{self.resultsDirectory.resolve()}."
            )

        candidates = []

        for path in metricFiles:
            candidate = self.readCandidate(path)

            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            raise ValueError(
                "Foram encontrados arquivos de métricas, mas nenhum atende "
                "aos filtros informados."
            )

        return pd.DataFrame(candidates)

    def keepLatestConfigurationRuns(
        self,
        candidates,
    ):
        configurationColumns = [
            "Scenario",
            "Model",
            "_trainingKey",
            "_thresholdKey",
            "_scoreSourceKey",
            "_imputerKey",
            "_normalizerKey",
        ]

        ordered = candidates.sort_values(
            ["_runNumber", "_sourcePath"],
            ascending=[True, True],
            kind="mergesort",
        )

        return (
            ordered.groupby(
                configurationColumns,
                dropna=False,
                sort=False,
            )
            .tail(1)
            .reset_index(drop=True)
        )

    def selectBestByScenario(self, candidates):
        if candidates.empty:
            raise ValueError(
                "Não existem resultados para comparar."
            )

        ordered = candidates.sort_values(
            [
                "Scenario",
                "F1-Score",
                "Recall",
                "Precision",
                "FN",
                "FP",
                "_runNumber",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                True,
                True,
                False,
            ],
            kind="mergesort",
        )

        best = (
            ordered.groupby(
                "Scenario",
                sort=False,
                dropna=False,
            )
            .head(1)
            .copy()
        )

        best["_scenarioOrder"] = best[
            "Scenario"
        ].apply(self.scenarioSortKey)

        return (
            best.sort_values(
                "_scenarioOrder",
                kind="mergesort",
            )
            .drop(
                columns=["_scenarioOrder"],
                errors="ignore",
            )
            .reset_index(drop=True)
        )

    def buildDisplayFrame(self, bestResults):
        displayColumns = [
            "Scenario",
            "Attack Ratio (%)",
            "Model",
            "Training",
            "Threshold",
            "Score Source",
            "Imputer",
            "F1-Score",
            "Precision",
            "Recall",
            "FN",
            "FP",
            "Run",
        ]

        displayFrame = bestResults[
            displayColumns
        ].copy()

        displayFrame["Attack Ratio (%)"] = (
            displayFrame["Attack Ratio (%)"]
            .map(lambda value: f"{float(value):.2f}")
        )

        for column in [
            "F1-Score",
            "Precision",
            "Recall",
        ]:
            displayFrame[column] = (
                displayFrame[column]
                .map(lambda value: f"{float(value):.4f}")
            )

        for column in ["FN", "FP"]:
            displayFrame[column] = (
                displayFrame[column]
                .map(lambda value: str(int(value)))
            )

        return displayFrame

    def savePng(self, displayFrame):
        if displayFrame.empty:
            raise ValueError(
                "A tabela final está vazia."
            )

        self.outputDirectory.mkdir(
            parents=True,
            exist_ok=True,
        )
        outputPath = (
            self.outputDirectory
            / self.filename
        )

        figureHeight = max(
            3.0,
            0.34 * len(displayFrame) + 1.5,
        )
        figure, axis = plt.subplots(
            figsize=(20, figureHeight),
        )
        axis.axis("off")

        columnWidths = [
            0.125,
            0.075,
            0.050,
            0.060,
            0.065,
            0.080,
            0.095,
            0.070,
            0.070,
            0.070,
            0.045,
            0.045,
            0.045,
        ]

        table = axis.table(
            cellText=displayFrame.values,
            colLabels=displayFrame.columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
            colWidths=columnWidths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.55)

        for (rowIndex, _), cell in (
            table.get_celld().items()
        ):
            cell.set_edgecolor(self.gridColor)
            cell.set_linewidth(0.45)
            cell.PAD = 0.04

            if rowIndex == 0:
                cell.set_facecolor(
                    self.headerColor
                )
                cell.set_text_props(
                    weight="bold",
                    color="white",
                )

            elif rowIndex % 2 == 0:
                cell.set_facecolor(
                    self.evenRowColor
                )

        subtitleParts = [
            "Single stream result per scenario",
            "no averaging",
        ]

        if self.featureCount is not None:
            subtitleParts.append(
                f"{self.featureCount} selected features"
            )

        axis.set_title(
            self.title
            + "\n"
            + " | ".join(subtitleParts),
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        figure.tight_layout()
        figure.savefig(
            outputPath,
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)

        return str(outputPath)

    def create(self):
        candidates = self.loadCandidates()
        currentCandidates = (
            self.keepLatestConfigurationRuns(
                candidates
            )
        )
        bestResults = self.selectBestByScenario(
            currentCandidates
        )
        displayFrame = self.buildDisplayFrame(
            bestResults
        )
        pngPath = self.savePng(displayFrame)

        print(
            "Arquivos válidos encontrados: "
            f"{len(candidates)}"
        )
        print(
            "Configurações atuais comparadas: "
            f"{len(currentCandidates)}"
        )
        print(
            "Cenários selecionados: "
            f"{len(displayFrame)}"
        )
        print(f"PNG salvo em: {pngPath}")

        return displayFrame, pngPath

    def normalizeScenarioName(self, value):
        scenarioStem = Path(
            str(value).strip()
        ).stem
        asciiName = (
            unicodedata.normalize(
                "NFKD",
                scenarioStem,
            )
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        safeName = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            asciiName,
        ).strip("_")

        parts = safeName.rsplit("_", 1)
        scenarioKey = parts[0].lower()
        scenarioNames = {
            "adaptacao": "Adaptation",
            "adaptation": "Adaptation",
            "consistencia": "Consistency",
            "consistency": "Consistency",
            "generalizacao": "Generalization",
            "generalization": "Generalization",
            "recorrencia": "Recurrence",
            "recurrence": "Recurrence",
        }

        scenarioName = scenarioNames.get(
            scenarioKey,
            parts[0],
        )

        if len(parts) == 1:
            return scenarioName

        return f"{scenarioName}_{parts[1]}"

    def scenarioSortKey(self, scenarioName):
        match = re.match(
            r"^([^_]+)_(\d+)$",
            str(scenarioName),
        )

        if not match:
            return (99, 99, str(scenarioName))

        scenario, blockText = match.groups()
        block = int(blockText)

        return (
            self.SCENARIO_ORDER.get(
                scenario,
                99,
            ),
            self.BLOCK_ORDER.get(
                block,
                block,
            ),
            str(scenarioName),
        )

    def normalizeToken(self, value):
        return (
            str(value)
            .strip()
            .lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

    def formatTraining(self, value):
        normalized = self.normalizeToken(value)

        if normalized == "all":
            return "ALL"

        if normalized in [
            "belowthreshold",
            "predicttrue",
        ]:
            return "PREDICT_TRUE"

        return str(value).strip().upper()

    def formatScoreSource(
        self,
        label,
        source,
    ):
        labelText = str(label).strip()

        if labelText and labelText.lower() not in [
            "nan",
            "none",
        ]:
            return labelText

        normalized = self.normalizeToken(source)

        if normalized == "raw":
            return "Raw"

        match = re.search(r"(\d+)$", normalized)

        if match:
            return f"MA {match.group(1)}"

        return str(source).strip()

    def formatImputer(self, value):
        normalized = self.normalizeToken(value)

        if normalized == "zero":
            return "Zero"

        if normalized in [
            "incrementalmean",
            "mean",
            "media",
        ]:
            return "Incremental mean"

        return str(value).strip()
