import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.Pipeline.ResultContracts import PipelineRunContext
from src.Pipeline.ResultFrameBuilder import ResultFrameBuilder
from src.Plots.Plots import Plots


class ResultManager:
    def __init__(self, outputPath="output"):
        self.outputPath = Path(outputPath)
        self.frameBuilder = ResultFrameBuilder()
        self.context = None
        self.rows = []

    def start(self, context: PipelineRunContext):
        self.context = context
        self.rows = []

    def collect(self, row):
        if self.context is None:
            raise RuntimeError(
                "O gerenciador de resultados deve ser iniciado antes da coleta."
            )
        self.rows.append(row)

    def finish(self):
        if self.context is None:
            raise RuntimeError("Nenhuma execução foi iniciada.")

        context = self.context
        rows = self.rows
        self.context = None
        self.rows = []
        return self.save(
            rows=rows,
            datasetName=context.datasetName,
            modelCode=context.modelCode,
            windowSize=context.metricsWindowSize,
            movingAverageColumns=context.movingAverageColumns,
            generatePlots=context.generatePlots,
        )

    def save(
        self,
        rows,
        datasetName,
        modelCode,
        windowSize,
        movingAverageColumns=None,
        generatePlots=True,
    ):
        if not rows:
            raise ValueError("Não existem resultados para salvar.")

        instanceFrame = pd.DataFrame(rows)
        trainingName = self.normalizeTrainingName(
            instanceFrame["trainingStrategy"].iloc[0]
        )
        thresholdName = self.normalizeThresholdName(
            instanceFrame["thresholdStrategy"].iloc[0]
        )
        thresholdScoreName = self.normalizeThresholdScoreName(
            instanceFrame["thresholdScoreSource"].iloc[0]
        )
        imputerName = self.normalizeImputerName(
            instanceFrame["imputer"].iloc[0]
        )
        runDirectory = self.createRunDirectory(
            modelCode=modelCode,
            datasetName=datasetName,
            trainingName=trainingName,
            thresholdName=thresholdName,
            thresholdScoreName=(
                thresholdScoreName if thresholdName == "DSPOT" else None
            ),
            imputerName=imputerName,
        )
        plotDirectory = runDirectory / "plots"
        plotDirectory.mkdir(parents=True, exist_ok=True)
        instancePath = runDirectory / "instances.csv"
        windowPath = runDirectory / "windows.csv"
        streamMetricsPath = runDirectory / "stream_metrics.csv"

        windowFrame = self.frameBuilder.buildWindowFrame(
            instanceFrame,
            windowSize,
        )
        streamMetricsFrame = self.frameBuilder.buildStreamMetricsFrame(
            instanceFrame
        )
        instanceFrame.to_csv(instancePath, index=False)
        windowFrame.to_csv(windowPath, index=False)
        streamMetricsFrame.to_csv(streamMetricsPath, index=False)

        plotPaths = {}
        if generatePlots:
            plotPaths = self.generatePlots(
                instancePath=instancePath,
                windowPath=windowPath,
                plotDirectory=plotDirectory,
                movingAverageColumns=movingAverageColumns,
                windowSize=windowSize,
            )

        return {
            "runDirectory": str(runDirectory),
            "instancePath": str(instancePath),
            "windowPath": str(windowPath),
            "streamMetricsPath": str(streamMetricsPath),
            "plotDirectory": str(plotDirectory),
            "plotPaths": plotPaths,
            "instanceFrame": instanceFrame,
            "windowFrame": windowFrame,
            "streamMetricsFrame": streamMetricsFrame,
        }

    def createRunDirectory(
        self,
        modelCode,
        datasetName,
        trainingName,
        thresholdName,
        thresholdScoreName=None,
        imputerName=None,
    ):
        modelDirectory = self.outputPath / str(modelCode).strip().upper()
        modelDirectory.mkdir(parents=True, exist_ok=True)
        datasetDirectory = modelDirectory / self.normalizeDatasetName(
            datasetName
        )
        datasetDirectory.mkdir(parents=True, exist_ok=True)

        existingNumbers = []
        for path in datasetDirectory.iterdir():
            if not path.is_dir():
                continue
            prefix = path.name.split("-", 1)[0]
            if prefix.isdigit():
                existingNumbers.append(int(prefix))

        directoryParts = [
            f"{max(existingNumbers, default=0) + 1:03d}",
            trainingName,
            thresholdName,
        ]
        if thresholdScoreName:
            directoryParts.append(thresholdScoreName)
        if imputerName:
            directoryParts.append(imputerName)

        runDirectory = datasetDirectory / "-".join(directoryParts)
        runDirectory.mkdir(parents=True, exist_ok=False)
        return runDirectory

    def normalizeDatasetName(self, datasetName):
        datasetStem = Path(str(datasetName).strip()).stem
        asciiName = (
            unicodedata.normalize("NFKD", datasetStem)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        safeName = re.sub(r"[^A-Za-z0-9]+", "_", asciiName).strip("_")
        if not safeName:
            raise ValueError("O nome do dataset não pode ser vazio.")

        nameParts = safeName.rsplit("_", 1)
        scenarioNames = {
            "adaptacao": "Adaptation",
            "consistencia": "Consistency",
            "generalizacao": "Generalization",
            "recorrencia": "Recurrence",
        }
        scenarioName = scenarioNames.get(nameParts[0].lower())
        if scenarioName is None:
            return safeName
        if len(nameParts) == 1:
            return scenarioName
        return f"{scenarioName}_{nameParts[1]}"

    def normalizeTrainingName(self, trainingStrategy):
        value = str(trainingStrategy).strip().lower()
        if value == "all":
            return "ALL"
        if value in {"belowthreshold", "predicttrue", "predict_true"}:
            return "PREDICT_TRUE"
        return value.upper()

    def normalizeThresholdName(self, thresholdStrategy):
        value = str(thresholdStrategy).strip().lower()
        if value == "fixed":
            return "FIXED"
        if value == "dspot":
            return "DSPOT"
        return value.upper()

    def normalizeThresholdScoreName(self, thresholdScoreSource):
        value = str(thresholdScoreSource).strip()
        if value.lower() == "raw":
            return "RAW"
        if value.lower().startswith("scorema"):
            return f"MA{value[len('scoreMa'):]}"
        return value.upper()

    def normalizeImputerName(self, imputer):
        value = (
            str(imputer)
            .strip()
            .lower()
            .replace("_", "")
            .replace("-", "")
        )
        if value == "zero":
            return "ZERO"
        if value in {"incrementalmean", "mean", "media", "média"}:
            return "MEAN"
        return value.upper()

    def buildWindowFrame(self, instanceFrame, windowSize):
        return self.frameBuilder.buildWindowFrame(instanceFrame, windowSize)

    def buildStreamMetricsFrame(self, instanceFrame):
        return self.frameBuilder.buildStreamMetricsFrame(instanceFrame)

    def selectEvaluatedFrame(self, instanceFrame):
        return self.frameBuilder.selectEvaluatedFrame(instanceFrame)

    def generatePlots(
        self,
        instancePath,
        windowPath,
        plotDirectory,
        movingAverageColumns,
        windowSize,
    ):
        plotter = Plots()
        movingAverageColumns = list(movingAverageColumns or [])
        scoreLabels = [
            f"Média móvel ({column.replace('scoreMa', '')})"
            for column in movingAverageColumns
        ]
        scoreColors = ["#5f86ad", "#f0a43a", "#6f2dbd", "#2a9d8f"]

        scorePath = plotter.plotScoreArtifact(
            instancePath,
            outputPath=plotDirectory / "scores.png",
            movingAverageColumns=movingAverageColumns,
            movingAverageLabels=scoreLabels,
            movingAverageColors=scoreColors,
            showWarmup=True,
            attackAlpha=0.30,
            legendColumns=8,
        )
        errorPath = plotter.plotWindowErrors(
            windowPath,
            attackSource=instancePath,
            outputPath=plotDirectory / "fp_fn_windows.png",
            windowSize=windowSize,
            attackAlpha=0.30,
            legendColumns=8,
        )
        metricsPath = plotter.plotWindowMetrics(
            windowPath,
            attackSource=instancePath,
            outputPath=plotDirectory / "metrics_windows.png",
            windowSize=windowSize,
            attackAlpha=0.30,
            legendColumns=8,
        )
        return {
            "scores": scorePath,
            "errors": errorPath,
            "metrics": metricsPath,
        }
