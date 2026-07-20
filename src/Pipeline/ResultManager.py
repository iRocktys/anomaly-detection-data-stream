from pathlib import Path

import pandas as pd

from src.Metrics.Metrics import Metrics
from src.Plots.Plots import Plots


class ResultManager:
    def __init__(self, outputPath="output"):
        self.outputPath = Path(outputPath)

    def save(self, rows, datasetName, modelCode, windowSize, movingAverageColumns=None, generatePlots=True):
        if not rows:
            raise ValueError("Não existem resultados para salvar.")

        instanceFrame = pd.DataFrame(rows)
        trainingName = self.normalizeTrainingName(instanceFrame["trainingStrategy"].iloc[0])
        thresholdName = self.normalizeThresholdName(instanceFrame["thresholdStrategy"].iloc[0])

        runDirectory = self.createRunDirectory(modelCode, trainingName, thresholdName)
        plotDirectory = runDirectory / "plots"
        plotDirectory.mkdir(parents=True, exist_ok=True)

        instancePath = runDirectory / "instances.csv"
        windowPath = runDirectory / "windows.csv"

        windowFrame = self.buildWindowFrame(instanceFrame, windowSize)

        instanceFrame.to_csv(instancePath, index=False)
        windowFrame.to_csv(windowPath, index=False)

        plotPaths = {}

        if generatePlots:
            plotPaths = self.generatePlots(instancePath, windowPath, plotDirectory, movingAverageColumns, windowSize)

        return {
            "runDirectory": str(runDirectory),
            "instancePath": str(instancePath),
            "windowPath": str(windowPath),
            "plotDirectory": str(plotDirectory),
            "plotPaths": plotPaths,
            "instanceFrame": instanceFrame,
            "windowFrame": windowFrame,
        }

    def createRunDirectory(self, modelCode, trainingName, thresholdName):
        modelDirectory = self.outputPath / str(modelCode).strip().upper()
        modelDirectory.mkdir(parents=True, exist_ok=True)

        existingNumbers = []

        for path in modelDirectory.iterdir():
            if not path.is_dir():
                continue

            prefix = path.name.split("-", 1)[0]

            if prefix.isdigit():
                existingNumbers.append(int(prefix))

        nextNumber = max(existingNumbers, default=0) + 1
        directoryName = f"{nextNumber:03d}-{trainingName}-{thresholdName}"
        runDirectory = modelDirectory / directoryName
        runDirectory.mkdir(parents=True, exist_ok=False)

        return runDirectory

    def normalizeTrainingName(self, trainingStrategy):
        trainingStrategy = str(trainingStrategy).strip().lower()

        if trainingStrategy == "all":
            return "ALL"

        if trainingStrategy in ["belowthreshold", "predicttrue", "predict_true"]:
            return "PREDICT_TRUE"

        return trainingStrategy.upper()

    def normalizeThresholdName(self, thresholdStrategy):
        thresholdStrategy = str(thresholdStrategy).strip().lower()

        if thresholdStrategy == "fixed":
            return "FIXED"

        if thresholdStrategy == "dspot":
            return "DSPOT"

        return thresholdStrategy.upper()

    def buildWindowFrame(self, instanceFrame, windowSize):
        windowSize = max(1, int(windowSize))
        rows = []

        for start in range(0, len(instanceFrame), windowSize):
            end = min(start + windowSize, len(instanceFrame))
            window = instanceFrame.iloc[start:end]
            evaluatedWindow = window[window["thresholdReady"].astype(bool)]
            cumulative = instanceFrame.iloc[:end]
            evaluatedCumulative = cumulative[cumulative["thresholdReady"].astype(bool)]

            windowMetrics = Metrics.calculate(evaluatedWindow["isAttack"], evaluatedWindow["predictedLabel"])
            cumulativeMetrics = Metrics.calculate(evaluatedCumulative["isAttack"], evaluatedCumulative["predictedLabel"])

            row = {
                "dataset": str(instanceFrame["dataset"].iloc[0]),
                "model": str(instanceFrame["model"].iloc[0]),
                "modelConfig": str(instanceFrame["modelConfig"].iloc[0]),
                "normalizer": str(instanceFrame["normalizer"].iloc[0]),
                "trainingStrategy": str(instanceFrame["trainingStrategy"].iloc[0]),
                "thresholdStrategy": str(instanceFrame["thresholdStrategy"].iloc[0]),
                "evaluationName": str(instanceFrame["evaluationName"].iloc[0]),
                "windowSize": windowSize,
                "windowIndex": len(rows),
                "windowStart": int(start),
                "windowEnd": int(end - 1),
                **windowMetrics,
                "cumulativeInstances": cumulativeMetrics["instances"],
                "cumulativeTp": cumulativeMetrics["tp"],
                "cumulativeTn": cumulativeMetrics["tn"],
                "cumulativeFp": cumulativeMetrics["fp"],
                "cumulativeFn": cumulativeMetrics["fn"],
                "cumulativeAccuracy": cumulativeMetrics["accuracy"],
                "cumulativePrecision": cumulativeMetrics["precision"],
                "cumulativeRecall": cumulativeMetrics["recall"],
                "cumulativeSpecificity": cumulativeMetrics["specificity"],
                "cumulativeF1": cumulativeMetrics["f1"],
                "cumulativeMcc": cumulativeMetrics["mcc"],
            }

            rows.append(row)

        return pd.DataFrame(rows)

    def generatePlots(self, instancePath, windowPath, plotDirectory, movingAverageColumns, windowSize):
        plotter = Plots()
        movingAverageColumns = list(movingAverageColumns or [])

        scoreLabels = [f"Média móvel ({column.replace('scoreMa', '')})" for column in movingAverageColumns]
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