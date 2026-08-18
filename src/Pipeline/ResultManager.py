import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.Metrics.Metrics import Metrics
from src.Plots.Plots import Plots


class ResultManager:
    def __init__(self, outputPath="output"):
        self.outputPath = Path(outputPath)

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
            raise ValueError(
                "Não existem resultados para salvar."
            )

        instanceFrame = pd.DataFrame(rows)

        trainingName = self.normalizeTrainingName(
            instanceFrame["trainingStrategy"].iloc[0]
        )

        thresholdName = self.normalizeThresholdName(
            instanceFrame["thresholdStrategy"].iloc[0]
        )

        thresholdScoreName = (
            self.normalizeThresholdScoreName(
                instanceFrame[
                    "thresholdScoreSource"
                ].iloc[0]
            )
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
                thresholdScoreName
                if thresholdName == "DSPOT"
                else None
            ),
            imputerName=imputerName,
        )

        plotDirectory = runDirectory / "plots"
        plotDirectory.mkdir(
            parents=True,
            exist_ok=True,
        )

        instancePath = (
            runDirectory
            / "instances.csv"
        )

        windowPath = (
            runDirectory
            / "windows.csv"
        )

        streamMetricsPath = (
            runDirectory
            / "stream_metrics.csv"
        )

        windowFrame = self.buildWindowFrame(
            instanceFrame,
            windowSize,
        )

        streamMetricsFrame = (
            self.buildStreamMetricsFrame(
                instanceFrame
            )
        )

        instanceFrame.to_csv(
            instancePath,
            index=False,
        )

        windowFrame.to_csv(
            windowPath,
            index=False,
        )

        streamMetricsFrame.to_csv(
            streamMetricsPath,
            index=False,
        )

        plotPaths = {}

        if generatePlots:
            plotPaths = self.generatePlots(
                instancePath,
                windowPath,
                plotDirectory,
                movingAverageColumns,
                windowSize,
            )

        return {
            "runDirectory": str(
                runDirectory
            ),
            "instancePath": str(
                instancePath
            ),
            "windowPath": str(
                windowPath
            ),
            "streamMetricsPath": str(
                streamMetricsPath
            ),
            "plotDirectory": str(
                plotDirectory
            ),
            "plotPaths": plotPaths,
            "instanceFrame": instanceFrame,
            "windowFrame": windowFrame,
            "streamMetricsFrame": (
                streamMetricsFrame
            ),
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
        modelDirectory = (
            self.outputPath
            / str(modelCode)
            .strip()
            .upper()
        )

        modelDirectory.mkdir(
            parents=True,
            exist_ok=True,
        )

        normalizedDatasetName = (
            self.normalizeDatasetName(
                datasetName
            )
        )

        datasetDirectory = (
            modelDirectory
            / normalizedDatasetName
        )

        datasetDirectory.mkdir(
            parents=True,
            exist_ok=True,
        )

        existingNumbers = []

        for path in datasetDirectory.iterdir():
            if not path.is_dir():
                continue

            prefix = path.name.split(
                "-",
                1,
            )[0]

            if prefix.isdigit():
                existingNumbers.append(
                    int(prefix)
                )

        nextNumber = (
            max(
                existingNumbers,
                default=0,
            )
            + 1
        )

        directoryParts = [
            f"{nextNumber:03d}",
            trainingName,
            thresholdName,
        ]

        if thresholdScoreName:
            directoryParts.append(
                thresholdScoreName
            )

        if imputerName:
            directoryParts.append(
                imputerName
            )

        directoryName = "-".join(
            directoryParts
        )

        runDirectory = (
            datasetDirectory
            / directoryName
        )

        runDirectory.mkdir(
            parents=True,
            exist_ok=False,
        )

        return runDirectory

    def normalizeDatasetName(
        self,
        datasetName,
    ):
        datasetStem = Path(
            str(datasetName).strip()
        ).stem

        asciiName = (
            unicodedata.normalize(
                "NFKD",
                datasetStem,
            )
            .encode(
                "ascii",
                "ignore",
            )
            .decode("ascii")
        )

        safeName = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            asciiName,
        ).strip("_")

        if not safeName:
            raise ValueError(
                "O nome do dataset "
                "não pode ser vazio."
            )

        nameParts = safeName.rsplit(
            "_",
            1,
        )

        scenarioKey = (
            nameParts[0].lower()
        )

        scenarioNames = {
            "adaptacao": "Adaptation",
            "consistencia": "Consistency",
            "generalizacao": "Generalization",
            "recorrencia": "Recurrence",
        }

        if scenarioKey not in scenarioNames:
            return safeName

        scenarioName = scenarioNames[
            scenarioKey
        ]

        if len(nameParts) == 1:
            return scenarioName

        return (
            f"{scenarioName}_"
            f"{nameParts[1]}"
        )

    def normalizeTrainingName(
        self,
        trainingStrategy,
    ):
        trainingStrategy = (
            str(trainingStrategy)
            .strip()
            .lower()
        )

        if trainingStrategy == "all":
            return "ALL"

        if trainingStrategy in [
            "belowthreshold",
            "predicttrue",
            "predict_true",
        ]:
            return "PREDICT_TRUE"

        return trainingStrategy.upper()

    def normalizeThresholdName(
        self,
        thresholdStrategy,
    ):
        thresholdStrategy = (
            str(thresholdStrategy)
            .strip()
            .lower()
        )

        if thresholdStrategy == "fixed":
            return "FIXED"

        if thresholdStrategy == "dspot":
            return "DSPOT"

        return thresholdStrategy.upper()

    def normalizeThresholdScoreName(
        self,
        thresholdScoreSource,
    ):
        thresholdScoreSource = str(
            thresholdScoreSource
        ).strip()

        if (
            thresholdScoreSource.lower()
            == "raw"
        ):
            return "RAW"

        if (
            thresholdScoreSource
            .lower()
            .startswith("scorema")
        ):
            windowSize = (
                thresholdScoreSource[
                    len("scoreMa"):
                ]
            )

            return f"MA{windowSize}"

        return (
            thresholdScoreSource.upper()
        )

    def normalizeImputerName(
        self,
        imputer,
    ):
        imputer = (
            str(imputer)
            .strip()
            .lower()
            .replace("_", "")
            .replace("-", "")
        )

        if imputer == "zero":
            return "ZERO"

        if imputer in [
            "incrementalmean",
            "mean",
            "media",
            "média",
        ]:
            return "MEAN"

        return imputer.upper()

    def buildWindowFrame(
        self,
        instanceFrame,
        windowSize,
    ):
        windowSize = max(
            1,
            int(windowSize),
        )

        evaluatedFrame = (
            self.selectEvaluatedFrame(
                instanceFrame
            )
        )

        rows = []

        for start in range(
            0,
            len(evaluatedFrame),
            windowSize,
        ):
            end = min(
                start + windowSize,
                len(evaluatedFrame),
            )

            window = (
                evaluatedFrame.iloc[
                    start:end
                ]
            )

            cumulative = (
                evaluatedFrame.iloc[
                    :end
                ]
            )

            windowMetrics = (
                Metrics.calculate(
                    window["isAttack"],
                    window[
                        "predictedLabel"
                    ],
                )
            )

            cumulativeMetrics = (
                Metrics.calculate(
                    cumulative[
                        "isAttack"
                    ],
                    cumulative[
                        "predictedLabel"
                    ],
                )
            )

            row = {
                "dataset": str(
                    evaluatedFrame[
                        "dataset"
                    ].iloc[0]
                ),
                "model": str(
                    evaluatedFrame[
                        "model"
                    ].iloc[0]
                ),
                "modelConfig": str(
                    evaluatedFrame[
                        "modelConfig"
                    ].iloc[0]
                ),
                "imputer": str(
                    evaluatedFrame[
                        "imputer"
                    ].iloc[0]
                ),
                "normalizer": str(
                    evaluatedFrame[
                        "normalizer"
                    ].iloc[0]
                ),
                "trainingStrategy": str(
                    evaluatedFrame[
                        "trainingStrategy"
                    ].iloc[0]
                ),
                "thresholdStrategy": str(
                    evaluatedFrame[
                        "thresholdStrategy"
                    ].iloc[0]
                ),
                "thresholdScoreSource": str(
                    evaluatedFrame[
                        "thresholdScoreSource"
                    ].iloc[0]
                ),
                "thresholdScoreLabel": str(
                    evaluatedFrame[
                        "thresholdScoreLabel"
                    ].iloc[0]
                ),
                "evaluationName": str(
                    evaluatedFrame[
                        "evaluationName"
                    ].iloc[0]
                ),
                "warmup": int(
                    evaluatedFrame[
                        "warmup"
                    ].iloc[0]
                ),
                "windowSize": windowSize,
                "windowIndex": len(rows),
                "windowStart": int(
                    window[
                        "instanceId"
                    ].iloc[0]
                ),
                "windowEnd": int(
                    window[
                        "instanceId"
                    ].iloc[-1]
                ),
                **windowMetrics,
                "cumulativeInstances": (
                    cumulativeMetrics[
                        "instances"
                    ]
                ),
                "cumulativeTp": (
                    cumulativeMetrics["tp"]
                ),
                "cumulativeTn": (
                    cumulativeMetrics["tn"]
                ),
                "cumulativeFp": (
                    cumulativeMetrics["fp"]
                ),
                "cumulativeFn": (
                    cumulativeMetrics["fn"]
                ),
                "cumulativeAccuracy": (
                    cumulativeMetrics[
                        "accuracy"
                    ]
                ),
                "cumulativePrecision": (
                    cumulativeMetrics[
                        "precision"
                    ]
                ),
                "cumulativeRecall": (
                    cumulativeMetrics[
                        "recall"
                    ]
                ),
                "cumulativeSpecificity": (
                    cumulativeMetrics[
                        "specificity"
                    ]
                ),
                "cumulativeF1": (
                    cumulativeMetrics["f1"]
                ),
                "cumulativeMcc": (
                    cumulativeMetrics["mcc"]
                ),
            }

            rows.append(row)

        return pd.DataFrame(rows)

    def buildStreamMetricsFrame(
        self,
        instanceFrame,
    ):
        evaluatedFrame = (
            self.selectEvaluatedFrame(
                instanceFrame
            )
        )

        metricValues = Metrics.calculate(
            evaluatedFrame["isAttack"],
            evaluatedFrame[
                "predictedLabel"
            ],
        )

        evaluatedInstances = int(
            metricValues.pop("instances")
        )

        attackInstances = int(
            evaluatedFrame[
                "isAttack"
            ]
            .astype(int)
            .sum()
        )

        benignInstances = (
            evaluatedInstances
            - attackInstances
        )

        attackRatioPercent = (
            Metrics.safeDivide(
                attackInstances,
                evaluatedInstances,
            )
            * 100.0
        )

        row = {
            "dataset": str(
                evaluatedFrame[
                    "dataset"
                ].iloc[0]
            ),
            "model": str(
                evaluatedFrame[
                    "model"
                ].iloc[0]
            ),
            "modelConfig": str(
                evaluatedFrame[
                    "modelConfig"
                ].iloc[0]
            ),
            "imputer": str(
                evaluatedFrame[
                    "imputer"
                ].iloc[0]
            ),
            "normalizer": str(
                evaluatedFrame[
                    "normalizer"
                ].iloc[0]
            ),
            "trainingStrategy": str(
                evaluatedFrame[
                    "trainingStrategy"
                ].iloc[0]
            ),
            "thresholdStrategy": str(
                evaluatedFrame[
                    "thresholdStrategy"
                ].iloc[0]
            ),
            "thresholdScoreSource": str(
                evaluatedFrame[
                    "thresholdScoreSource"
                ].iloc[0]
            ),
            "thresholdScoreLabel": str(
                evaluatedFrame[
                    "thresholdScoreLabel"
                ].iloc[0]
            ),
            "evaluationName": str(
                evaluatedFrame[
                    "evaluationName"
                ].iloc[0]
            ),
            "warmup": int(
                evaluatedFrame[
                    "warmup"
                ].iloc[0]
            ),
            "thresholdCalibrationWindow": int(
                evaluatedFrame[
                    "thresholdCalibrationWindow"
                ].iloc[0]
            ),
            "thresholdCalibrationStart": int(
                evaluatedFrame[
                    "thresholdCalibrationStart"
                ].iloc[0]
            ),
            "totalInstances": int(
                len(instanceFrame)
            ),
            "evaluationStart": int(
                evaluatedFrame[
                    "instanceId"
                ].iloc[0]
            ),
            "evaluationEnd": int(
                evaluatedFrame[
                    "instanceId"
                ].iloc[-1]
            ),
            "evaluatedInstances": (
                evaluatedInstances
            ),
            "benignInstances": (
                benignInstances
            ),
            "attackInstances": (
                attackInstances
            ),
            "attackRatioPercent": (
                attackRatioPercent
            ),
            **metricValues,
        }

        return pd.DataFrame([row])

    def selectEvaluatedFrame(
        self,
        instanceFrame,
    ):
        if (
            "evaluationReady"
            in instanceFrame.columns
        ):
            evaluationMask = (
                instanceFrame[
                    "evaluationReady"
                ].astype(bool)
            )

        elif (
            "thresholdReady"
            in instanceFrame.columns
        ):
            evaluationMask = (
                instanceFrame[
                    "thresholdReady"
                ].astype(bool)
            )

            if (
                "isWarmup"
                in instanceFrame.columns
            ):
                evaluationMask = (
                    evaluationMask
                    & ~instanceFrame[
                        "isWarmup"
                    ].astype(bool)
                )

            elif (
                "warmup"
                in instanceFrame.columns
                and "instanceId"
                in instanceFrame.columns
            ):
                warmup = int(
                    instanceFrame[
                        "warmup"
                    ].iloc[0]
                )

                evaluationMask = (
                    evaluationMask
                    & (
                        instanceFrame[
                            "instanceId"
                        ]
                        .astype(int)
                        >= warmup
                    )
                )

        else:
            raise ValueError(
                "Não foi encontrada uma coluna "
                "que indique quais instâncias "
                "podem ser avaliadas."
            )

        evaluatedFrame = (
            instanceFrame[
                evaluationMask
            ]
            .reset_index(drop=True)
        )

        if evaluatedFrame.empty:
            raise ValueError(
                "Não existem instâncias "
                "disponíveis após o warm-up "
                "para calcular as métricas."
            )

        return evaluatedFrame

    def generatePlots(
        self,
        instancePath,
        windowPath,
        plotDirectory,
        movingAverageColumns,
        windowSize,
    ):
        plotter = Plots()

        movingAverageColumns = list(
            movingAverageColumns or []
        )

        scoreLabels = [
            (
                "Média móvel "
                f"({column.replace('scoreMa', '')})"
            )
            for column in movingAverageColumns
        ]

        scoreColors = [
            "#5f86ad",
            "#f0a43a",
            "#6f2dbd",
            "#2a9d8f",
        ]

        scorePath = (
            plotter.plotScoreArtifact(
                instancePath,
                outputPath=(
                    plotDirectory
                    / "scores.png"
                ),
                movingAverageColumns=(
                    movingAverageColumns
                ),
                movingAverageLabels=(
                    scoreLabels
                ),
                movingAverageColors=(
                    scoreColors
                ),
                showWarmup=True,
                attackAlpha=0.30,
                legendColumns=8,
            )
        )

        errorPath = (
            plotter.plotWindowErrors(
                windowPath,
                attackSource=instancePath,
                outputPath=(
                    plotDirectory
                    / "fp_fn_windows.png"
                ),
                windowSize=windowSize,
                attackAlpha=0.30,
                legendColumns=8,
            )
        )

        metricsPath = (
            plotter.plotWindowMetrics(
                windowPath,
                attackSource=instancePath,
                outputPath=(
                    plotDirectory
                    / "metrics_windows.png"
                ),
                windowSize=windowSize,
                attackAlpha=0.30,
                legendColumns=8,
            )
        )

        return {
            "scores": scorePath,
            "errors": errorPath,
            "metrics": metricsPath,
        }