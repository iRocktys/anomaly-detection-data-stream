import json

import pandas as pd

from src.Metrics.Metrics import Metrics
from src.Pipeline.ExperimentBuilder import ExperimentBuilder


class ThresholdReplayEvaluator:
    """Reproduz decisões causais sobre scores persistidos, sem retreinar o modelo."""

    def __init__(self, builder=None):
        self.builder = builder or ExperimentBuilder()

    def evaluateFrame(self, scoreFrame, evaluationConfig, warmup=200, evaluationId=None):
        evaluationConfig.validate()
        if evaluationConfig.scoreColumn not in scoreFrame.columns:
            raise ValueError(
                f"Coluna de score ausente: {evaluationConfig.scoreColumn}. "
                f"Disponíveis: {list(scoreFrame.columns)}"
            )

        resolvedWarmup = self.resolveWarmup(scoreFrame, warmup)
        components = self.builder.buildEvaluationComponents(
            evaluationConfig,
            warmup=resolvedWarmup,
        )
        baseColumns = [
            column
            for column in scoreFrame.columns
            if column in {
                "runId",
                "scoreArtifactId",
                "dataset",
                "instanceId",
                "trueLabel",
                "labelName",
                "isAttack",
                "modelCode",
                "modelConfig",
                "modelName",
                "normalizer",
                "normalizerUpdatePolicy",
                "trainingStrategy",
                "trainingFeedbackEvaluation",
                "trainingSourceScore",
                "trainingScore",
                "trainingThreshold",
                "trainingThresholdReady",
                "trainingPrediction",
                "trainingAllowed",
                "wasTrained",
                "warmup",
                "isWarmup",
                "runSeed",
                "rawScore",
            }
            or column.startswith("scoreMa")
            or column == evaluationConfig.scoreColumn
        ]

        rows = []
        for position, (_, scoreRow) in enumerate(scoreFrame.iterrows()):
            rawValue = float(scoreRow[evaluationConfig.scoreColumn])
            smoothValue = float(components["scoreSmoother"].transform(rawValue))
            thresholdValue = float(components["threshold"].getThreshold())
            rowIsWarmup = self.resolveRowWarmup(
                scoreRow,
                position=position,
                warmup=resolvedWarmup,
            )
            thresholdReady = bool(components["threshold"].isReady()) and not rowIsWarmup
            prediction = int(
                components["decision"].predict(
                    smoothValue,
                    thresholdValue,
                    thresholdReady,
                )
            )

            row = {column: scoreRow[column] for column in baseColumns}
            row.update({
                "warmup": resolvedWarmup,
                "isWarmup": int(rowIsWarmup),
                "evaluationId": evaluationId or evaluationConfig.name,
                "evaluationName": evaluationConfig.name,
                "scoreColumn": evaluationConfig.scoreColumn,
                "scoreSmoother": evaluationConfig.scoreSmoother.name,
                "score": smoothValue,
                "thresholdStrategy": evaluationConfig.threshold.name,
                "threshold": thresholdValue,
                "thresholdReady": thresholdReady,
                "prediction": prediction,
                "isFalsePositive": int(int(row["isAttack"]) == 0 and prediction == 1),
                "isFalseNegative": int(int(row["isAttack"]) == 1 and prediction == 0),
            })
            if evaluationConfig.saveThresholdState:
                row["thresholdState"] = json.dumps(
                    components["threshold"].getState(),
                    ensure_ascii=False,
                    default=self.jsonDefault,
                )
            rows.append(row)

            components["scoreSmoother"].update(rawValue)
            components["decision"].update(
                smoothValue,
                thresholdValue,
                prediction,
                int(row["trueLabel"]),
            )
            components["threshold"].update(smoothValue)

        evaluationFrame = pd.DataFrame(rows)
        readyColumn = "thresholdReady" if evaluationConfig.evaluateOnlyReady else None
        summary = Metrics.evaluateFrame(
            evaluationFrame,
            readyColumn=readyColumn,
        )
        windowMetrics = Metrics.windowed(
            evaluationFrame,
            windowSize=evaluationConfig.metricsWindow,
            readyColumn=readyColumn,
        )
        evaluatedRows = evaluationFrame
        if readyColumn:
            evaluatedRows = evaluationFrame[evaluationFrame[readyColumn].astype(bool)]
        summary.update({
            "attackBreakdown": Metrics.attackBreakdown(evaluatedRows),
            "evaluationName": evaluationConfig.name,
            "thresholdStrategy": evaluationConfig.threshold.name,
            "scoreSmoother": evaluationConfig.scoreSmoother.name,
            "scoreColumn": evaluationConfig.scoreColumn,
            "evaluateOnlyReady": evaluationConfig.evaluateOnlyReady,
            "warmup": resolvedWarmup,
            "finalThresholdState": components["threshold"].getState(),
        })
        return evaluationFrame, summary, windowMetrics

    def evaluateFile(self, scorePath, evaluationConfig, warmup=200, evaluationId=None):
        return self.evaluateFrame(
            pd.read_csv(scorePath),
            evaluationConfig,
            warmup=warmup,
            evaluationId=evaluationId,
        )

    @staticmethod
    def resolveWarmup(scoreFrame, requestedWarmup):
        resolvedWarmup = int(requestedWarmup)
        if resolvedWarmup < 20:
            raise ValueError("warmup deve ser maior ou igual a 20.")
        if "warmup" not in scoreFrame.columns:
            return resolvedWarmup

        artifactWarmups = (
            pd.to_numeric(scoreFrame["warmup"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if len(artifactWarmups) > 1:
            raise ValueError("O artefato possui mais de um valor de warmup.")
        if len(artifactWarmups) == 1 and int(artifactWarmups[0]) != resolvedWarmup:
            raise ValueError(
                "O warmup do plano deve ser igual ao warmup usado para gerar os scores: "
                f"plano={resolvedWarmup}, artefato={int(artifactWarmups[0])}."
            )
        return resolvedWarmup

    @staticmethod
    def resolveRowWarmup(scoreRow, position, warmup):
        if "isWarmup" in scoreRow.index and pd.notna(scoreRow["isWarmup"]):
            return bool(int(scoreRow["isWarmup"]))
        if "instanceId" in scoreRow.index and pd.notna(scoreRow["instanceId"]):
            return int(scoreRow["instanceId"]) < int(warmup)
        return int(position) < int(warmup)

    @staticmethod
    def jsonDefault(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)
