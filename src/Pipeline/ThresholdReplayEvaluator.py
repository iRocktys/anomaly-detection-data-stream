import json

import pandas as pd

from src.Metrics.Metrics import Metrics
from src.Pipeline.ExperimentBuilder import ExperimentBuilder


class ThresholdReplayEvaluator:
    """Reproduz decisões causais sobre scores já persistidos, sem retreinar o modelo."""

    def __init__(self, builder=None):
        self.builder = builder or ExperimentBuilder()

    def evaluateFrame(self, scoreFrame, evaluationConfig, evaluationId=None):
        evaluationConfig.validate()
        if evaluationConfig.scoreColumn not in scoreFrame.columns:
            raise ValueError(
                f"Coluna de score ausente: {evaluationConfig.scoreColumn}. "
                f"Disponíveis: {list(scoreFrame.columns)}"
            )

        components = self.builder.buildEvaluationComponents(evaluationConfig)
        baseColumns = [
            column
            for column in scoreFrame.columns
            if column in {
                "runId", "scoreArtifactId", "dataset", "instanceId", "trueLabel",
                "labelName", "isAttack", "modelCode", "modelConfig", "modelName", "normalizer",
                "normalizerUpdatePolicy", "trainingStrategy", "runSeed", "rawScore",
            } or column.startswith("scoreMa") or column == evaluationConfig.scoreColumn
        ]
        rows = []
        for _, scoreRow in scoreFrame.iterrows():
            rawValue = float(scoreRow[evaluationConfig.scoreColumn])
            smoothValue = float(components["scoreSmoother"].transform(rawValue))
            thresholdValue = float(components["threshold"].getThreshold())
            thresholdReady = bool(components["threshold"].isReady())
            prediction = int(components["decision"].predict(
                smoothValue,
                thresholdValue,
                thresholdReady,
            ))

            row = {column: scoreRow[column] for column in baseColumns}
            row.update({
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
            "finalThresholdState": components["threshold"].getState(),
        })
        return evaluationFrame, summary, windowMetrics

    def evaluateFile(self, scorePath, evaluationConfig, evaluationId=None):
        return self.evaluateFrame(
            pd.read_csv(scorePath),
            evaluationConfig,
            evaluationId=evaluationId,
        )

    @staticmethod
    def jsonDefault(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)
