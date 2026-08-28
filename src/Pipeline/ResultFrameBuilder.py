import pandas as pd

from src.Metrics.Metrics import Metrics


class ResultFrameBuilder:
    metadataColumns = [
        "dataset",
        "model",
        "modelConfig",
        "imputer",
        "normalizer",
        "trainingStrategy",
        "thresholdStrategy",
        "thresholdScoreSource",
        "thresholdScoreLabel",
        "evaluationName",
    ]

    def buildWindowFrame(self, instanceFrame, windowSize):
        windowSize = max(1, int(windowSize))
        evaluatedFrame = self.selectEvaluatedFrame(instanceFrame)
        rows = []

        for start in range(0, len(evaluatedFrame), windowSize):
            end = min(start + windowSize, len(evaluatedFrame))
            window = evaluatedFrame.iloc[start:end]
            cumulative = evaluatedFrame.iloc[:end]
            windowMetrics = Metrics.calculate(
                window["isAttack"],
                window["predictedLabel"],
            )
            cumulativeMetrics = Metrics.calculate(
                cumulative["isAttack"],
                cumulative["predictedLabel"],
            )

            row = self._metadata(evaluatedFrame)
            row.update(
                {
                    "warmup": int(evaluatedFrame["warmup"].iloc[0]),
                    "windowSize": windowSize,
                    "windowIndex": len(rows),
                    "windowStart": int(window["instanceId"].iloc[0]),
                    "windowEnd": int(window["instanceId"].iloc[-1]),
                    **windowMetrics,
                    **self._cumulativeMetrics(cumulativeMetrics),
                }
            )
            rows.append(row)

        return pd.DataFrame(rows)

    def buildStreamMetricsFrame(self, instanceFrame):
        evaluatedFrame = self.selectEvaluatedFrame(instanceFrame)
        metricValues = Metrics.calculate(
            evaluatedFrame["isAttack"],
            evaluatedFrame["predictedLabel"],
        )
        evaluatedInstances = int(metricValues.pop("instances"))
        attackInstances = int(
            evaluatedFrame["isAttack"].astype(int).sum()
        )
        benignInstances = evaluatedInstances - attackInstances

        row = self._metadata(evaluatedFrame)
        row.update(
            {
                "warmup": int(evaluatedFrame["warmup"].iloc[0]),
                "thresholdCalibrationWindow": int(
                    evaluatedFrame["thresholdCalibrationWindow"].iloc[0]
                ),
                "thresholdCalibrationStart": int(
                    evaluatedFrame["thresholdCalibrationStart"].iloc[0]
                ),
                "totalInstances": int(len(instanceFrame)),
                "evaluationStart": int(
                    evaluatedFrame["instanceId"].iloc[0]
                ),
                "evaluationEnd": int(
                    evaluatedFrame["instanceId"].iloc[-1]
                ),
                "evaluatedInstances": evaluatedInstances,
                "benignInstances": benignInstances,
                "attackInstances": attackInstances,
                "attackRatioPercent": Metrics.safeDivide(
                    attackInstances,
                    evaluatedInstances,
                )
                * 100.0,
                **metricValues,
            }
        )
        return pd.DataFrame([row])

    def selectEvaluatedFrame(self, instanceFrame):
        if "evaluationReady" in instanceFrame.columns:
            evaluationMask = instanceFrame["evaluationReady"].astype(bool)
        elif "thresholdReady" in instanceFrame.columns:
            evaluationMask = instanceFrame["thresholdReady"].astype(bool)

            if "isWarmup" in instanceFrame.columns:
                evaluationMask &= ~instanceFrame["isWarmup"].astype(bool)
            elif {
                "warmup",
                "instanceId",
            }.issubset(instanceFrame.columns):
                warmup = int(instanceFrame["warmup"].iloc[0])
                evaluationMask &= (
                    instanceFrame["instanceId"].astype(int) >= warmup
                )
        else:
            raise ValueError(
                "Não foi encontrada uma coluna que indique quais instâncias "
                "podem ser avaliadas."
            )

        evaluatedFrame = instanceFrame[evaluationMask].reset_index(drop=True)
        if evaluatedFrame.empty:
            raise ValueError(
                "Não existem instâncias disponíveis após o warm-up para "
                "calcular as métricas."
            )
        return evaluatedFrame

    def _metadata(self, frame):
        return {
            column: str(frame[column].iloc[0])
            for column in self.metadataColumns
        }

    def _cumulativeMetrics(self, metrics):
        return {
            "cumulativeInstances": metrics["instances"],
            "cumulativeTp": metrics["tp"],
            "cumulativeTn": metrics["tn"],
            "cumulativeFp": metrics["fp"],
            "cumulativeFn": metrics["fn"],
            "cumulativeAccuracy": metrics["accuracy"],
            "cumulativePrecision": metrics["precision"],
            "cumulativeRecall": metrics["recall"],
            "cumulativeSpecificity": metrics["specificity"],
            "cumulativeF1": metrics["f1"],
            "cumulativeMcc": metrics["mcc"],
        }
