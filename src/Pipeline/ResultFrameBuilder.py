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
        # Constrói a tabela de métricas calculadas por janela de avaliação.
        windowSize = max(1, int(windowSize))
        evaluatedFrame = self.selectEvaluatedFrame(instanceFrame)
        rows = []
        cumulativeCounts = dict(tp=0, tn=0, fp=0, fn=0)
        metadata = self._metadata(evaluatedFrame)

        for start in range(0, len(evaluatedFrame), windowSize):
            end = min(start + windowSize, len(evaluatedFrame))
            window = evaluatedFrame.iloc[start:end]
            windowMetrics = Metrics.calculate(
                window["isAttack"],
                window["predictedLabel"],
            )
            for name in cumulativeCounts:
                cumulativeCounts[name] += windowMetrics[name]
            cumulativeMetrics = Metrics.fromCounts(**cumulativeCounts)

            row = dict(metadata)
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
        # Constrói a tabela com as métricas globais da stream processada.
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
        # Seleciona somente as instâncias efetivamente liberadas para avaliação.
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
                "No column indicating evaluable instances was found."
            )

        evaluatedFrame = instanceFrame[evaluationMask].reset_index(drop=True)
        if evaluatedFrame.empty:
            raise ValueError(
                "No instances are available after warm-up to calculate metrics."
            )
        return evaluatedFrame

    def _metadata(self, frame):
        # Extrai os metadados comuns necessários para compor os resultados.
        return {
            column: str(frame[column].iloc[0])
            for column in self.metadataColumns
        }

    def _cumulativeMetrics(self, metrics):
        # Obtém as métricas acumuladas da última instância avaliada.
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
