import pandas as pd

from src.Metrics.Metrics import Metrics


class ResultFrameBuilder:
    metadataColumns = (
        "task",
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
    )

    def buildWindowFrame(self, instanceFrame, windowSize):
        # Constrói métricas binárias por janela usando somente instâncias prontas para avaliação.
        evaluated_frame = self.selectEvaluatedFrame(instanceFrame)
        window_size = max(1, int(windowSize))
        rows = []
        cumulative_tp = cumulative_tn = cumulative_fp = cumulative_fn = 0
        for window_index, start in enumerate(range(0, len(evaluated_frame), window_size)):
            window = evaluated_frame.iloc[start:start + window_size]
            metrics = Metrics.calculate(window["isAttack"], window["predictedLabel"])
            cumulative_tp += metrics["tp"]
            cumulative_tn += metrics["tn"]
            cumulative_fp += metrics["fp"]
            cumulative_fn += metrics["fn"]
            cumulative = Metrics.fromCounts(cumulative_tp, cumulative_tn, cumulative_fp, cumulative_fn)
            row = self._metadata(window)
            row.update({
                "windowSize": window_size,
                "windowIndex": window_index,
                "windowStart": int(window["instanceId"].iloc[0]),
                "windowEnd": int(window["instanceId"].iloc[-1]),
                **metrics,
                "cumulativeInstances": cumulative["instances"],
                "cumulativeTp": cumulative["tp"],
                "cumulativeTn": cumulative["tn"],
                "cumulativeFp": cumulative["fp"],
                "cumulativeFn": cumulative["fn"],
                "cumulativeAccuracy": cumulative["accuracy"],
                "cumulativePrecision": cumulative["precision"],
                "cumulativeRecall": cumulative["recall"],
                "cumulativeSpecificity": cumulative["specificity"],
                "cumulativeF1": cumulative["f1"],
                "cumulativeMcc": cumulative["mcc"],
            })
            rows.append(row)
        return pd.DataFrame(rows)

    def buildStreamMetricsFrame(self, instanceFrame):
        # Resume toda a região avaliada em uma única linha de métricas binárias e metadados.
        evaluated_frame = self.selectEvaluatedFrame(instanceFrame)
        metric_values = Metrics.calculate(evaluated_frame["isAttack"], evaluated_frame["predictedLabel"])
        evaluated_instances = int(metric_values.pop("instances"))
        attack_instances = int(evaluated_frame["isAttack"].astype(int).sum())
        benign_instances = evaluated_instances - attack_instances
        row = self._metadata(evaluated_frame)
        row.update({
            "warmup": int(instanceFrame["warmup"].iloc[0]) if "warmup" in instanceFrame.columns else 0,
            "totalInstances": int(len(instanceFrame)),
            "evaluationStart": int(evaluated_frame["instanceId"].iloc[0]),
            "evaluationEnd": int(evaluated_frame["instanceId"].iloc[-1]),
            "evaluatedInstances": evaluated_instances,
            "benignInstances": benign_instances,
            "attackInstances": attack_instances,
            "attackRatioPercent": Metrics.safeDivide(attack_instances, evaluated_instances) * 100.0,
            **metric_values,
        })
        for column in ("modelWarmup", "thresholdWarmup", "thresholdCalibrationWindow", "thresholdCalibrationStart"):
            if column in instanceFrame.columns:
                row[column] = int(instanceFrame[column].iloc[0])
        return pd.DataFrame([row])

    def selectEvaluatedFrame(self, instanceFrame):
        # Filtra somente as linhas explicitamente marcadas como avaliáveis pelos pipelines.
        if "evaluationReady" not in instanceFrame.columns:
            raise ValueError("The 'evaluationReady' column was not found in the results.")
        evaluated_frame = instanceFrame[instanceFrame["evaluationReady"].astype(bool)].reset_index(drop=True)
        if evaluated_frame.empty:
            raise ValueError("No instances are available to calculate metrics.")
        return evaluated_frame

    def _metadata(self, frame):
        # Copia apenas metadados que realmente existem no contrato da tarefa atual.
        return {
            column: str(frame[column].iloc[0])
            for column in self.metadataColumns
            if column in frame.columns
        }
