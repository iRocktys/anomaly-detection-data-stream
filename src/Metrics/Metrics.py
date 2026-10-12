import math

import numpy as np
import pandas as pd


class Metrics:
    metricNames = [
        "instances",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
    ]

    @staticmethod
    def calculate(yTrue, yPred):
        # Calcula a matriz de confusão e todas as métricas para rótulos binários 0 e 1.
        true_values = np.asarray(yTrue, dtype=int)
        predicted_values = np.asarray(yPred, dtype=int)
        if true_values.shape != predicted_values.shape:
            raise ValueError("yTrue and yPred must have the same length.")
        if np.any(~np.isin(true_values, [0, 1])) or np.any(~np.isin(predicted_values, [0, 1])):
            raise ValueError("Metrics accept only binary labels 0 and 1.")
        tp = int(np.sum((true_values == 1) & (predicted_values == 1)))
        tn = int(np.sum((true_values == 0) & (predicted_values == 0)))
        fp = int(np.sum((true_values == 0) & (predicted_values == 1)))
        fn = int(np.sum((true_values == 1) & (predicted_values == 0)))
        return Metrics.fromCounts(tp, tn, fp, fn)

    @staticmethod
    def fromCounts(tp, tn, fp, fn):
        # Converte contagens binárias em accuracy, precision, recall, specificity, F1 e MCC.
        tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)
        if min(tp, tn, fp, fn) < 0:
            raise ValueError("Confusion matrix values cannot be negative.")
        total = tp + tn + fp + fn
        precision = Metrics.safeDivide(tp, tp + fp)
        recall = Metrics.safeDivide(tp, tp + fn)
        specificity = Metrics.safeDivide(tn, tn + fp)
        accuracy = Metrics.safeDivide(tp + tn, total)
        f1 = Metrics.safeDivide(2.0 * precision * recall, precision + recall)
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = Metrics.safeDivide((tp * tn) - (fp * fn), denominator)
        return {
            "instances": total,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "mcc": mcc,
        }

    @staticmethod
    def evaluateFrame(frame, trueColumn="isAttack", predictionColumn="predictedLabel", readyColumn="evaluationReady"):
        # Avalia um DataFrame binário respeitando opcionalmente a máscara de instâncias avaliáveis.
        selected = frame
        if readyColumn and readyColumn in selected.columns:
            selected = selected[selected[readyColumn].astype(bool)]
        if selected.empty:
            return Metrics.fromCounts(0, 0, 0, 0)
        return Metrics.calculate(selected[trueColumn], selected[predictionColumn])

    @staticmethod
    def windowed(frame, windowSize=100, trueColumn="isAttack", predictionColumn="predictedLabel", readyColumn="evaluationReady"):
        # Calcula métricas binárias independentes em blocos sequenciais de tamanho fixo.
        selected = frame
        if readyColumn and readyColumn in selected.columns:
            selected = selected[selected[readyColumn].astype(bool)].reset_index(drop=True)
        window_size = max(1, int(windowSize))
        rows = []
        for start in range(0, len(selected), window_size):
            window = selected.iloc[start:start + window_size]
            if window.empty:
                continue
            summary = Metrics.calculate(window[trueColumn], window[predictionColumn])
            summary.update({"windowStart": int(start), "windowEnd": int(start + len(window) - 1)})
            rows.append(summary)
        return pd.DataFrame(rows, columns=["windowStart", "windowEnd"] + Metrics.metricNames)

    @staticmethod
    def safeDivide(numerator, denominator):
        # Executa divisão segura retornando zero quando o denominador é nulo.
        return float(numerator / denominator) if denominator else 0.0
