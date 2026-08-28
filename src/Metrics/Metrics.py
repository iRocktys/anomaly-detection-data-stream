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
        trueValues = np.asarray(yTrue, dtype=int)
        predictedValues = np.asarray(yPred, dtype=int)
        if trueValues.shape != predictedValues.shape:
            raise ValueError("yTrue e yPred devem possuir o mesmo tamanho.")

        tp = int(np.sum((trueValues == 1) & (predictedValues == 1)))
        tn = int(np.sum((trueValues == 0) & (predictedValues == 0)))
        fp = int(np.sum((trueValues == 0) & (predictedValues == 1)))
        fn = int(np.sum((trueValues == 1) & (predictedValues == 0)))

        return Metrics.fromCounts(
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
        )

    @staticmethod
    def fromCounts(tp, tn, fp, fn):
        tp = int(tp)
        tn = int(tn)
        fp = int(fp)
        fn = int(fn)

        if min(tp, tn, fp, fn) < 0:
            raise ValueError(
                "Os valores da matriz de confusão não podem ser negativos."
            )

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
    def evaluateFrame(frame, trueColumn="isAttack", predictionColumn="prediction", readyColumn=None):
        selected = frame
        if readyColumn and readyColumn in selected.columns:
            selected = selected[selected[readyColumn].astype(bool)]
        return Metrics.calculate(selected[trueColumn], selected[predictionColumn])

    @staticmethod
    def windowed(frame, windowSize=1000, trueColumn="isAttack", predictionColumn="prediction", readyColumn=None):
        windowSize = max(1, int(windowSize))
        rows = []
        for start in range(0, len(frame), windowSize):
            end = min(start + windowSize, len(frame))
            window = frame.iloc[start:end]
            if readyColumn and readyColumn in window.columns:
                window = window[window[readyColumn].astype(bool)]
            summary = Metrics.calculate(window[trueColumn], window[predictionColumn])
            summary.update({
                "windowStart": int(start),
                "windowEnd": int(end - 1),
            })
            rows.append(summary)
        columns = ["windowStart", "windowEnd"] + Metrics.metricNames
        return pd.DataFrame(rows, columns=columns)


    @staticmethod
    def attackBreakdown(frame, labelColumn="trueLabel", nameColumn="labelName", predictionColumn="prediction", normalClassIndex=0):
        rows = []
        if labelColumn not in frame.columns:
            return rows
        for labelValue, group in frame.groupby(labelColumn, sort=True):
            labelValue = int(labelValue)
            if labelValue == int(normalClassIndex):
                continue
            labelName = (
                str(group[nameColumn].iloc[0])
                if nameColumn in group.columns and not group.empty
                else str(labelValue)
            )
            total = int(len(group))
            detected = int((group[predictionColumn].astype(int) == 1).sum())
            rows.append({
                "trueLabel": labelValue,
                "labelName": labelName,
                "instances": total,
                "detected": detected,
                "falseNegatives": total - detected,
                "recall": Metrics.safeDivide(detected, total),
            })
        return rows

    @staticmethod
    def safeDivide(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0
