import math


class IncrementalMetrics:
    def __init__(self):
        self.reset()

    def update(self, trueLabel, predictedLabel):
        trueLabel = int(trueLabel)
        predictedLabel = int(predictedLabel)

        tp = int(trueLabel == 1 and predictedLabel == 1)
        tn = int(trueLabel == 0 and predictedLabel == 0)
        fp = int(trueLabel == 0 and predictedLabel == 1)
        fn = int(trueLabel == 1 and predictedLabel == 0)

        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn
        self.instances += 1

        precision = self.safeDivide(self.tp, self.tp + self.fp)
        recall = self.safeDivide(self.tp, self.tp + self.fn)
        specificity = self.safeDivide(self.tn, self.tn + self.fp)
        accuracy = self.safeDivide(self.tp + self.tn, self.instances)
        f1 = self.safeDivide(2.0 * precision * recall, precision + recall)
        denominator = math.sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))
        mcc = self.safeDivide((self.tp * self.tn) - (self.fp * self.fn), denominator)

        return {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "cumulativeTp": self.tp,
            "cumulativeTn": self.tn,
            "cumulativeFp": self.fp,
            "cumulativeFn": self.fn,
            "cumulativeAccuracy": accuracy,
            "cumulativePrecision": precision,
            "cumulativeRecall": recall,
            "cumulativeSpecificity": specificity,
            "cumulativeF1": f1,
            "cumulativeMcc": mcc,
        }

    def reset(self):
        self.instances = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def safeDivide(self, numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0