from src.Metrics.Metrics import Metrics


class IncrementalMetrics:
    def __init__(self):
        # Inicializa o acumulador incremental da matriz de confusão binária.
        self.reset()

    def reset(self):
        # Zera todas as contagens acumuladas da avaliação binária.
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.instances = 0

    def update(self, trueLabel, predictedLabel):
        # Atualiza uma única observação binária e retorna métricas instantâneas e acumuladas.
        true_label = int(trueLabel)
        predicted_label = int(predictedLabel)
        if true_label not in (0, 1) or predicted_label not in (0, 1):
            raise ValueError("Incremental labels must be binary values 0 and 1.")
        tp = int(true_label == 1 and predicted_label == 1)
        tn = int(true_label == 0 and predicted_label == 0)
        fp = int(true_label == 0 and predicted_label == 1)
        fn = int(true_label == 1 and predicted_label == 0)
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn
        self.instances += 1
        cumulative = Metrics.fromCounts(self.tp, self.tn, self.fp, self.fn)
        return {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "cumulativeTp": self.tp,
            "cumulativeTn": self.tn,
            "cumulativeFp": self.fp,
            "cumulativeFn": self.fn,
            "cumulativeAccuracy": cumulative["accuracy"],
            "cumulativePrecision": cumulative["precision"],
            "cumulativeRecall": cumulative["recall"],
            "cumulativeSpecificity": cumulative["specificity"],
            "cumulativeF1": cumulative["f1"],
            "cumulativeMcc": cumulative["mcc"],
        }

    def snapshot(self):
        # Retorna uma cópia lógica das métricas acumuladas no instante atual.
        return Metrics.fromCounts(self.tp, self.tn, self.fp, self.fn)
