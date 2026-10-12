from src.Metrics.Metrics import Metrics


class IncrementalMetrics:
    def __init__(self):
        # Inicializa o estado e as dependências necessárias para a execução da classe.
        self.reset()

    def update(self, trueLabel, predictedLabel):
        # Atualiza o estado incremental com os valores da instância atual.
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

        cumulative = self.snapshot()

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
        # Retorna uma cópia das métricas acumuladas no estado atual.
        return Metrics.fromCounts(
            tp=self.tp,
            tn=self.tn,
            fp=self.fp,
            fn=self.fn,
        )

    def reset(self):
        # Reinicia o estado interno para permitir uma nova execução.
        self.instances = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def safeDivide(self, numerator, denominator):
        # Executa uma divisão protegida contra denominadores iguais a zero.
        return Metrics.safeDivide(numerator, denominator)
