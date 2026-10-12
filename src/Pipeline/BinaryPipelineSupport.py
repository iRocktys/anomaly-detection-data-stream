import numpy as np


class BinaryPipelineSupport:
    def trainModel(self, modelInstance):
        # Treina o learner atual e respeita a configuração de propagação de erros de treinamento.
        try:
            self.model.train(modelInstance)
            return True
        except ValueError:
            if self.strictTrainingErrors:
                raise
            return False

    def emptyMetricValues(self):
        # Retorna métricas vazias sem modificar o acumulador binário durante regiões não avaliáveis.
        return {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "cumulativeTp": self.metrics.tp,
            "cumulativeTn": self.metrics.tn,
            "cumulativeFp": self.metrics.fp,
            "cumulativeFn": self.metrics.fn,
            "cumulativeAccuracy": np.nan,
            "cumulativePrecision": np.nan,
            "cumulativeRecall": np.nan,
            "cumulativeSpecificity": np.nan,
            "cumulativeF1": np.nan,
            "cumulativeMcc": np.nan,
        }
