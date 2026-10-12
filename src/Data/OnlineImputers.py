import numpy as np


class BaseOnlineImputer:
    name = "base"

    def transform(self, values):
        # Transforma os valores de entrada conforme a estratégia incremental configurada.
        raise NotImplementedError

    def update(self, values):
        # Atualiza o estado incremental com os valores da instância atual.
        raise NotImplementedError

    def reset(self):
        # Reinicia o estado interno para permitir uma nova execução.
        raise NotImplementedError

    def prepareValues(self, values):
        # Converte e valida os valores de entrada antes do processamento incremental.
        preparedValues = np.asarray(values, dtype=np.float64)

        if preparedValues.ndim != 1:
            raise ValueError("The imputer accepts only one-dimensional vectors.")

        return preparedValues.copy()


class IncrementalMeanImputer(BaseOnlineImputer):
    name = "incrementalMean"

    def __init__(self):
        # Inicializa o estado e as dependências necessárias para a execução da classe.
        self.reset()

    def transform(self, values):
        # Transforma os valores de entrada conforme a estratégia incremental configurada.
        imputedValues = self.prepareValues(values)
        self.ensureState(imputedValues.size)

        missingMask = ~np.isfinite(imputedValues)
        imputedValues[missingMask] = self.means[missingMask]

        return imputedValues

    def update(self, values):
        # Atualiza o estado incremental com os valores da instância atual.
        observedValues = self.prepareValues(values)
        self.ensureState(observedValues.size)

        observedMask = np.isfinite(observedValues)

        if not np.any(observedMask):
            return

        updatedCounts = self.counts[observedMask] + 1
        differences = observedValues[observedMask] - self.means[observedMask]

        self.means[observedMask] += differences / updatedCounts
        self.counts[observedMask] = updatedCounts

    def ensureState(self, featureCount):
        # Inicializa ou valida o estado interno de acordo com a quantidade de features.
        featureCount = int(featureCount)

        if self.counts is None:
            self.counts = np.zeros(featureCount, dtype=np.int64)
            self.means = np.zeros(featureCount, dtype=np.float64)
            return

        if self.counts.size != featureCount:
            raise ValueError(
                "The number of features received by the imputer changed during execution."
            )

    def reset(self):
        # Reinicia o estado interno para permitir uma nova execução.
        self.counts = None
        self.means = None
