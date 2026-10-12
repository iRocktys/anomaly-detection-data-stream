import numpy as np


class BaseOnlineNormalizer:
    def __init__(self, epsilon=1e-8):
        # Inicializa o estado e as dependências necessárias para a execução da classe.
        self.epsilon = float(epsilon)

    def transform(self, values):
        # Transforma os valores de entrada conforme a estratégia incremental configurada.
        return self.prepareValues(values)

    def update(self, values, observedMask=None):
        # Atualiza o estado incremental com os valores da instância atual.
        return None

    def reset(self):
        # Reinicia o estado interno para permitir uma nova execução.
        return None

    def prepareValues(self, values):
        # Converte e valida os valores de entrada antes do processamento incremental.
        preparedValues = np.asarray(values, dtype=np.float64)

        if preparedValues.ndim != 1:
            raise ValueError("The normalizer accepts only one-dimensional vectors.")

        if np.any(~np.isfinite(preparedValues)):
            raise ValueError(
                "The normalizer received non-finite values. Imputation must occur before normalization."
            )

        return preparedValues.copy()

    def prepareObservedMask(self, observedMask, featureCount):
        # Prepara a máscara que identifica as features observadas na instância atual.
        if observedMask is None:
            return np.ones(featureCount, dtype=bool)

        preparedMask = np.asarray(observedMask, dtype=bool)

        if preparedMask.ndim != 1 or preparedMask.size != featureCount:
            raise ValueError(
                "The observed-value mask must contain one position for each feature."
            )

        return preparedMask


class IncrementalZScoreNormalizer(BaseOnlineNormalizer):
    def __init__(self, epsilon=1e-8, clip=None):
        # Inicializa o estado e as dependências necessárias para a execução da classe.
        super().__init__(epsilon)
        self.clip = clip
        self.reset()

    def transform(self, values):
        # Transforma os valores de entrada conforme a estratégia incremental configurada.
        preparedValues = self.prepareValues(values)
        self.ensureState(preparedValues.size)

        normalizedValues = np.zeros_like(preparedValues)
        readyMask = self.counts >= 2

        if np.any(readyMask):
            variance = self.squareDistances[readyMask] / (self.counts[readyMask] - 1)
            deviation = np.sqrt(np.maximum(variance, 0.0))
            safeDeviation = np.where(deviation < self.epsilon, 1.0, deviation)
            normalizedValues[readyMask] = (
                preparedValues[readyMask] - self.means[readyMask]
            ) / safeDeviation

        if self.clip is not None:
            normalizedValues = np.clip(
                normalizedValues,
                -float(self.clip),
                float(self.clip),
            )

        return normalizedValues

    def update(self, values, observedMask=None):
        # Atualiza o estado incremental com os valores da instância atual.
        preparedValues = self.prepareValues(values)
        self.ensureState(preparedValues.size)
        preparedMask = self.prepareObservedMask(
            observedMask,
            preparedValues.size,
        )

        if not np.any(preparedMask):
            return

        updatedCounts = self.counts[preparedMask] + 1
        differences = preparedValues[preparedMask] - self.means[preparedMask]
        updatedMeans = self.means[preparedMask] + differences / updatedCounts
        secondDifferences = preparedValues[preparedMask] - updatedMeans

        self.squareDistances[preparedMask] += differences * secondDifferences
        self.means[preparedMask] = updatedMeans
        self.counts[preparedMask] = updatedCounts

    def ensureState(self, featureCount):
        # Inicializa ou valida o estado interno de acordo com a quantidade de features.
        featureCount = int(featureCount)

        if self.counts is None:
            self.counts = np.zeros(featureCount, dtype=np.int64)
            self.means = np.zeros(featureCount, dtype=np.float64)
            self.squareDistances = np.zeros(featureCount, dtype=np.float64)
            return

        if self.counts.size != featureCount:
            raise ValueError(
                "The number of features received by the normalizer changed during execution."
            )

    def reset(self):
        # Reinicia o estado interno para permitir uma nova execução.
        self.counts = None
        self.means = None
        self.squareDistances = None
