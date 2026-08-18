import numpy as np


class BaseOnlineNormalizer:
    def __init__(self, epsilon=1e-8):
        self.epsilon = float(epsilon)

    def transform(self, values):
        return self.prepareValues(values)

    def update(self, values, observedMask=None):
        return None

    def reset(self):
        return None

    def prepareValues(self, values):
        preparedValues = np.asarray(values, dtype=np.float64)

        if preparedValues.ndim != 1:
            raise ValueError("O normalizador aceita apenas vetores unidimensionais.")

        if np.any(~np.isfinite(preparedValues)):
            raise ValueError(
                "O normalizador recebeu valores não finitos. A imputação deve ocorrer antes da normalização."
            )

        return preparedValues.copy()

    def prepareObservedMask(self, observedMask, featureCount):
        if observedMask is None:
            return np.ones(featureCount, dtype=bool)

        preparedMask = np.asarray(observedMask, dtype=bool)

        if preparedMask.ndim != 1 or preparedMask.size != featureCount:
            raise ValueError(
                "A máscara de valores observados deve possuir uma posição para cada feature."
            )

        return preparedMask


class NoOnlineNormalizer(BaseOnlineNormalizer):
    pass


class IncrementalZScoreNormalizer(BaseOnlineNormalizer):
    def __init__(self, epsilon=1e-8, clip=None):
        super().__init__(epsilon)
        self.clip = clip
        self.reset()

    def transform(self, values):
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
        featureCount = int(featureCount)

        if self.counts is None:
            self.counts = np.zeros(featureCount, dtype=np.int64)
            self.means = np.zeros(featureCount, dtype=np.float64)
            self.squareDistances = np.zeros(featureCount, dtype=np.float64)
            return

        if self.counts.size != featureCount:
            raise ValueError(
                "A quantidade de features recebida pelo normalizador mudou durante a execução."
            )

    def reset(self):
        self.counts = None
        self.means = None
        self.squareDistances = None