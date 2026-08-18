import numpy as np


class BaseOnlineImputer:
    name = "base"

    def transform(self, values):
        raise NotImplementedError

    def update(self, values):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def prepareValues(self, values):
        preparedValues = np.asarray(values, dtype=np.float64)

        if preparedValues.ndim != 1:
            raise ValueError("O imputador aceita apenas vetores unidimensionais.")

        return preparedValues.copy()


class ZeroOnlineImputer(BaseOnlineImputer):
    name = "zero"

    def transform(self, values):
        imputedValues = self.prepareValues(values)
        imputedValues[~np.isfinite(imputedValues)] = 0.0

        return imputedValues

    def update(self, values):
        self.prepareValues(values)

    def reset(self):
        return None


class IncrementalMeanImputer(BaseOnlineImputer):
    name = "incrementalMean"

    def __init__(self):
        self.reset()

    def transform(self, values):
        imputedValues = self.prepareValues(values)
        self.ensureState(imputedValues.size)

        missingMask = ~np.isfinite(imputedValues)
        imputedValues[missingMask] = self.means[missingMask]

        return imputedValues

    def update(self, values):
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
        featureCount = int(featureCount)

        if self.counts is None:
            self.counts = np.zeros(featureCount, dtype=np.int64)
            self.means = np.zeros(featureCount, dtype=np.float64)
            return

        if self.counts.size != featureCount:
            raise ValueError(
                "A quantidade de features recebida pelo imputador mudou durante a execução."
            )

    def reset(self):
        self.counts = None
        self.means = None