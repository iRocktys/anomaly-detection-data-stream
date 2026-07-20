import numpy as np
class BaseOnlineNormalizer:
    def __init__(self, epsilon=1e-8):
        self.epsilon = float(epsilon)

    def transform(self, values):
        return self.clean(values)

    def update(self, values):
        pass

    def reset(self):
        pass

    def clean(self, values):
        cleanValues = np.asarray(values, dtype=np.float64)
        return np.nan_to_num(
            cleanValues, 
            nan=0.0, 
            posinf=np.finfo(np.float32).max, 
            neginf=np.finfo(np.float32).min
        )
class NoOnlineNormalizer(BaseOnlineNormalizer):
    pass
class IncrementalZScoreNormalizer(BaseOnlineNormalizer):
    def __init__(self, epsilon=1e-8, clip=None):
        super().__init__(epsilon)
        self.clip = clip
        self.reset()

    def transform(self, values):
        cleanValues = self.clean(values)

        if self.count < 2:
            return np.zeros_like(cleanValues)

        variance = self.squareDistance / (self.count - 1)
        deviation = np.sqrt(np.maximum(variance, 0.0))
        safeDeviation = np.where(deviation < self.epsilon, 1.0, deviation)
        normalizedValues = (cleanValues - self.mean) / safeDeviation

        if self.clip is not None:
            normalizedValues = np.clip(normalizedValues, -float(self.clip), float(self.clip))

        return normalizedValues

    def update(self, values):
        cleanValues = self.clean(values)
        self.count += 1

        if self.count == 1:
            self.mean = cleanValues.copy()
            self.squareDistance = np.zeros_like(cleanValues)
            return

        difference = cleanValues - self.mean
        self.mean += difference / self.count
        secondDifference = cleanValues - self.mean
        self.squareDistance += difference * secondDifference

    def reset(self):
        self.count = 0
        self.mean = None
        self.squareDistance = None