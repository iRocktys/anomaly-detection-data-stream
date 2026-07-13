from collections import deque

import numpy as np


class BaseOnlineNormalizer:
    def __init__(self, epsilon=1e-8):
        self.epsilon = float(epsilon)

    def transform(self, values):
        return self.clean(values)

    def update(self, values):
        return None

    def reset(self):
        return None

    def clean(self, values):
        cleanValues = np.asarray(values, dtype=np.float64)
        return np.nan_to_num(
            cleanValues,
            nan=0.0,
            posinf=np.finfo(np.float32).max,
            neginf=np.finfo(np.float32).min,
        )


class NoOnlineNormalizer(BaseOnlineNormalizer):
    pass


class IncrementalMinMaxNormalizer(BaseOnlineNormalizer):
    def __init__(self, epsilon=1e-8, clip=True):
        super().__init__(epsilon)
        self.clip = bool(clip)
        self.reset()

    def transform(self, values):
        cleanValues = self.clean(values)
        if self.count == 0:
            return np.zeros_like(cleanValues)
        denominator = self.maximum - self.minimum
        safeDenominator = np.where(np.abs(denominator) < self.epsilon, 1.0, denominator)
        normalized = (cleanValues - self.minimum) / safeDenominator
        return np.clip(normalized, 0.0, 1.0) if self.clip else normalized

    def update(self, values):
        cleanValues = self.clean(values)
        if self.count == 0:
            self.minimum = cleanValues.copy()
            self.maximum = cleanValues.copy()
        else:
            self.minimum = np.minimum(self.minimum, cleanValues)
            self.maximum = np.maximum(self.maximum, cleanValues)
        self.count += 1

    def reset(self):
        self.minimum = None
        self.maximum = None
        self.count = 0


class IncrementalZScoreNormalizer(BaseOnlineNormalizer):
    def __init__(self, epsilon=1e-8, clip=None):
        super().__init__(epsilon)
        self.clip = clip
        self.reset()

    def transform(self, values):
        cleanValues = self.clean(values)
        if self.count < 2:
            return np.zeros_like(cleanValues)
        variance = self.squareDistance / max(self.count - 1, 1)
        deviation = np.sqrt(np.maximum(variance, 0.0))
        safeDeviation = np.where(deviation < self.epsilon, 1.0, deviation)
        normalized = (cleanValues - self.mean) / safeDeviation
        if self.clip is not None:
            normalized = np.clip(normalized, -float(self.clip), float(self.clip))
        return normalized

    def update(self, values):
        cleanValues = self.clean(values)
        if self.count == 0:
            self.count = 1
            self.mean = cleanValues.copy()
            self.squareDistance = np.zeros_like(cleanValues)
            return
        self.count += 1
        difference = cleanValues - self.mean
        self.mean = self.mean + (difference / self.count)
        secondDifference = cleanValues - self.mean
        self.squareDistance = self.squareDistance + (difference * secondDifference)

    def reset(self):
        self.count = 0
        self.mean = None
        self.squareDistance = None


class RollingMinMaxNormalizer(BaseOnlineNormalizer):
    def __init__(self, windowSize=200, epsilon=1e-8, clip=True):
        super().__init__(epsilon)
        self.windowSize = max(2, int(windowSize))
        self.clip = bool(clip)
        self.reset()

    def transform(self, values):
        cleanValues = self.clean(values)
        if not self.window:
            return np.zeros_like(cleanValues)
        history = np.asarray(self.window, dtype=np.float64)
        minimum = np.min(history, axis=0)
        maximum = np.max(history, axis=0)
        denominator = maximum - minimum
        safeDenominator = np.where(np.abs(denominator) < self.epsilon, 1.0, denominator)
        normalized = (cleanValues - minimum) / safeDenominator
        return np.clip(normalized, 0.0, 1.0) if self.clip else normalized

    def update(self, values):
        self.window.append(self.clean(values).copy())

    def reset(self):
        self.window = deque(maxlen=self.windowSize)


class RollingZScoreNormalizer(BaseOnlineNormalizer):
    def __init__(self, windowSize=200, epsilon=1e-8, clip=None):
        super().__init__(epsilon)
        self.windowSize = max(2, int(windowSize))
        self.clip = clip
        self.reset()

    def transform(self, values):
        cleanValues = self.clean(values)
        if len(self.window) < 2:
            return np.zeros_like(cleanValues)
        history = np.asarray(self.window, dtype=np.float64)
        mean = np.mean(history, axis=0)
        deviation = np.std(history, axis=0, ddof=1)
        safeDeviation = np.where(deviation < self.epsilon, 1.0, deviation)
        normalized = (cleanValues - mean) / safeDeviation
        if self.clip is not None:
            normalized = np.clip(normalized, -float(self.clip), float(self.clip))
        return normalized

    def update(self, values):
        self.window.append(self.clean(values).copy())

    def reset(self):
        self.window = deque(maxlen=self.windowSize)
