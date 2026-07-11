from collections import deque

import numpy as np


class NoFeatureSmoother:
    def transform(self, values):
        return np.asarray(values, dtype=np.float64)

    def update(self, values):
        return None


class MovingAverageFeatureSmoother:
    def __init__(self, windowSize=5):
        self.windowSize = max(1, int(windowSize))
        self.window = deque(maxlen=self.windowSize)

    def transform(self, values):
        cleanValues = np.asarray(values, dtype=np.float64)
        if not self.window:
            return cleanValues
        history = np.asarray(self.window, dtype=np.float64)
        return np.mean(np.vstack([history, cleanValues]), axis=0)

    def update(self, values):
        self.window.append(np.asarray(values, dtype=np.float64).copy())


class FeatureSmoothers:
    @staticmethod
    def createSmoother(config):
        smootherName = str(config.name).strip().lower()
        if smootherName == "none":
            return NoFeatureSmoother()
        if smootherName in {"movingaverage", "mean"}:
            return MovingAverageFeatureSmoother(**config.parameters)
        raise ValueError(f"Suavizador de features desconhecido: {config.name}")
