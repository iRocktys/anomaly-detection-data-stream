from collections import deque

import numpy as np


class NoFeatureSmoother:
    def transform(self, values):
        return np.asarray(values, dtype=np.float64)

    def update(self, values):
        return None

    def reset(self):
        return None


class MovingAverageFeatureSmoother:
    def __init__(self, windowSize=5):
        self.windowSize = max(1, int(windowSize))
        self.reset()

    def transform(self, values):
        cleanValues = np.asarray(values, dtype=np.float64)
        if not self.window:
            return cleanValues
        history = np.asarray(self.window, dtype=np.float64)
        return np.mean(np.vstack([history, cleanValues]), axis=0)

    def update(self, values):
        self.window.append(np.asarray(values, dtype=np.float64).copy())

    def reset(self):
        self.window = deque(maxlen=self.windowSize)
