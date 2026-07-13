from collections import deque

import numpy as np


class NoScoreSmoother:
    def transform(self, score):
        return float(score)

    def update(self, score):
        return None

    def reset(self):
        return None


class MovingAverageScoreSmoother:
    def __init__(self, windowSize=5):
        self.windowSize = max(1, int(windowSize))
        self.reset()

    def transform(self, score):
        values = list(self.window) + [float(score)]
        return float(np.mean(values))

    def update(self, score):
        self.window.append(float(score))

    def reset(self):
        self.window = deque(maxlen=self.windowSize)


class ExponentialScoreSmoother:
    def __init__(self, alpha=0.1):
        self.alpha = float(alpha)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha deve estar no intervalo (0, 1].")
        self.reset()

    def transform(self, score):
        value = float(score)
        if self.currentScore is None:
            return value
        return float((self.alpha * value) + ((1.0 - self.alpha) * self.currentScore))

    def update(self, score):
        self.currentScore = self.transform(score)

    def reset(self):
        self.currentScore = None
