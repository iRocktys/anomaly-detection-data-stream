from collections import deque
import numpy as np

class NoScoreSmoother:
    def transform(self, score):
        return float(score)

    def update(self, score):
        return None


class MovingAverageScoreSmoother:
    def __init__(self, windowSize=5):
        self.windowSize = max(1, int(windowSize))
        self.window = deque(maxlen=self.windowSize)

    def transform(self, score):
        values = list(self.window) + [float(score)]
        return float(np.mean(values))

    def update(self, score):
        self.window.append(float(score))


class ExponentialScoreSmoother:
    def __init__(self, alpha=0.1):
        self.alpha = float(alpha)
        self.currentScore = None

    def transform(self, score):
        scoreValue = float(score)
        if self.currentScore is None:
            return scoreValue
        return float((self.alpha * scoreValue) + ((1.0 - self.alpha) * self.currentScore))

    def update(self, score):
        self.currentScore = self.transform(score)


class ScoreSmoothers:
    @staticmethod
    def createSmoother(config):
        smootherName = str(config.name).strip().lower()
        if smootherName == "none":
            return NoScoreSmoother()
        if smootherName in {"movingaverage", "mean"}:
            return MovingAverageScoreSmoother(**config.parameters)
        if smootherName in {"exponential", "ewma"}:
            return ExponentialScoreSmoother(**config.parameters)
        raise ValueError(f"Suavizador de score desconhecido: {config.name}")
