import math

from src.Anomaly.Thresholds.BaseThreshold import BaseThreshold


class FixedThreshold(BaseThreshold):
    def __init__(self, value=0.5):
        self.initialValue = float(value)

        if not math.isfinite(self.initialValue):
            raise ValueError("O threshold fixo deve ser um número finito.")

        self.reset()

    def initialize(self, scores=None):
        return None

    def getThreshold(self):
        return float(self.currentValue)

    def update(self, score, index=None):
        return None

    def reset(self):
        self.currentValue = self.initialValue

    def isReady(self):
        return True

    def getState(self):
        return {
            "name": "fixed",
            "ready": True,
            "threshold": float(self.currentValue),
        }