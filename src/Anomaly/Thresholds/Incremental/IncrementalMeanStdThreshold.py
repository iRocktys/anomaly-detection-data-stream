import math
from typing import Any, Iterable

from src.Anomaly.Thresholds.IncrementalThreshold import IncrementalThreshold


class IncrementalMeanStdThreshold(IncrementalThreshold):
    def __init__(self, standardDeviations: float = 3.0, minimumSamples: int = 30, initialValue: float = 0.5):
        self.standardDeviations = float(standardDeviations)
        self.minimumSamples = max(2, int(minimumSamples))
        self.initialValue = float(initialValue)
        self.reset()

    def initialize(self, scores: Iterable[float]) -> None:
        for score in scores:
            self.update(float(score))

    def getThreshold(self) -> float:
        if not self.isReady():
            return self.initialValue
        variance = self.squareDistance / self.count
        standardDeviation = math.sqrt(max(variance, 0.0))
        return self.mean + (self.standardDeviations * standardDeviation)

    def update(self, score: float) -> None:
        value = float(score)
        self.count += 1
        difference = value - self.mean
        self.mean += difference / self.count
        secondDifference = value - self.mean
        self.squareDistance += difference * secondDifference

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.squareDistance = 0.0

    def isReady(self) -> bool:
        return self.count >= self.minimumSamples

    def getState(self) -> dict[str, Any]:
        variance = self.squareDistance / self.count if self.count else 0.0
        return {
            "name": "incrementalMeanStd",
            "threshold": self.getThreshold(),
            "ready": self.isReady(),
            "count": self.count,
            "mean": self.mean,
            "standardDeviation": math.sqrt(max(variance, 0.0)),
            "standardDeviations": self.standardDeviations,
        }
