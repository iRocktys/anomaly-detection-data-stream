from typing import Any, Iterable
from src.Anomaly.Thresholds.BaseThreshold import BaseThreshold

class FixedThreshold(BaseThreshold):
    def __init__(self, value: float = 0.5):
        self.initialValue = float(value)
        self.currentValue = float(value)

    def initialize(self, scores: Iterable[float]) -> None:
        return None

    def getThreshold(self) -> float:
        return self.currentValue

    def update(self, score: float) -> None:
        return None

    def reset(self) -> None:
        self.currentValue = self.initialValue

    def isReady(self) -> bool:
        return True

    def getState(self) -> dict[str, Any]:
        return {
            "name": "fixed",
            "threshold": self.currentValue,
            "ready": True,
        }
