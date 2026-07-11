from abc import abstractmethod
from src.Anomaly.Thresholds.BaseThreshold import BaseThreshold

class IncrementalThreshold(BaseThreshold):
    @abstractmethod
    def update(self, score: float) -> None:
        raise NotImplementedError
