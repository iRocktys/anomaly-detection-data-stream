from typing import Any, Iterable

from src.Anomaly.Thresholds.IncrementalThreshold import IncrementalThreshold


class DspotThreshold(IncrementalThreshold):
    def __init__(self, risk: float = 0.001, initialQuantile: float = 0.98, minimumSamples: int = 200, driftDepth: int = 200):
        self.risk = float(risk)
        self.initialQuantile = float(initialQuantile)
        self.minimumSamples = max(20, int(minimumSamples))
        self.driftDepth = max(2, int(driftDepth))
        self.reset()

    def initialize(self, scores: Iterable[float]) -> None:
        raise NotImplementedError("DSPOT ainda não foi implementado. Este módulo reserva seu contrato incremental.")

    def getThreshold(self) -> float:
        raise NotImplementedError("DSPOT ainda não foi implementado.")

    def update(self, score: float) -> None:
        raise NotImplementedError("DSPOT ainda não foi implementado.")

    def reset(self) -> None:
        self.count = 0

    def isReady(self) -> bool:
        return False

    def getState(self) -> dict[str, Any]:
        return {
            "name": "dspot",
            "ready": False,
            "risk": self.risk,
            "initialQuantile": self.initialQuantile,
            "minimumSamples": self.minimumSamples,
            "driftDepth": self.driftDepth,
        }
