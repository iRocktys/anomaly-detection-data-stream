import math
from typing import Any, Iterable

from src.Anomaly.Thresholds.BaseThreshold import BaseThreshold


class FixedThreshold(BaseThreshold):
    """Limiar constante aplicado após o warmup global do experimento.

    A classe está sempre pronta internamente, mas o pipeline mantém o período
    global de aquecimento fora da avaliação e do gráfico. Depois do warmup, a
    predição é ataque quando o score ultrapassa ``value``.
    """

    def __init__(self, value: float = 0.5):
        self.initialValue = float(value)
        if not math.isfinite(self.initialValue):
            raise ValueError("value deve ser um número finito.")
        self.currentValue = self.initialValue

    def initialize(self, scores: Iterable[float]) -> None:
        return None

    def getThreshold(self) -> float:
        return float(self.currentValue)

    def update(self, score: float) -> None:
        return None

    def reset(self) -> None:
        self.currentValue = self.initialValue

    def isReady(self) -> bool:
        return True

    def getState(self) -> dict[str, Any]:
        return {
            "name": "fixed",
            "threshold": float(self.currentValue),
            "ready": True,
        }
