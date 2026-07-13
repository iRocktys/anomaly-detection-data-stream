from collections import deque
from typing import Any, Iterable

import numpy as np

from src.Anomaly.Thresholds.IncrementalThreshold import IncrementalThreshold
from src.Anomaly.Thresholds.Incremental.SpotThreshold import SpotThreshold


class DspotThreshold(IncrementalThreshold):
    """DSPOT causal com remoção de drift por média móvel anterior ao ponto atual."""

    def __init__(
        self,
        risk: float = 0.001,
        initialQuantile: float = 0.98,
        minimumSamples: int = 200,
        driftDepth: int = 200,
        refitEvery: int = 25,
    ):
        self.risk = float(risk)
        self.initialQuantile = float(initialQuantile)
        self.minimumSamples = max(20, int(minimumSamples))
        self.driftDepth = max(2, int(driftDepth))
        self.refitEvery = max(1, int(refitEvery))
        self.reset()

    def initialize(self, scores: Iterable[float]) -> None:
        for score in scores:
            self.update(float(score))

    def currentDrift(self):
        if not self.history:
            return 0.0
        return float(np.mean(self.history))

    def getThreshold(self) -> float:
        return self.currentDrift() + self.spot.getThreshold()

    def update(self, score: float) -> None:
        value = float(score)
        drift = self.currentDrift()
        residual = value - drift
        self.spot.update(residual)
        self.history.append(value)
        self.count += 1

    def reset(self) -> None:
        self.history = deque(maxlen=self.driftDepth)
        self.spot = SpotThreshold(
            risk=self.risk,
            initialQuantile=self.initialQuantile,
            minimumSamples=self.minimumSamples,
            refitEvery=self.refitEvery,
        )
        self.count = 0

    def isReady(self) -> bool:
        return self.spot.isReady()

    def getState(self) -> dict[str, Any]:
        state = self.spot.getState()
        return {
            "name": "dspot",
            "ready": self.isReady(),
            "count": self.count,
            "drift": self.currentDrift(),
            "driftDepth": self.driftDepth,
            "threshold": self.getThreshold(),
            "spot": state,
        }
