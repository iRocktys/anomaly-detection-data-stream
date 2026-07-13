import math
from typing import Any, Iterable

import numpy as np

from src.Anomaly.Thresholds.IncrementalThreshold import IncrementalThreshold


class SpotThreshold(IncrementalThreshold):
    """Streaming Peaks-Over-Threshold para cauda superior.

    A implementação usa ajuste GPD por momentos com fallback exponencial. O
    threshold base é definido pelo quantil inicial e apenas excessos entre o
    threshold base e o extremo atual atualizam a distribuição de picos.
    """

    def __init__(
        self,
        risk: float = 0.001,
        initialQuantile: float = 0.98,
        minimumSamples: int = 200,
        refitEvery: int = 25,
    ):
        self.risk = float(risk)
        self.initialQuantile = float(initialQuantile)
        self.minimumSamples = max(20, int(minimumSamples))
        self.refitEvery = max(1, int(refitEvery))
        if not 0.0 < self.risk < 1.0:
            raise ValueError("risk deve estar no intervalo (0, 1).")
        if not 0.5 < self.initialQuantile < 1.0:
            raise ValueError("initialQuantile deve estar no intervalo (0.5, 1).")
        self.reset()

    def initialize(self, scores: Iterable[float]) -> None:
        for score in scores:
            self.update(float(score))

    def getThreshold(self) -> float:
        if not self.isReady():
            return self.initialThreshold if self.initialValues else math.inf
        return float(self.extremeThreshold)

    def update(self, score: float) -> None:
        value = float(score)
        self.count += 1

        if not self.ready:
            self.initialValues.append(value)
            if len(self.initialValues) >= self.minimumSamples:
                self.fitInitialModel()
            return

        if value > self.extremeThreshold:
            self.anomalyCount += 1
            return

        if value > self.initialThreshold:
            self.peaks.append(value - self.initialThreshold)
            self.peaksSinceFit += 1
            if self.peaksSinceFit >= self.refitEvery:
                self.fitTail()

    def fitInitialModel(self):
        values = np.asarray(self.initialValues, dtype=np.float64)
        self.initialThreshold = float(np.quantile(values, self.initialQuantile))
        self.peaks = [float(value - self.initialThreshold) for value in values if value > self.initialThreshold]
        if not self.peaks:
            self.peaks = [max(float(np.std(values)), 1e-8)]
        self.ready = True
        self.fitTail()

    def fitTail(self):
        peaks = np.asarray(self.peaks, dtype=np.float64)
        peaks = peaks[np.isfinite(peaks) & (peaks > 0)]
        if peaks.size == 0:
            self.shape = 0.0
            self.scale = 1e-8
            self.extremeThreshold = self.initialThreshold
            self.peaksSinceFit = 0
            return

        mean = float(np.mean(peaks))
        variance = float(np.var(peaks, ddof=1)) if peaks.size > 1 else 0.0

        if variance > mean * mean and variance > 1e-16:
            shape = 0.5 * (1.0 - ((mean * mean) / variance))
            shape = float(np.clip(shape, -0.45, 0.45))
            scale = 0.5 * mean * (1.0 + ((mean * mean) / variance))
        else:
            shape = 0.0
            scale = mean

        self.shape = shape
        self.scale = max(float(scale), 1e-8)
        peakRate = max(len(peaks) / max(self.count, 1), 1e-12)
        ratio = max(self.risk / peakRate, 1e-12)

        if abs(self.shape) < 1e-8:
            excess = -self.scale * math.log(ratio)
        else:
            excess = (self.scale / self.shape) * (ratio ** (-self.shape) - 1.0)

        self.extremeThreshold = max(self.initialThreshold, self.initialThreshold + float(excess))
        self.peaksSinceFit = 0

    def reset(self) -> None:
        self.count = 0
        self.anomalyCount = 0
        self.initialValues = []
        self.initialThreshold = math.inf
        self.extremeThreshold = math.inf
        self.peaks = []
        self.peaksSinceFit = 0
        self.shape = 0.0
        self.scale = 0.0
        self.ready = False

    def isReady(self) -> bool:
        return bool(self.ready)

    def getState(self) -> dict[str, Any]:
        return {
            "name": "spot",
            "ready": self.isReady(),
            "count": self.count,
            "anomalyCount": self.anomalyCount,
            "initialThreshold": self.initialThreshold,
            "threshold": self.getThreshold(),
            "peakCount": len(self.peaks),
            "shape": self.shape,
            "scale": self.scale,
            "risk": self.risk,
            "initialQuantile": self.initialQuantile,
            "minimumSamples": self.minimumSamples,
        }
