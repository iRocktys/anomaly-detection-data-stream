import math
from collections import deque
from typing import Any, Iterable

import numpy as np

from src.Anomaly.Thresholds.IncrementalThreshold import IncrementalThreshold


class DspotThreshold(IncrementalThreshold):
    """DSPOT causal para a cauda superior de uma sequência de scores.

    O período de calibração da cauda é recebido pelo parâmetro ``warmup`` e
    deve ser exatamente o mesmo warmup global usado pelo experimento. Durante
    esse período o limiar ainda não está pronto; os resíduos são acumulados e
    a distribuição Generalized Pareto (GPD) é inicializada somente ao final do
    aquecimento.

    A lógica Peaks-Over-Threshold necessária ao DSPOT está contida nesta
    classe. SPOT não é exposto como uma técnica independente no projeto.
    """

    def __init__(
        self,
        risk: float = 0.001,
        initialQuantile: float = 0.98,
        warmup: int = 200,
        driftDepth: int | None = None,
        refitEvery: int = 25,
    ):
        self.risk = float(risk)
        self.initialQuantile = float(initialQuantile)
        self.warmup = int(warmup)
        self.driftDepth = int(driftDepth) if driftDepth is not None else self.warmup
        self.refitEvery = int(refitEvery)
        self.validateParameters()
        self.reset()

    def validateParameters(self):
        if not 0.0 < self.risk < 1.0:
            raise ValueError("risk deve estar no intervalo (0, 1).")
        if not 0.5 < self.initialQuantile < 1.0:
            raise ValueError("initialQuantile deve estar no intervalo (0.5, 1).")
        if self.warmup < 20:
            raise ValueError("warmup deve ser maior ou igual a 20 para calibrar a cauda.")
        if self.driftDepth < 2:
            raise ValueError("driftDepth deve ser maior ou igual a 2.")
        if self.refitEvery < 1:
            raise ValueError("refitEvery deve ser maior ou igual a 1.")

    def initialize(self, scores: Iterable[float]) -> None:
        for score in scores:
            self.update(float(score))

    def currentDrift(self) -> float:
        if not self.history:
            return 0.0
        return float(np.mean(self.history))

    def getThreshold(self) -> float:
        if not self.isReady():
            return math.nan
        return float(self.currentDrift() + self.extremeResidualThreshold)

    def update(self, score: float) -> None:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("O DSPOT aceita somente scores finitos.")

        drift = self.currentDrift()
        residual = value - drift
        self.count += 1

        if not self.ready:
            self.initialResiduals.append(residual)
            self.history.append(value)
            if self.count >= self.warmup:
                self.fitInitialTail()
            return

        if residual > self.extremeResidualThreshold:
            self.anomalyCount += 1
        elif residual > self.initialResidualThreshold:
            self.peaks.append(residual - self.initialResidualThreshold)
            self.peaksSinceFit += 1
            if self.peaksSinceFit >= self.refitEvery:
                self.fitTail()

        self.updateDriftHistory(value)

    def updateDriftHistory(self, value: float) -> None:
        """Atualiza a estimativa local de drift com o comportamento atual.

        Pontos para evolução futura, sem inverter o sentido do limiar:

        1. O artigo original mantém observações acima do limiar extremo fora da
           atualização da cauda. Esta classe já faz isso: valores extremos não
           entram em ``peaks`` nem no ajuste GPD.
        2. Para impedir que ataques prolongados elevem a média local de drift,
           esta atualização pode futuramente ignorar valores classificados como
           extremos, inserir o valor limitado ao limiar atual (winsorização) ou
           usar mediana/média aparada no lugar da média simples.
        3. Outra alternativa é atualizar o drift por EWMA robusta e congelar a
           atualização enquanto o detector estiver em estado de ataque.
        4. Essas mudanças preservam a regra correta para scores de anomalia:
           ataque quando ``score > threshold``. Inverter a onda do threshold
           destruiria a interpretação probabilística da cauda extrema.

        A implementação atual mantém o histórico completo para preservar o
        comportamento causal já utilizado nos experimentos. O método isolado
        permite aplicar posteriormente uma das políticas robustas acima sem
        alterar o ajuste da cauda.
        """
        self.history.append(float(value))

    def fitInitialTail(self) -> None:
        values = np.asarray(self.initialResiduals, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < self.warmup:
            return

        self.initialResidualThreshold = float(
            np.quantile(values, self.initialQuantile)
        )
        self.peaks = [
            float(value - self.initialResidualThreshold)
            for value in values
            if value > self.initialResidualThreshold
        ]
        if not self.peaks:
            self.peaks = [max(float(np.std(values)), 1e-8)]
        self.ready = True
        self.fitTail()

    def fitTail(self) -> None:
        peaks = np.asarray(self.peaks, dtype=np.float64)
        peaks = peaks[np.isfinite(peaks) & (peaks > 0)]
        if peaks.size == 0:
            self.shape = 0.0
            self.scale = 1e-8
            self.extremeResidualThreshold = self.initialResidualThreshold
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
            excess = (
                self.scale
                / self.shape
                * (ratio ** (-self.shape) - 1.0)
            )

        self.extremeResidualThreshold = max(
            self.initialResidualThreshold,
            self.initialResidualThreshold + float(excess),
        )
        self.peaksSinceFit = 0

    def reset(self) -> None:
        self.count = 0
        self.anomalyCount = 0
        self.history = deque(maxlen=self.driftDepth)
        self.initialResiduals = []
        self.initialResidualThreshold = math.nan
        self.extremeResidualThreshold = math.nan
        self.peaks = []
        self.peaksSinceFit = 0
        self.shape = 0.0
        self.scale = 0.0
        self.ready = False

    def isReady(self) -> bool:
        return bool(self.ready)

    def getState(self) -> dict[str, Any]:
        return {
            "name": "dspot",
            "ready": self.isReady(),
            "count": self.count,
            "warmup": self.warmup,
            "warmupRemaining": max(0, self.warmup - self.count),
            "anomalyCount": self.anomalyCount,
            "drift": self.currentDrift(),
            "driftDepth": self.driftDepth,
            "initialResidualThreshold": self.initialResidualThreshold,
            "extremeResidualThreshold": self.extremeResidualThreshold,
            "threshold": self.getThreshold(),
            "peakCount": len(self.peaks),
            "peaksSinceFit": self.peaksSinceFit,
            "shape": self.shape,
            "scale": self.scale,
            "risk": self.risk,
            "initialQuantile": self.initialQuantile,
            "refitEvery": self.refitEvery,
        }
