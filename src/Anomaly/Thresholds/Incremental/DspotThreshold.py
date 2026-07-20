from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.Anomaly.Thresholds.BaseThreshold import BaseThreshold


@dataclass
class DspotConfig:
    driftDepth: int = 50
    calibrationSize: int = 1000
    initialQuantile: float = 0.98
    risk: float = 0.001
    refitEvery: int = 1
    optimizationStarts: int = 10
    tolerance: float = 1e-8

    @property
    def warmupSize(self):
        return self.driftDepth + self.calibrationSize

    def validate(self):
        if self.driftDepth < 2:
            raise ValueError("driftDepth deve ser maior ou igual a 2.")

        if self.calibrationSize < 20:
            raise ValueError("calibrationSize deve ser maior ou igual a 20.")

        if not 0.5 < self.initialQuantile < 1.0:
            raise ValueError("initialQuantile deve estar no intervalo (0.5, 1).")

        if not 0.0 < self.risk < 1.0:
            raise ValueError("risk deve estar no intervalo (0, 1).")

        if self.refitEvery < 1:
            raise ValueError("refitEvery deve ser maior ou igual a 1.")

        if self.optimizationStarts < 2:
            raise ValueError("optimizationStarts deve ser maior ou igual a 2.")

        if self.tolerance <= 0:
            raise ValueError("tolerance deve ser maior que zero.")


class DspotThreshold(BaseThreshold):
    def __init__(self, config=None):
        self.config = config if config is not None else DspotConfig()
        self.config.validate()
        self.reset()

    def initialize(self, scores):
        values = np.asarray(scores, dtype=np.float64)

        if len(values) != self.config.warmupSize:
            raise ValueError(f"O DSPOT deve receber exatamente {self.config.warmupSize} scores no aquecimento.")

        if np.any(~np.isfinite(values)):
            raise ValueError("O aquecimento do DSPOT contém scores não finitos.")

        driftValues = values[:self.config.driftDepth]
        calibrationValues = values[self.config.driftDepth:]

        self.normalHistory = [float(value) for value in driftValues]
        residuals = []

        for value in calibrationValues:
            drift = self.calculateDrift()
            residuals.append(float(value - drift))
            self.addNormalValue(value)

        residuals = np.asarray(residuals, dtype=np.float64)
        self.initialResidualThreshold = float(np.quantile(residuals, self.config.initialQuantile))
        self.excesses = [float(value - self.initialResidualThreshold) for value in residuals if value > self.initialResidualThreshold]
        self.observationCount = len(residuals)
        self.peakCount = len(self.excesses)

        if self.peakCount < 3:
            raise ValueError("O DSPOT encontrou menos de três picos durante a calibração.")

        self.fitTail()
        self.ready = True

    def getThreshold(self):
        if not self.ready:
            return np.nan

        return float(self.calculateDrift() + self.extremeResidualThreshold)

    def update(self, score, index=None):
        if not self.ready:
            raise RuntimeError("O DSPOT ainda não foi inicializado.")

        score = float(score)

        if not np.isfinite(score):
            raise ValueError("O DSPOT aceita somente scores finitos.")

        drift = self.calculateDrift()
        residual = score - drift
        classification = "normal"
        updatedTail = False

        if residual > self.extremeResidualThreshold:
            classification = "anomaly"

        elif residual > self.initialResidualThreshold:
            classification = "peak"
            excess = residual - self.initialResidualThreshold
            self.excesses.append(float(excess))
            self.peakCount += 1
            self.observationCount += 1
            self.peaksSinceFit += 1
            self.addNormalValue(score)

            if self.peaksSinceFit >= self.config.refitEvery:
                self.fitTail()
                self.peaksSinceFit = 0
                updatedTail = True

        else:
            self.observationCount += 1
            self.addNormalValue(score)

        return {
            "index": index,
            "drift": drift,
            "residual": residual,
            "classification": classification,
            "updatedTail": updatedTail,
            "threshold": self.getThreshold(),
            "shape": self.shape,
            "scale": self.scale,
        }

    def processValue(self, score, index=None):
        return self.update(score, index)

    def reset(self):
        self.initialResidualThreshold = np.nan
        self.extremeResidualThreshold = np.nan
        self.shape = np.nan
        self.scale = np.nan
        self.normalHistory = []
        self.excesses = []
        self.observationCount = 0
        self.peakCount = 0
        self.peaksSinceFit = 0
        self.ready = False

    def isReady(self):
        return bool(self.ready)

    def getState(self):
        return {
            "name": "dspot",
            "ready": self.ready,
            "warmupSize": self.config.warmupSize,
            "driftDepth": self.config.driftDepth,
            "calibrationSize": self.config.calibrationSize,
            "observationCount": self.observationCount,
            "peakCount": self.peakCount,
            "initialResidualThreshold": self.initialResidualThreshold,
            "extremeResidualThreshold": self.extremeResidualThreshold,
            "threshold": self.getThreshold(),
            "shape": self.shape,
            "scale": self.scale,
        }

    def calculateDrift(self):
        if not self.normalHistory:
            return 0.0

        return float(np.mean(self.normalHistory[-self.config.driftDepth:]))

    def addNormalValue(self, value):
        self.normalHistory.append(float(value))

        if len(self.normalHistory) > self.config.driftDepth:
            self.normalHistory.pop(0)

    def fitTail(self):
        excesses = np.asarray(self.excesses, dtype=np.float64)
        excesses = excesses[np.isfinite(excesses) & (excesses > 0)]

        if len(excesses) < 3:
            raise ValueError("Não existem excessos suficientes para ajustar a distribuição GPD.")

        candidates = []

        exponentialScale = float(np.mean(excesses))
        exponentialLikelihood = self.logLikelihood(excesses, 0.0, exponentialScale)
        candidates.append((exponentialLikelihood, 0.0, exponentialScale))

        maximum = float(np.max(excesses))
        minimum = float(np.min(excesses))
        mean = float(np.mean(excesses))
        intervals = [(-1.0 / maximum + self.config.tolerance, -self.config.tolerance)]

        if mean > minimum:
            positiveLower = 2.0 * (mean - minimum) / (mean * minimum)
            positiveUpper = 2.0 * (mean - minimum) / (minimum * minimum)

            if positiveUpper > positiveLower:
                intervals.append((positiveLower, positiveUpper))

        for lower, upper in intervals:
            initialValues = np.linspace(lower, upper, self.config.optimizationStarts)

            for initialValue in initialValues:
                result = minimize(self.objective, np.array([initialValue]), args=(excesses,), method="L-BFGS-B", bounds=[(lower, upper)])

                if not result.success:
                    continue

                root = float(result.x[0])
                functionValue = self.grimshawFunction(root, excesses)

                if not np.isfinite(functionValue) or abs(functionValue) > self.config.tolerance:
                    continue

                functionV = 1.0 + np.mean(np.log(1.0 + root * excesses))
                shape = float(functionV - 1.0)
                scale = float(shape / root)

                if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
                    continue

                likelihood = self.logLikelihood(excesses, shape, scale)

                if np.isfinite(likelihood):
                    candidates.append((likelihood, shape, scale))

        bestLikelihood, self.shape, self.scale = max(candidates, key=lambda candidate: candidate[0])
        ratio = self.config.risk * self.observationCount / self.peakCount

        if np.isclose(self.shape, 0.0):
            self.extremeResidualThreshold = self.initialResidualThreshold + self.scale * np.log(self.peakCount / (self.config.risk * self.observationCount))
        else:
            self.extremeResidualThreshold = self.initialResidualThreshold + self.scale * (ratio ** (-self.shape) - 1.0) / self.shape

        if not np.isfinite(bestLikelihood) or not np.isfinite(self.extremeResidualThreshold):
            raise RuntimeError("O ajuste do DSPOT produziu valores inválidos.")

    def grimshawFunction(self, value, excesses):
        terms = 1.0 + value * excesses

        if np.any(terms <= 0):
            return np.nan

        functionU = np.mean(1.0 / terms)
        functionV = 1.0 + np.mean(np.log(terms))

        return float(functionU * functionV - 1.0)

    def objective(self, value, excesses):
        scalarValue = float(np.asarray(value).reshape(-1)[0])
        functionValue = self.grimshawFunction(scalarValue, excesses)

        if not np.isfinite(functionValue):
            return 1e100

        return float(functionValue * functionValue)

    def logLikelihood(self, excesses, shape, scale):
        if scale <= 0:
            return -np.inf

        if np.isclose(shape, 0.0):
            return float(-len(excesses) * np.log(scale) - np.sum(excesses) / scale)

        support = 1.0 + shape * excesses / scale

        if np.any(support <= 0):
            return -np.inf

        return float(-len(excesses) * np.log(scale) - (1.0 + 1.0 / shape) * np.sum(np.log(support)))


DSPOT = DspotThreshold
DSPOTConfig = DspotConfig