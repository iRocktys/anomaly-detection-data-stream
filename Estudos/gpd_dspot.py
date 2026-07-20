from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class GeneralizedPareto:
    @staticmethod
    def cdf(values, shape, scale):
        values = np.asarray(values, dtype=np.float64)

        if scale <= 0:
            raise ValueError("scale deve ser maior que zero.")

        if np.any(values < 0):
            raise ValueError("Os excessos devem ser não negativos.")

        if np.isclose(shape, 0.0):
            return 1.0 - np.exp(-values / scale)

        support = 1.0 + (shape * values / scale)
        result = np.full(values.shape, np.nan, dtype=np.float64)
        valid = support > 0
        result[valid] = 1.0 - support[valid] ** (-1.0 / shape)
        return result

    @staticmethod
    def survival(values, shape, scale):
        return 1.0 - GeneralizedPareto.cdf(values, shape, scale)

    @staticmethod
    def density(values, shape, scale):
        values = np.asarray(values, dtype=np.float64)

        if scale <= 0:
            raise ValueError("scale deve ser maior que zero.")

        if np.any(values < 0):
            raise ValueError("Os excessos devem ser não negativos.")

        if np.isclose(shape, 0.0):
            return np.exp(-values / scale) / scale

        support = 1.0 + (shape * values / scale)
        result = np.full(values.shape, np.nan, dtype=np.float64)
        valid = support > 0
        result[valid] = (1.0 / scale) * support[valid] ** (-1.0 / shape - 1.0)
        return result

    @staticmethod
    def logLikelihood(values, shape, scale):
        values = np.asarray(values, dtype=np.float64)
        count = len(values)

        if count == 0 or scale <= 0 or np.any(values < 0):
            return -np.inf

        if np.isclose(shape, 0.0):
            return -count * np.log(scale) - np.sum(values) / scale

        support = 1.0 + (shape * values / scale)

        if np.any(support <= 0):
            return -np.inf

        return -count * np.log(scale) - (1.0 + 1.0 / shape) * np.sum(np.log(support))


class TailSelection:
    @staticmethod
    def select(values, quantile):
        values = np.asarray(values, dtype=np.float64)

        if len(values) == 0:
            raise ValueError("A série está vazia.")

        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile deve estar no intervalo (0, 1).")

        threshold = float(np.quantile(values, quantile))
        mask = values > threshold
        peaks = values[mask]
        excesses = peaks - threshold

        return {
            "threshold": threshold,
            "mask": mask,
            "peaks": peaks,
            "excesses": excesses,
        }


class GrimshawGPD:
    @staticmethod
    def functionU(x, values):
        values = np.asarray(values, dtype=np.float64)
        terms = 1.0 + x * values

        if np.any(terms <= 0):
            return np.nan

        return float(np.mean(1.0 / terms))

    @staticmethod
    def functionV(x, values):
        values = np.asarray(values, dtype=np.float64)
        terms = 1.0 + x * values

        if np.any(terms <= 0):
            return np.nan

        return float(1.0 + np.mean(np.log(terms)))

    @staticmethod
    def functionW(x, values):
        functionU = GrimshawGPD.functionU(x, values)
        functionV = GrimshawGPD.functionV(x, values)

        if not np.isfinite(functionU) or not np.isfinite(functionV):
            return np.nan

        return functionU * functionV - 1.0

    @staticmethod
    def objective(x, values):
        scalarX = float(np.asarray(x).reshape(-1)[0])
        functionW = GrimshawGPD.functionW(scalarX, values)

        if not np.isfinite(functionW):
            return 1e100

        return float(functionW * functionW)

    @staticmethod
    def createIntervals(values, epsilon=1e-8):
        values = np.asarray(values, dtype=np.float64)

        if len(values) == 0:
            raise ValueError("Não existem excessos.")

        if np.any(values <= 0):
            raise ValueError("O Grimshaw exige excessos estritamente positivos.")

        maximum = float(np.max(values))
        minimum = float(np.min(values))
        mean = float(np.mean(values))

        negativeLower = -1.0 / maximum + epsilon
        negativeUpper = -epsilon
        intervals = [(negativeLower, negativeUpper)]

        if mean > minimum:
            positiveLower = 2.0 * (mean - minimum) / (mean * minimum)
            positiveUpper = 2.0 * (mean - minimum) / (minimum * minimum)

            if positiveUpper > positiveLower:
                intervals.append((positiveLower, positiveUpper))

        return intervals

    @staticmethod
    def findRoots(values, intervals, starts=10, tolerance=1e-8):
        values = np.asarray(values, dtype=np.float64)
        roots = []

        for lower, upper in intervals:
            initialPoints = np.linspace(lower, upper, max(int(starts), 2))

            for initialPoint in initialPoints:
                result = minimize(
                    fun=GrimshawGPD.objective,
                    x0=np.array([initialPoint], dtype=np.float64),
                    args=(values,),
                    method="L-BFGS-B",
                    bounds=[(lower, upper)],
                )

                if not result.success:
                    continue

                root = float(result.x[0])
                functionW = GrimshawGPD.functionW(root, values)

                if np.isfinite(functionW) and abs(functionW) <= tolerance:
                    roots.append(root)

        uniqueRoots = []

        for root in sorted(roots):
            if np.isclose(root, 0.0, atol=tolerance):
                continue

            if not any(np.isclose(root, current, atol=tolerance, rtol=0.0) for current in uniqueRoots):
                uniqueRoots.append(root)

        return uniqueRoots

    @staticmethod
    def estimate(excesses, starts=10, tolerance=1e-8):
        excesses = np.asarray(excesses, dtype=np.float64)
        excesses = excesses[np.isfinite(excesses) & (excesses > 0)]

        if len(excesses) == 0:
            raise ValueError("Não existem excessos válidos para ajustar a GPD.")

        intervals = GrimshawGPD.createIntervals(excesses)
        roots = GrimshawGPD.findRoots(excesses, intervals, starts, tolerance)
        candidates = []

        exponentialShape = 0.0
        exponentialScale = float(np.mean(excesses))
        exponentialLogLikelihood = GeneralizedPareto.logLikelihood(
            excesses,
            exponentialShape,
            exponentialScale,
        )

        candidates.append(
            {
                "x": 0.0,
                "shape": exponentialShape,
                "scale": exponentialScale,
                "logLikelihood": exponentialLogLikelihood,
                "source": "exponential",
            }
        )

        for root in roots:
            functionV = GrimshawGPD.functionV(root, excesses)
            shape = functionV - 1.0
            scale = shape / root

            if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
                continue

            support = 1.0 + (shape * excesses / scale)

            if np.any(support <= 0):
                continue

            logLikelihood = GeneralizedPareto.logLikelihood(excesses, shape, scale)

            if not np.isfinite(logLikelihood):
                continue

            candidates.append(
                {
                    "x": root,
                    "shape": shape,
                    "scale": scale,
                    "logLikelihood": logLikelihood,
                    "source": "grimshawRoot",
                }
            )

        candidatesTable = (
            pd.DataFrame(candidates)
            .sort_values("logLikelihood", ascending=False)
            .reset_index(drop=True)
        )

        bestCandidate = candidatesTable.iloc[0]

        return {
            "shape": float(bestCandidate["shape"]),
            "scale": float(bestCandidate["scale"]),
            "logLikelihood": float(bestCandidate["logLikelihood"]),
            "x": float(bestCandidate["x"]),
            "source": str(bestCandidate["source"]),
            "roots": roots,
            "intervals": intervals,
            "candidates": candidatesTable,
        }


class POTThreshold:
    @staticmethod
    def calculate(
        initialThreshold,
        shape,
        scale,
        observationCount,
        peakCount,
        risk,
    ):
        if scale <= 0:
            raise ValueError("scale deve ser maior que zero.")

        if observationCount <= 0:
            raise ValueError("observationCount deve ser maior que zero.")

        if peakCount <= 0:
            raise ValueError("peakCount deve ser maior que zero.")

        if not 0.0 < risk < 1.0:
            raise ValueError("risk deve estar no intervalo (0, 1).")

        ratio = risk * observationCount / peakCount

        if ratio <= 0:
            raise ValueError("A razão usada no cálculo do limiar é inválida.")

        if np.isclose(shape, 0.0):
            extremeThreshold = initialThreshold + scale * np.log(peakCount / (risk * observationCount))
        else:
            extremeThreshold = initialThreshold + scale / shape * (ratio ** (-shape) - 1.0)

        if not np.isfinite(extremeThreshold):
            raise RuntimeError("O cálculo do limiar extremo produziu um valor inválido.")

        if extremeThreshold <= initialThreshold:
            raise RuntimeError(
                "O limiar extremo deve ser maior que o limiar intermediário."
            )

        return float(extremeThreshold)


@dataclass
class DSPOTConfig:
    driftDepth: int = 50
    calibrationSize: int = 1024
    initialQuantile: float = 0.98
    risk: float = 0.001
    grimshawStarts: int = 10
    grimshawTolerance: float = 1e-8
    refitEvery: int = 1

    def validate(self):
        if self.driftDepth < 2:
            raise ValueError("driftDepth deve ser maior ou igual a 2.")

        if self.calibrationSize < 20:
            raise ValueError("calibrationSize deve ser maior ou igual a 20.")

        if not 0.5 < self.initialQuantile < 1.0:
            raise ValueError("initialQuantile deve estar no intervalo (0.5, 1).")

        if not 0.0 < self.risk < 1.0:
            raise ValueError("risk deve estar no intervalo (0, 1).")

        if self.grimshawStarts < 2:
            raise ValueError("grimshawStarts deve ser maior ou igual a 2.")

        if self.grimshawTolerance <= 0:
            raise ValueError("grimshawTolerance deve ser maior que zero.")

        if self.refitEvery < 1:
            raise ValueError("refitEvery deve ser maior ou igual a 1.")

    @property
    def warmupSize(self):
        return self.driftDepth + self.calibrationSize


class DSPOT:
    def __init__(self, config=None):
        self.config = config if config is not None else DSPOTConfig()
        self.config.validate()
        self.reset()

    def reset(self):
        self.initialResidualThreshold = np.nan
        self.extremeResidualThreshold = np.nan
        self.shape = np.nan
        self.scale = np.nan
        self.logLikelihood = np.nan
        self.normalHistory = []
        self.excesses = []
        self.adjustmentHistory = []
        self.observationCount = 0
        self.peakCount = 0
        self.peaksSinceFit = 0
        self.initialResult = None
        self.results = None
        self.ready = False

    def currentDrift(self):
        if len(self.normalHistory) == 0:
            return 0.0

        return float(np.mean(self.normalHistory[-self.config.driftDepth:]))

    def addNormalValue(self, value):
        self.normalHistory.append(float(value))

        if len(self.normalHistory) > self.config.driftDepth:
            self.normalHistory.pop(0)

    def fillDriftWindow(self, values):
        values = np.asarray(values, dtype=np.float64)

        if len(values) != self.config.driftDepth:
            raise ValueError("A quantidade de valores não corresponde a driftDepth.")

        if np.any(~np.isfinite(values)):
            raise ValueError("A janela inicial contém valores não finitos.")

        self.normalHistory = [float(value) for value in values]

    def calculateCalibrationResiduals(self, values):
        values = np.asarray(values, dtype=np.float64)

        if len(values) != self.config.calibrationSize:
            raise ValueError("A quantidade de valores não corresponde a calibrationSize.")

        drifts = []
        residuals = []

        for value in values:
            if not np.isfinite(value):
                raise ValueError("A calibração contém valores não finitos.")

            drift = self.currentDrift()
            residual = float(value) - drift
            drifts.append(drift)
            residuals.append(residual)
            self.addNormalValue(value)

        return (
            np.asarray(drifts, dtype=np.float64),
            np.asarray(residuals, dtype=np.float64),
        )

    def fitTail(self, phase, index):
        fit = GrimshawGPD.estimate(
            self.excesses,
            starts=self.config.grimshawStarts,
            tolerance=self.config.grimshawTolerance,
        )

        self.shape = fit["shape"]
        self.scale = fit["scale"]
        self.logLikelihood = fit["logLikelihood"]

        previousThreshold = self.extremeResidualThreshold

        self.extremeResidualThreshold = POTThreshold.calculate(
            initialThreshold=self.initialResidualThreshold,
            shape=self.shape,
            scale=self.scale,
            observationCount=self.observationCount,
            peakCount=self.peakCount,
            risk=self.config.risk,
        )

        self.adjustmentHistory.append(
            {
                "index": index,
                "phase": phase,
                "observationCount": self.observationCount,
                "peakCount": self.peakCount,
                "shape": self.shape,
                "scale": self.scale,
                "logLikelihood": self.logLikelihood,
                "initialResidualThreshold": self.initialResidualThreshold,
                "previousExtremeResidualThreshold": previousThreshold,
                "extremeResidualThreshold": self.extremeResidualThreshold,
                "source": fit["source"],
                "rootCount": len(fit["roots"]),
            }
        )

    def initialize(self, values):
        values = np.asarray(values, dtype=np.float64)

        if len(values) != self.config.warmupSize:
            raise ValueError(
                "A inicialização deve receber driftDepth + calibrationSize valores."
            )

        if np.any(~np.isfinite(values)):
            raise ValueError("A inicialização contém valores não finitos.")

        driftValues = values[:self.config.driftDepth]
        calibrationValues = values[self.config.driftDepth:]

        self.fillDriftWindow(driftValues)

        calibrationDrifts, calibrationResiduals = self.calculateCalibrationResiduals(
            calibrationValues
        )

        tail = TailSelection.select(
            calibrationResiduals,
            self.config.initialQuantile,
        )

        self.initialResidualThreshold = tail["threshold"]
        self.excesses = [float(value) for value in tail["excesses"]]
        self.peakCount = len(self.excesses)
        self.observationCount = self.config.calibrationSize

        if self.peakCount < 3:
            raise ValueError(
                "Foram encontrados poucos picos na calibração. "
                "Aumente calibrationSize ou reduza initialQuantile."
            )

        self.fitTail(
            phase="initialization",
            index=self.config.warmupSize - 1,
        )

        self.ready = True

        self.initialResult = {
            "driftValues": driftValues,
            "calibrationValues": calibrationValues,
            "calibrationDrifts": calibrationDrifts,
            "calibrationResiduals": calibrationResiduals,
            "peakMask": tail["mask"],
            "peaks": tail["peaks"],
            "excesses": tail["excesses"],
        }

        return self.initialResult

    def processValue(self, value, index):
        if not self.ready:
            raise RuntimeError("O DSPOT ainda não foi inicializado.")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("O DSPOT aceita somente valores finitos.")

        drift = self.currentDrift()
        residual = value - drift
        initialThresholdBefore = self.initialResidualThreshold
        extremeThresholdBefore = self.extremeResidualThreshold
        originalInitialThresholdBefore = drift + initialThresholdBefore
        originalExtremeThresholdBefore = drift + extremeThresholdBefore
        classification = "normal"
        updatedTail = False

        if residual > extremeThresholdBefore:
            classification = "anomaly"

        elif residual > initialThresholdBefore:
            classification = "incrementalPeak"
            excess = residual - initialThresholdBefore
            self.excesses.append(float(excess))
            self.peakCount += 1
            self.observationCount += 1
            self.peaksSinceFit += 1
            self.addNormalValue(value)

            if self.peaksSinceFit >= self.config.refitEvery:
                self.fitTail(phase="incremental", index=index)
                self.peaksSinceFit = 0
                updatedTail = True

        else:
            self.observationCount += 1
            self.addNormalValue(value)

        initialThresholdAfter = self.initialResidualThreshold
        extremeThresholdAfter = self.extremeResidualThreshold
        originalInitialThresholdAfter = drift + initialThresholdAfter
        originalExtremeThresholdAfter = drift + extremeThresholdAfter

        return {
            "drift": drift,
            "residual": residual,
            "classification": classification,
            "updatedTail": updatedTail,
            "initialResidualThresholdBefore": initialThresholdBefore,
            "extremeResidualThresholdBefore": extremeThresholdBefore,
            "initialResidualThresholdAfter": initialThresholdAfter,
            "extremeResidualThresholdAfter": extremeThresholdAfter,
            "originalInitialThresholdBefore": originalInitialThresholdBefore,
            "originalExtremeThresholdBefore": originalExtremeThresholdBefore,
            "originalInitialThresholdAfter": originalInitialThresholdAfter,
            "originalExtremeThresholdAfter": originalExtremeThresholdAfter,
            "shape": self.shape,
            "scale": self.scale,
        }

    def run(self, values):
        values = np.asarray(values, dtype=np.float64)

        if len(values) <= self.config.warmupSize:
            raise ValueError(
                "A série precisa conter valores posteriores ao aquecimento completo."
            )

        if np.any(~np.isfinite(values)):
            raise ValueError("A série contém valores não finitos.")

        self.reset()

        initialValues = values[:self.config.warmupSize]
        initialResult = self.initialize(initialValues)
        records = []

        for index in range(self.config.driftDepth):
            value = values[index]

            records.append(
                {
                    "index": index,
                    "value": value,
                    "phase": "driftWindow",
                    "classification": "driftInitialization",
                    "drift": np.nan,
                    "residual": np.nan,
                    "initialResidualThresholdBefore": np.nan,
                    "extremeResidualThresholdBefore": np.nan,
                    "initialResidualThresholdAfter": np.nan,
                    "extremeResidualThresholdAfter": np.nan,
                    "originalInitialThresholdBefore": np.nan,
                    "originalExtremeThresholdBefore": np.nan,
                    "originalInitialThresholdAfter": np.nan,
                    "originalExtremeThresholdAfter": np.nan,
                    "shape": self.shape,
                    "scale": self.scale,
                    "updatedTail": False,
                }
            )

        calibrationStart = self.config.driftDepth

        for calibrationIndex in range(self.config.calibrationSize):
            index = calibrationStart + calibrationIndex
            drift = initialResult["calibrationDrifts"][calibrationIndex]
            residual = initialResult["calibrationResiduals"][calibrationIndex]
            isPeak = initialResult["peakMask"][calibrationIndex]
            classification = "initialPeak" if isPeak else "calibrationNormal"

            records.append(
                {
                    "index": index,
                    "value": values[index],
                    "phase": "calibration",
                    "classification": classification,
                    "drift": drift,
                    "residual": residual,
                    "initialResidualThresholdBefore": self.initialResidualThreshold,
                    "extremeResidualThresholdBefore": self.extremeResidualThreshold,
                    "initialResidualThresholdAfter": self.initialResidualThreshold,
                    "extremeResidualThresholdAfter": self.extremeResidualThreshold,
                    "originalInitialThresholdBefore": drift + self.initialResidualThreshold,
                    "originalExtremeThresholdBefore": drift + self.extremeResidualThreshold,
                    "originalInitialThresholdAfter": drift + self.initialResidualThreshold,
                    "originalExtremeThresholdAfter": drift + self.extremeResidualThreshold,
                    "shape": self.shape,
                    "scale": self.scale,
                    "updatedTail": False,
                }
            )

        for index in range(self.config.warmupSize, len(values)):
            result = self.processValue(values[index], index)

            records.append(
                {
                    "index": index,
                    "value": values[index],
                    "phase": "incremental",
                    **result,
                }
            )

        self.results = pd.DataFrame(records)
        return self.results

    def getThreshold(self):
        if not self.ready:
            return np.nan

        return float(self.currentDrift() + self.extremeResidualThreshold)

    def isReady(self):
        return bool(self.ready)

    def getAdjustmentHistory(self):
        return pd.DataFrame(self.adjustmentHistory)

    def getClassificationSummary(self):
        if self.results is None:
            raise RuntimeError("Execute o DSPOT antes de solicitar o resumo.")

        return (
            self.results["classification"]
            .value_counts()
            .rename_axis("classification")
            .reset_index(name="count")
        )

    def getState(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "driftDepth": self.config.driftDepth,
            "calibrationSize": self.config.calibrationSize,
            "warmupSize": self.config.warmupSize,
            "observationCount": self.observationCount,
            "peakCount": self.peakCount,
            "shape": self.shape,
            "scale": self.scale,
            "initialResidualThreshold": self.initialResidualThreshold,
            "extremeResidualThreshold": self.extremeResidualThreshold,
            "threshold": self.getThreshold(),
            "risk": self.config.risk,
            "initialQuantile": self.config.initialQuantile,
            "refitEvery": self.config.refitEvery,
        }