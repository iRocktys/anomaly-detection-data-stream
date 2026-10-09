from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ProjectDefaults import DEFAULT_DSPOT_PARAMETERS
from src.Anomaly.Thresholds.BaseThreshold import BaseThreshold


class DspotCalibrationError(ValueError):
    pass


@dataclass
class DspotConfig:
    driftDepth: int = DEFAULT_DSPOT_PARAMETERS["driftDepth"]
    calibrationSize: int = DEFAULT_DSPOT_PARAMETERS["calibrationSize"]
    initialQuantile: float = DEFAULT_DSPOT_PARAMETERS["initialQuantile"]
    risk: float = DEFAULT_DSPOT_PARAMETERS["risk"]
    refitEvery: int = DEFAULT_DSPOT_PARAMETERS["refitEvery"]
    optimizationStarts: int = DEFAULT_DSPOT_PARAMETERS["optimizationStarts"]
    tolerance: float = DEFAULT_DSPOT_PARAMETERS["tolerance"]

    @property
    def warmupSize(self):
        # Retorna o total de scores consumidos pela calibração inicial do DSPOT.
        return self.driftDepth + self.calibrationSize

    def validate(self):
        # Valida somente combinações matematicamente permitidas pelo DSPOT.
        if self.driftDepth < 2:
            raise ValueError(
                "driftDepth deve ser maior ou igual a 2."
            )

        if self.calibrationSize < 20:
            raise ValueError(
                "calibrationSize deve ser maior ou igual a 20."
            )

        if not 0.5 < self.initialQuantile < 1.0:
            raise ValueError(
                "initialQuantile deve estar no intervalo (0.5, 1)."
            )

        if not 0.0 < self.risk < 1.0:
            raise ValueError(
                "risk deve estar no intervalo (0, 1)."
            )

        if self.refitEvery < 1:
            raise ValueError(
                "refitEvery deve ser maior ou igual a 1."
            )

        if self.optimizationStarts < 2:
            raise ValueError(
                "optimizationStarts deve ser maior ou igual a 2."
            )

        if self.tolerance <= 0:
            raise ValueError(
                "tolerance deve ser maior que zero."
            )


class DspotThreshold(BaseThreshold):
    def __init__(self, config=None):
        # Inicializa o DSPOT com configuração validada e estado vazio.
        self.config = (
            config
            if config is not None
            else DspotConfig()
        )
        self.config.validate()
        self.reset()

    def initialize(self, scores):
        # Ajusta drift, cauda inicial e GPD usando exatamente a janela de calibração configurada.
        values = np.asarray(
            scores,
            dtype=np.float64,
        )

        if len(values) != self.config.warmupSize:
            raise ValueError(
                "O DSPOT deve receber exatamente "
                f"{self.config.warmupSize} scores no aquecimento."
            )

        if np.any(
            ~np.isfinite(
                values
            )
        ):
            raise DspotCalibrationError(
                "O aquecimento do DSPOT contém scores não finitos."
            )

        driftValues = values[
            :self.config.driftDepth
        ]

        calibrationValues = values[
            self.config.driftDepth:
        ]

        self.normalHistory = [
            float(value)
            for value in driftValues
        ]

        residuals = []

        for value in calibrationValues:
            drift = self.calculateDrift()
            residuals.append(
                float(
                    value
                    - drift
                )
            )
            self.addNormalValue(
                value
            )

        residuals = np.asarray(
            residuals,
            dtype=np.float64,
        )

        self.initialResidualThreshold = float(
            np.quantile(
                residuals,
                self.config.initialQuantile,
            )
        )

        self.excesses = [
            float(
                value
                - self.initialResidualThreshold
            )
            for value in residuals
            if value
            > self.initialResidualThreshold
        ]

        self.observationCount = len(
            residuals
        )

        self.peakCount = len(
            self.excesses
        )

        if self.peakCount < 3:
            raise DspotCalibrationError(
                "O DSPOT encontrou menos de três picos durante a calibração."
            )

        try:
            self.fitTail()
        except DspotCalibrationError:
            raise
        except Exception as error:
            raise DspotCalibrationError(
                f"Falha ao ajustar a cauda inicial do DSPOT: {error}"
            ) from error

        self.ready = True

    def getThreshold(self):
        # Retorna o limiar extremo atual somado ao drift estimado.
        if not self.ready:
            return np.nan

        return float(
            self.calculateDrift()
            + self.extremeResidualThreshold
        )

    def update(self, score, index=None):
        # Atualiza drift e cauda incrementalmente após a calibração inicial.
        if not self.ready:
            raise RuntimeError(
                "O DSPOT ainda não foi inicializado."
            )

        score = float(
            score
        )

        if not np.isfinite(
            score
        ):
            raise ValueError(
                "O DSPOT aceita somente scores finitos."
            )

        drift = self.calculateDrift()
        residual = (
            score
            - drift
        )
        classification = "normal"
        updatedTail = False

        if residual > self.extremeResidualThreshold:
            classification = "anomaly"

        elif residual > self.initialResidualThreshold:
            classification = "peak"
            excess = (
                residual
                - self.initialResidualThreshold
            )

            self.excesses.append(
                float(
                    excess
                )
            )

            self.peakCount += 1
            self.observationCount += 1
            self.peaksSinceFit += 1

            self.addNormalValue(
                score
            )

            if (
                self.peaksSinceFit
                >= self.config.refitEvery
            ):
                try:
                    self.fitTail()
                except DspotCalibrationError:
                    pass

                self.peaksSinceFit = 0
                updatedTail = True

        else:
            self.observationCount += 1

            self.addNormalValue(
                score
            )

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
        # Mantém compatibilidade com chamadas que usam a interface processValue.
        return self.update(
            score,
            index,
        )

    def reset(self):
        # Limpa completamente o estado incremental do DSPOT.
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
        # Informa se a calibração inicial terminou com sucesso.
        return bool(
            self.ready
        )

    def getState(self):
        # Retorna um resumo do estado interno atual do DSPOT.
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
        # Calcula o drift pela média dos últimos valores normais armazenados.
        if not self.normalHistory:
            return 0.0

        return float(
            np.mean(
                self.normalHistory[
                    -self.config.driftDepth:
                ]
            )
        )

    def addNormalValue(self, value):
        # Atualiza a memória limitada usada para estimar o drift.
        self.normalHistory.append(
            float(
                value
            )
        )

        if (
            len(
                self.normalHistory
            )
            > self.config.driftDepth
        ):
            self.normalHistory.pop(
                0
            )

    def fitTail(self):
        # Ajusta a GPD sobre os excessos atuais e recalcula o limiar extremo.
        excesses = np.asarray(
            self.excesses,
            dtype=np.float64,
        )

        excesses = excesses[
            np.isfinite(
                excesses
            )
            & (
                excesses
                > 0
            )
        ]

        if len(excesses) < 3:
            raise DspotCalibrationError(
                "Não existem excessos suficientes para ajustar a distribuição GPD."
            )

        candidates = []

        exponentialScale = float(
            np.mean(
                excesses
            )
        )

        if (
            not np.isfinite(
                exponentialScale
            )
            or exponentialScale <= 0
        ):
            raise DspotCalibrationError(
                "A escala inicial da GPD é inválida."
            )

        exponentialLikelihood = self.logLikelihood(
            excesses,
            0.0,
            exponentialScale,
        )

        candidates.append(
            (
                exponentialLikelihood,
                0.0,
                exponentialScale,
            )
        )

        maximum = float(
            np.max(
                excesses
            )
        )

        minimum = float(
            np.min(
                excesses
            )
        )

        mean = float(
            np.mean(
                excesses
            )
        )

        intervals = [
            (
                -1.0 / maximum
                + self.config.tolerance,
                -self.config.tolerance,
            )
        ]

        if mean > minimum:
            positiveLower = (
                2.0
                * (
                    mean
                    - minimum
                )
                / (
                    mean
                    * minimum
                )
            )

            positiveUpper = (
                2.0
                * (
                    mean
                    - minimum
                )
                / (
                    minimum
                    * minimum
                )
            )

            if (
                positiveUpper
                > positiveLower
            ):
                intervals.append(
                    (
                        positiveLower,
                        positiveUpper,
                    )
                )

        for lower, upper in intervals:
            if (
                not np.isfinite(
                    lower
                )
                or not np.isfinite(
                    upper
                )
                or upper <= lower
            ):
                continue

            initialValues = np.linspace(
                lower,
                upper,
                self.config.optimizationStarts,
            )

            for initialValue in initialValues:
                result = minimize(
                    self.objective,
                    np.array(
                        [
                            initialValue
                        ]
                    ),
                    args=(
                        excesses,
                    ),
                    method="L-BFGS-B",
                    bounds=[
                        (
                            lower,
                            upper,
                        )
                    ],
                )

                if not result.success:
                    continue

                root = float(
                    result.x[
                        0
                    ]
                )

                functionValue = self.grimshawFunction(
                    root,
                    excesses,
                )

                if (
                    not np.isfinite(
                        functionValue
                    )
                    or abs(
                        functionValue
                    )
                    > self.config.tolerance
                ):
                    continue

                functionV = (
                    1.0
                    + np.mean(
                        np.log(
                            1.0
                            + root
                            * excesses
                        )
                    )
                )

                shape = float(
                    functionV
                    - 1.0
                )

                scale = float(
                    shape
                    / root
                )

                if (
                    not np.isfinite(
                        shape
                    )
                    or not np.isfinite(
                        scale
                    )
                    or scale <= 0
                ):
                    continue

                likelihood = self.logLikelihood(
                    excesses,
                    shape,
                    scale,
                )

                if np.isfinite(
                    likelihood
                ):
                    candidates.append(
                        (
                            likelihood,
                            shape,
                            scale,
                        )
                    )

        if not candidates:
            raise DspotCalibrationError(
                "Nenhum ajuste válido da GPD foi encontrado."
            )

        (
            bestLikelihood,
            self.shape,
            self.scale,
        ) = max(
            candidates,
            key=lambda candidate: candidate[
                0
            ],
        )

        ratio = (
            self.config.risk
            * self.observationCount
            / self.peakCount
        )

        if ratio <= 0:
            raise DspotCalibrationError(
                "A razão usada no limiar extremo do DSPOT é inválida."
            )

        if np.isclose(
            self.shape,
            0.0,
        ):
            self.extremeResidualThreshold = (
                self.initialResidualThreshold
                + self.scale
                * np.log(
                    self.peakCount
                    / (
                        self.config.risk
                        * self.observationCount
                    )
                )
            )

        else:
            self.extremeResidualThreshold = (
                self.initialResidualThreshold
                + self.scale
                * (
                    ratio
                    ** (
                        -self.shape
                    )
                    - 1.0
                )
                / self.shape
            )

        if (
            not np.isfinite(
                bestLikelihood
            )
            or not np.isfinite(
                self.extremeResidualThreshold
            )
        ):
            raise DspotCalibrationError(
                "O ajuste do DSPOT produziu valores inválidos."
            )

    def grimshawFunction(self, value, excesses):
        # Calcula a equação de Grimshaw usada no ajuste dos parâmetros da GPD.
        terms = (
            1.0
            + value
            * excesses
        )

        if np.any(
            terms <= 0
        ):
            return np.nan

        functionU = np.mean(
            1.0
            / terms
        )

        functionV = (
            1.0
            + np.mean(
                np.log(
                    terms
                )
            )
        )

        return float(
            functionU
            * functionV
            - 1.0
        )

    def objective(self, value, excesses):
        # Converte a equação de Grimshaw em objetivo escalar para o otimizador numérico.
        scalarValue = float(
            np.asarray(
                value
            ).reshape(
                -1
            )[
                0
            ]
        )

        functionValue = self.grimshawFunction(
            scalarValue,
            excesses,
        )

        if not np.isfinite(
            functionValue
        ):
            return 1e100

        return float(
            functionValue
            * functionValue
        )

    def logLikelihood(self, excesses, shape, scale):
        # Calcula a log-verossimilhança da GPD para selecionar o melhor ajuste de cauda.
        if scale <= 0:
            return -np.inf

        if np.isclose(
            shape,
            0.0,
        ):
            return float(
                -len(
                    excesses
                )
                * np.log(
                    scale
                )
                - np.sum(
                    excesses
                )
                / scale
            )

        support = (
            1.0
            + shape
            * excesses
            / scale
        )

        if np.any(
            support <= 0
        ):
            return -np.inf

        return float(
            -len(
                excesses
            )
            * np.log(
                scale
            )
            - (
                1.0
                + 1.0
                / shape
            )
            * np.sum(
                np.log(
                    support
                )
            )
        )


DSPOT = DspotThreshold
DSPOTConfig = DspotConfig
