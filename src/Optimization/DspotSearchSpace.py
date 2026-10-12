from dataclasses import dataclass
import math
from numbers import Integral

import ProjectDefaults as defaults


@dataclass(frozen=True)
class DspotSearchSpaceConfig:
    scoreModes: tuple[str, ...] = (
        "raw",
        "movingAverage",
    )
    movingAverageMinimum: int = 2
    movingAverageMaximum: int = 200
    calibrationWindow: int = defaults.DEFAULT_DSPOT_CALIBRATION_WINDOW
    driftDepthMinimum: int = 20
    driftDepthMaximum: int = 200
    driftDepthStep: int = 5
    initialQuantileMinimum: float = 0.90
    initialQuantileMaximum: float = 0.98
    riskMinimum: float = 1e-5
    riskMaximum: float = 1e-2
    refitEveryMinimum: int = 1
    refitEveryMaximum: int = 25
    optimizationStartsMinimum: int = 5
    optimizationStartsMaximum: int = 20
    toleranceMinimum: float = 1e-10
    toleranceMaximum: float = 1e-6

    def __post_init__(self):
        # Valida o espaço DSPOT mantendo fixa a janela total de 500 scores.
        integerNames = (
            "movingAverageMinimum",
            "movingAverageMaximum",
            "calibrationWindow",
            "driftDepthMinimum",
            "driftDepthMaximum",
            "driftDepthStep",
            "refitEveryMinimum",
            "refitEveryMaximum",
            "optimizationStartsMinimum",
            "optimizationStartsMaximum",
        )

        if any(
            not isinstance(
                getattr(
                    self,
                    name,
                ),
                Integral,
            )
            or isinstance(
                getattr(
                    self,
                    name,
                ),
                bool,
            )
            for name in integerNames
        ):
            raise ValueError(
                "Os limites inteiros do DSPOT devem usar valores inteiros."
            )

        if (
            self.calibrationWindow
            != defaults.DEFAULT_DSPOT_CALIBRATION_WINDOW
        ):
            raise ValueError(
                "A janela total de calibração do DSPOT deve permanecer em 500."
            )

        if (
            self.driftDepthMinimum
            < 2
            or self.driftDepthMaximum
            >= self.calibrationWindow
            - 20
            or self.driftDepthMaximum
            < self.driftDepthMinimum
        ):
            raise ValueError(
                "Intervalo de driftDepth incompatível com a janela DSPOT de 500."
            )

        if (
            self.driftDepthMaximum
            - self.driftDepthMinimum
        ) % self.driftDepthStep:
            raise ValueError(
                "O intervalo de driftDepth deve ser divisível pelo passo."
            )

        if not set(
            self.scoreModes
        ) <= {
            "raw",
            "movingAverage",
        }:
            raise ValueError(
                "Fonte de score inválida no espaço de busca."
            )

        if not self.scoreModes:
            raise ValueError(
                "O espaço deve possuir ao menos uma fonte de score."
            )

        if (
            self.movingAverageMinimum
            < 1
            or self.movingAverageMaximum
            < self.movingAverageMinimum
        ):
            raise ValueError(
                "Intervalo de média móvel inválido."
            )

        if not (
            0.5
            < self.initialQuantileMinimum
            <= self.initialQuantileMaximum
            < 1.0
        ):
            raise ValueError(
                "Intervalo de initialQuantile inválido."
            )

        if not (
            0.0
            < self.riskMinimum
            <= self.riskMaximum
            < 1.0
        ):
            raise ValueError(
                "Intervalo de risk inválido."
            )

        if (
            self.refitEveryMinimum
            < 1
            or self.refitEveryMaximum
            < self.refitEveryMinimum
        ):
            raise ValueError(
                "Intervalo de refitEvery inválido."
            )

        if (
            self.optimizationStartsMinimum
            < 2
            or self.optimizationStartsMaximum
            < self.optimizationStartsMinimum
        ):
            raise ValueError(
                "Intervalo de optimizationStarts inválido."
            )

        if not (
            math.isfinite(
                self.toleranceMinimum
            )
            and math.isfinite(
                self.toleranceMaximum
            )
            and 0.0
            < self.toleranceMinimum
            <= self.toleranceMaximum
        ):
            raise ValueError(
                "Intervalo de tolerance inválido."
            )


class DspotSearchSpace:
    def __init__(self, config=None):
        # Inicializa o espaço DSPOT preservando o protocolo fixo de 500 scores de calibração.
        self.config = (
            config
            or DspotSearchSpaceConfig()
        )

    def suggest(self, trial, imputerName=None):
        # Sugere score, drift, cauda e parâmetros numéricos do DSPOT sem variar o warm-up total.
        from src.Optimization.OptimizationConfig import TrialConfiguration

        if imputerName is None:
            imputerName = "incrementalMean"

        scoreMode = trial.suggest_categorical(
            "scoreMode",
            list(
                self.config.scoreModes
            ),
        )

        if scoreMode == "raw":
            thresholdScoreSource = "raw"
            scoreWindowSizes = ()

        else:
            movingAverageWindow = trial.suggest_int(
                "movingAverageWindow",
                self.config.movingAverageMinimum,
                self.config.movingAverageMaximum,
            )

            thresholdScoreSource = (
                f"ma{movingAverageWindow}"
            )

            scoreWindowSizes = (
                movingAverageWindow,
            )

        driftDepth = trial.suggest_int(
            "driftDepth",
            self.config.driftDepthMinimum,
            self.config.driftDepthMaximum,
            step=self.config.driftDepthStep,
        )

        calibrationSize = (
            self.config.calibrationWindow
            - driftDepth
        )

        return TrialConfiguration(
            imputerName=imputerName,
            thresholdScoreSource=thresholdScoreSource,
            scoreWindowSizes=scoreWindowSizes,
            driftDepth=driftDepth,
            calibrationSize=calibrationSize,
            initialQuantile=trial.suggest_float(
                "initialQuantile",
                self.config.initialQuantileMinimum,
                self.config.initialQuantileMaximum,
            ),
            risk=trial.suggest_float(
                "risk",
                self.config.riskMinimum,
                self.config.riskMaximum,
                log=True,
            ),
            refitEvery=trial.suggest_int(
                "refitEvery",
                self.config.refitEveryMinimum,
                self.config.refitEveryMaximum,
            ),
            optimizationStarts=trial.suggest_int(
                "optimizationStarts",
                self.config.optimizationStartsMinimum,
                self.config.optimizationStartsMaximum,
            ),
            tolerance=trial.suggest_float(
                "tolerance",
                self.config.toleranceMinimum,
                self.config.toleranceMaximum,
                log=True,
            ),
        )
