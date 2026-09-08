"""Limites experimentais do DSPOT e da fonte de score.

imputerNames é mantido para compatibilidade com o uso direto deste espaço.
O otimizador padrão seleciona a imputação por OptimizationConfig.imputerNames.
"""
from dataclasses import dataclass
from numbers import Integral
import math
from ProjectDefaults import DEFAULT_DSPOT_PARAMETERS, DEFAULT_IMPUTER_NAMES


@dataclass(frozen=True)
class DspotSearchSpaceConfig:
    imputerNames: tuple[str, ...] = DEFAULT_IMPUTER_NAMES
    scoreModes: tuple[str, ...] = ("raw", "movingAverage")
    movingAverageMinimum: int = 2
    movingAverageMaximum: int = 200
    riskMinimum: float = 1e-5
    riskMaximum: float = 1e-2
    initialQuantileMinimum: float = 0.90
    initialQuantileMaximum: float = 0.99
    calibrationWindowMinimum: int = 500
    calibrationWindowMaximum: int = 1500
    calibrationWindowStep: int = 50
    driftDepthMinimum: int = 20
    driftDepthMaximum: int = 200
    driftDepthStep: int = 5
    refitEveryMinimum: int = 1
    refitEveryMaximum: int = 25
    optimizationStarts: int = DEFAULT_DSPOT_PARAMETERS["optimizationStarts"]
    tolerance: float = DEFAULT_DSPOT_PARAMETERS["tolerance"]

    def __post_init__(self):
        integerNames = (
            "movingAverageMinimum", "movingAverageMaximum", "calibrationWindowMinimum",
            "calibrationWindowMaximum", "calibrationWindowStep", "driftDepthMinimum",
            "driftDepthMaximum", "driftDepthStep", "refitEveryMinimum", "refitEveryMaximum",
            "optimizationStarts",
        )
        if any(not isinstance(getattr(self, name), Integral) or isinstance(getattr(self, name), bool)
               for name in integerNames):
            raise ValueError("Os limites e passos das janelas devem ser inteiros.")
        if self.driftDepthMinimum < 2:
            raise ValueError("driftDepthMinimum deve ser >= 2.")
        if not math.isfinite(self.tolerance):
            raise ValueError("tolerance deve ser finita.")
        if not set(self.imputerNames) <= {"zero", "incrementalMean"}:
            raise ValueError("Imputador inválido no espaço de busca.")
        if not set(self.scoreModes) <= {"raw", "movingAverage"}:
            raise ValueError("Fonte de score inválida no espaço de busca.")
        if not self.imputerNames or not self.scoreModes:
            raise ValueError(
                "O espaço de busca deve possuir imputadores e fontes de score."
            )
        if self.movingAverageMinimum < 1 or (
            self.movingAverageMaximum < self.movingAverageMinimum
        ):
            raise ValueError("Intervalo de média móvel inválido.")
        if not (
            0.0 < self.riskMinimum <= self.riskMaximum < 1.0
        ):
            raise ValueError("Intervalo de risk inválido.")
        if not (
            0.5
            < self.initialQuantileMinimum
            <= self.initialQuantileMaximum
            < 1.0
        ):
            raise ValueError("Intervalo de initialQuantile inválido.")
        if self.calibrationWindowStep < 1 or self.driftDepthStep < 1:
            raise ValueError("Os passos das janelas devem ser positivos.")
        if (
            self.calibrationWindowMaximum
            < self.calibrationWindowMinimum
            or self.driftDepthMaximum < self.driftDepthMinimum
        ):
            raise ValueError("Intervalos de calibração ou drift inválidos.")
        if self.calibrationWindowMinimum - self.driftDepthMaximum < 20:
            raise ValueError(
                "A menor janela de calibração deve preservar ao menos 20 "
                "valores após o driftDepth máximo."
            )
        for prefix in ("calibrationWindow", "driftDepth"):
            low, high, step = (getattr(self, prefix + suffix) for suffix in ("Minimum", "Maximum", "Step"))
            if (high - low) % step:
                raise ValueError(f"O intervalo de {prefix} deve ser divisível pelo passo.")
        if self.refitEveryMinimum < 1 or (
            self.refitEveryMaximum < self.refitEveryMinimum
        ):
            raise ValueError("Intervalo de refitEvery inválido.")
        if self.optimizationStarts < 2 or self.tolerance <= 0:
            raise ValueError(
                "optimizationStarts e tolerance possuem valores inválidos."
            )


class DspotSearchSpace:
    def __init__(self, config=None):
        self.config = config or DspotSearchSpaceConfig()

    def suggest(self, trial, imputerName=None):
        # Import local evita ciclo entre os contratos e seus reexports.
        from src.Optimization.OptimizationConfig import TrialConfiguration

        if imputerName is None:
            imputerName = trial.suggest_categorical("imputer", list(self.config.imputerNames))
        scoreMode = trial.suggest_categorical(
            "scoreMode",
            list(self.config.scoreModes),
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
            thresholdScoreSource = f"ma{movingAverageWindow}"
            scoreWindowSizes = (movingAverageWindow,)

        calibrationWindow = trial.suggest_int(
            "calibrationWindow",
            self.config.calibrationWindowMinimum,
            self.config.calibrationWindowMaximum,
            step=self.config.calibrationWindowStep,
        )
        driftDepth = trial.suggest_int(
            "driftDepth",
            self.config.driftDepthMinimum,
            self.config.driftDepthMaximum,
            step=self.config.driftDepthStep,
        )

        return TrialConfiguration(
            imputerName=imputerName,
            thresholdScoreSource=thresholdScoreSource,
            scoreWindowSizes=scoreWindowSizes,
            driftDepth=driftDepth,
            calibrationSize=calibrationWindow - driftDepth,
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
            optimizationStarts=self.config.optimizationStarts,
            tolerance=self.config.tolerance,
        )
