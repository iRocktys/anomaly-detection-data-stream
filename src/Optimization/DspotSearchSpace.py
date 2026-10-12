from dataclasses import dataclass

import ProjectDefaults as defaults


@dataclass(frozen=True)
class DspotSearchSpaceConfig:
    scoreModes: tuple[str, ...] = ("raw", "movingAverage")
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
        # Valida o espaço DSPOT e preserva as 500 amostras destinadas à calibração do threshold.
        if int(self.calibrationWindow) != defaults.DEFAULT_DSPOT_CALIBRATION_WINDOW:
            raise ValueError("The DSPOT calibration window must remain at 500 samples.")
        if self.driftDepthMinimum < 2 or self.driftDepthMaximum >= self.calibrationWindow - 20:
            raise ValueError("The driftDepth range is incompatible with DSPOT calibration.")
        if self.driftDepthMaximum < self.driftDepthMinimum or self.driftDepthStep < 1:
            raise ValueError("The driftDepth range is invalid.")
        if not self.scoreModes or not set(self.scoreModes) <= {"raw", "movingAverage"}:
            raise ValueError("scoreModes must contain only raw and movingAverage.")


class DspotSearchSpace:
    def __init__(self, config=None):
        # Inicializa o espaço de busca dos parâmetros de score e cauda do DSPOT.
        self.config = config or DspotSearchSpaceConfig()

    def suggest(self, trial):
        # Sugere uma configuração DSPOT mantendo fixa a quantidade total de scores de calibração.
        score_mode = trial.suggest_categorical("scoreMode", list(self.config.scoreModes))
        if score_mode == "movingAverage":
            window = trial.suggest_int("movingAverageWindow", self.config.movingAverageMinimum, self.config.movingAverageMaximum)
            score_source = f"ma{window}"
            score_windows = (window,)
        else:
            score_source = "raw"
            score_windows = ()
        drift_depth = trial.suggest_int(
            "driftDepth",
            self.config.driftDepthMinimum,
            self.config.driftDepthMaximum,
            step=self.config.driftDepthStep,
        )
        return {
            "name": "dspot",
            "scoreSource": score_source,
            "scoreWindowSizes": score_windows,
            "parameters": {
                "driftDepth": int(drift_depth),
                "calibrationSize": int(self.config.calibrationWindow - drift_depth),
                "initialQuantile": trial.suggest_float("initialQuantile", self.config.initialQuantileMinimum, self.config.initialQuantileMaximum),
                "risk": trial.suggest_float("risk", self.config.riskMinimum, self.config.riskMaximum, log=True),
                "refitEvery": trial.suggest_int("refitEvery", self.config.refitEveryMinimum, self.config.refitEveryMaximum),
                "optimizationStarts": trial.suggest_int("optimizationStarts", self.config.optimizationStartsMinimum, self.config.optimizationStartsMaximum),
                "tolerance": trial.suggest_float("tolerance", self.config.toleranceMinimum, self.config.toleranceMaximum, log=True),
            },
        }
