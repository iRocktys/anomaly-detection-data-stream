from src.Optimization.OptimizationConfig import (
    DspotSearchSpaceConfig,
    TrialConfiguration,
)


class DspotSearchSpace:
    def __init__(self, config=None):
        self.config = config or DspotSearchSpaceConfig()

    def suggest(self, trial):
        imputerName = trial.suggest_categorical(
            "imputer",
            list(self.config.imputerNames),
        )
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
