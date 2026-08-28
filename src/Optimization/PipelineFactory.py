from src.Anomaly.Thresholds.Incremental.DspotThreshold import (
    DspotConfig,
    DspotThreshold,
)
from src.Data.OnlineImputers import (
    IncrementalMeanImputer,
    ZeroOnlineImputer,
)
from src.Data.OnlineNormalizers import IncrementalZScoreNormalizer
from src.Optimization.OptimizationConfig import ModelProfile
from src.Pipeline.TrainingPipeline import TrainingPipeline
from src.Training.TrainAllStrategy import TrainAllStrategy


class PipelineFactory:
    imputerFactories = {
        "zero": ZeroOnlineImputer,
        "incrementalMean": IncrementalMeanImputer,
    }

    def __init__(self, optimizationConfig, modelProfile=None):
        self.optimizationConfig = optimizationConfig
        self.modelProfile = modelProfile or ModelProfile()

    def create(
        self,
        scenario,
        trialConfiguration,
        resultManager,
    ):
        try:
            imputerFactory = self.imputerFactories[
                trialConfiguration.imputerName
            ]
        except KeyError as error:
            raise ValueError(
                "Imputador desconhecido no espaço de busca: "
                f"{trialConfiguration.imputerName}."
            ) from error

        dspotConfig = DspotConfig(
            driftDepth=trialConfiguration.driftDepth,
            calibrationSize=trialConfiguration.calibrationSize,
            initialQuantile=trialConfiguration.initialQuantile,
            risk=trialConfiguration.risk,
            refitEvery=trialConfiguration.refitEvery,
            optimizationStarts=trialConfiguration.optimizationStarts,
            tolerance=trialConfiguration.tolerance,
        )
        return TrainingPipeline(
            stream=scenario.stream,
            datasetName=scenario.datasetName,
            modelCode=self.modelProfile.code,
            threshold=DspotThreshold(dspotConfig),
            labelNames=scenario.labelNames,
            modelParameters=dict(self.modelProfile.parameters),
            imputer=imputerFactory(),
            normalizer=IncrementalZScoreNormalizer(),
            trainingStrategy=TrainAllStrategy(),
            scoreWindowSizes=trialConfiguration.scoreWindowSizes,
            metricsWindowSize=self.optimizationConfig.metricsWindowSize,
            outputPath=self.optimizationConfig.outputRoot,
            normalClassIndex=0,
            seed=self.optimizationConfig.seed,
            generatePlots=False,
            initialWarmupSize=self.optimizationConfig.initialWarmupSize,
            thresholdScoreSource=(
                trialConfiguration.thresholdScoreSource
            ),
            resultManager=resultManager,
        )
