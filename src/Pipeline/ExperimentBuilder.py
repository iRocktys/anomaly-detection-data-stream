from src.Anomaly.Models import getAnomalyModels
from src.Anomaly.Thresholds.ThresholdFactory import ThresholdFactory
from src.Data.OnlineNormalizers import OnlineNormalizers
from src.Features.FeatureExtractors import FeatureExtractors
from src.Features.FeatureSelectors import FeatureSelectors
from src.Features.FeatureSmoothers import FeatureSmoothers
from src.Pipeline.DecisionComponents import DecisionComponents
from src.Scores.ScoreSmoothers import ScoreSmoothers
from src.Training.TrainingStrategies import TrainingStrategies


class ExperimentBuilder:
    def buildComponents(self, config, schema, modelCode, normalizerConfig):
        modelName, model = self.createModel(config, schema, modelCode)
        return {
            "modelName": modelName,
            "model": model,
            "featureSelector": FeatureSelectors.createSelector(config.featureSelector),
            "featureExtractor": FeatureExtractors.createExtractor(config.featureExtractor),
            "normalizer": OnlineNormalizers.createNormalizer(
                normalizerConfig.name,
                **normalizerConfig.parameters,
            ),
            "featureSmoother": FeatureSmoothers.createSmoother(config.featureSmoother),
            "scoreSmoother": ScoreSmoothers.createSmoother(config.scoreSmoother),
            "thresholdStrategy": ThresholdFactory.createThreshold(config.thresholdStrategy),
            "decisionStrategy": DecisionComponents.createStrategy(config.decisionStrategy),
            "trainingStrategy": TrainingStrategies.createStrategy(config.trainingStrategy),
        }

    def createModel(self, config, schema, modelCode):
        modelParams = config.modelParameters.get(modelCode, {})
        buildParams = {
            "schema": schema,
            "selectedModels": [modelCode],
            "runSeed": config.runSeed,
        }
        if modelCode == "HST":
            buildParams["hstParams"] = modelParams
        elif modelCode == "AE":
            buildParams["aeParams"] = modelParams
        elif modelCode == "AIF":
            buildParams["aifParams"] = modelParams

        models = getAnomalyModels(**buildParams)
        if not models:
            raise ValueError(f"Nenhum modelo foi criado para: {modelCode}")
        modelName = next(iter(models))
        return modelName, models[modelName]
