from src.Anomaly.Models import ModelRegistry
from src.Pipeline.ComponentRegistry import ComponentRegistry


class ExperimentBuilder:
    def buildScoreComponents(self, plan, schema, modelConfig, normalizerConfig, runSeed):
        modelName, model, resolvedParameters = ModelRegistry.create(
            schema=schema,
            modelConfig=modelConfig,
            runSeed=runSeed,
        )
        return {
            "modelName": modelName,
            "model": model,
            "modelParameters": resolvedParameters,
            "featureExtractor": ComponentRegistry.create("featureExtractor", plan.featureExtractor),
            "normalizer": ComponentRegistry.create("normalizer", normalizerConfig),
            "featureSmoother": ComponentRegistry.create("featureSmoother", plan.featureSmoother),
            "trainingStrategy": ComponentRegistry.create("training", plan.trainingStrategy),
        }

    def buildEvaluationComponents(self, evaluationConfig):
        return {
            "scoreSmoother": ComponentRegistry.create("scoreSmoother", evaluationConfig.scoreSmoother),
            "threshold": ComponentRegistry.create("threshold", evaluationConfig.threshold),
            "decision": ComponentRegistry.create("decision", evaluationConfig.decisionStrategy),
        }
