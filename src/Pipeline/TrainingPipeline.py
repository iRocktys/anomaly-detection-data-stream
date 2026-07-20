import numpy as np

from src.Anomaly.Models import ModelRegistry
from src.Data.OnlineNormalizers import NoOnlineNormalizer
from src.Metrics.IncrementalMetrics import IncrementalMetrics
from src.Pipeline.ResultManager import ResultManager
from src.Scores.ScoreMovingAverages import ScoreMovingAverages
from src.Training.TrainAllStrategy import TrainAllStrategy


class TrainingPipeline:
    def __init__(self, stream, datasetName, modelCode, threshold, labelNames, modelParameters=None, normalizer=None, trainingStrategy=None, scoreWindowSizes=None, metricsWindowSize=1000, outputPath="output", normalClassIndex=0, seed=1, generatePlots=True):
        self.stream = stream
        self.datasetName = str(datasetName)
        self.modelCode = ModelRegistry.normalizeCode(modelCode)
        self.threshold = threshold
        self.labelNames = list(labelNames)
        self.modelParameters = dict(modelParameters or {})
        self.normalizer = normalizer if normalizer is not None else NoOnlineNormalizer()
        self.trainingStrategy = trainingStrategy if trainingStrategy is not None else TrainAllStrategy()
        self.scoreMovingAverages = ScoreMovingAverages(scoreWindowSizes)
        self.metrics = IncrementalMetrics()
        self.metricsWindowSize = max(1, int(metricsWindowSize))
        self.resultManager = ResultManager(outputPath)
        self.normalClassIndex = int(normalClassIndex)
        self.seed = int(seed)
        self.generatePlots = bool(generatePlots)
        self.thresholdWarmupScores = []
        self.modelName = None
        self.model = None

    def run(self):
        self.resetComponents()
        self.stream.restart()
        self.createModel()

        rows = []
        instanceIndex = 0

        while self.stream.has_more_instances():
            if instanceIndex >= len(self.labelNames):
                raise RuntimeError("Não existe rótulo original para a instância atual da stream.")

            rawInstance = self.stream.next_instance()
            rows.append(self.processInstance(rawInstance, instanceIndex))
            instanceIndex += 1

        if instanceIndex != len(self.labelNames):
            raise RuntimeError("A quantidade de rótulos originais é diferente da quantidade de instâncias da stream.")

        movingAverageColumns = [f"scoreMa{windowSize}" for windowSize in self.scoreMovingAverages.windowSizes]

        return self.resultManager.save(
            rows,
            self.datasetName,
            self.modelCode,
            self.metricsWindowSize,
            movingAverageColumns,
            self.generatePlots,
        )

    def createModel(self):
        schema = self.getStreamSchema()
        definition = ModelRegistry.definitions[self.modelCode]
        parameters = dict(definition.defaults)
        parameters.update(self.modelParameters)
        parameters["schema"] = schema
        parameters.setdefault(definition.seedParameter, self.seed)

        ModelRegistry.validateParameters(self.modelCode, parameters)

        modelClass = definition.loader()
        self.model = modelClass(**parameters)
        self.modelName = definition.displayName

    def processInstance(self, rawInstance, instanceIndex):
        rawValues = np.asarray(rawInstance.x, dtype=np.float64)
        normalizedValues = self.normalizer.transform(rawValues)
        modelInstance = self.createModelInstance(rawInstance, normalizedValues)

        rawScore = float(self.model.score_instance(modelInstance))
        score = rawScore
        scoreAverages = self.scoreMovingAverages.calculate(score)

        thresholdReady = bool(self.threshold.isReady())
        thresholdValue = float(self.threshold.getThreshold()) if thresholdReady else np.nan
        predictedLabel = int(score > thresholdValue) if thresholdReady else 0
        isWarmup = not thresholdReady

        trueLabel = int(rawInstance.y_index)
        isAttack = int(trueLabel != self.normalClassIndex)
        labelName = str(self.labelNames[instanceIndex])

        shouldTrain = self.trainingStrategy.shouldTrain(
            prediction=predictedLabel,
            thresholdReady=thresholdReady,
            isWarmup=isWarmup,
        )

        wasTrained = self.trainModel(modelInstance) if shouldTrain else False
        metricValues = self.metrics.update(isAttack, predictedLabel) if thresholdReady else self.emptyMetricValues()

        row = {
            "instanceId": int(instanceIndex),
            "dataset": self.datasetName,
            "model": self.modelCode,
            "modelConfig": self.modelName,
            "normalizer": self.getNormalizerName(),
            "trainingStrategy": self.trainingStrategy.name,
            "thresholdStrategy": self.getThresholdName(),
            "evaluationName": f"{self.modelCode}_{self.getThresholdName()}_{self.trainingStrategy.name}",
            "warmup": self.getWarmupSize(),
            "isWarmup": int(isWarmup),
            "rawScore": rawScore,
            "score": score,
            **scoreAverages,
            "threshold": thresholdValue,
            "thresholdReady": int(thresholdReady),
            "trueLabel": trueLabel,
            "labelName": labelName,
            "isAttack": isAttack,
            "predictedLabel": predictedLabel,
            "trainingAllowed": int(bool(shouldTrain)),
            "wasTrained": int(bool(wasTrained)),
            **metricValues,
        }

        self.normalizer.update(rawValues)
        self.updateThreshold(score, instanceIndex)

        return row

    def updateThreshold(self, score, instanceIndex):
        if self.threshold.isReady():
            self.threshold.update(score, instanceIndex)
            return

        self.thresholdWarmupScores.append(float(score))
        warmupSize = self.getWarmupSize()

        if warmupSize > 0 and len(self.thresholdWarmupScores) == warmupSize:
            self.threshold.initialize(self.thresholdWarmupScores)

    def trainModel(self, modelInstance):
        try:
            self.model.train(modelInstance)
            return True
        except ValueError:
            return False

    def createModelInstance(self, rawInstance, values):
        from capymoa.instance import LabeledInstance

        return LabeledInstance.from_array(
            schema=rawInstance.schema,
            x=np.asarray(values, dtype=np.float64),
            y_index=int(rawInstance.y_index),
        )

    def getStreamSchema(self):
        if hasattr(self.stream, "get_schema"):
            return self.stream.get_schema()

        if hasattr(self.stream, "schema"):
            return self.stream.schema

        raise AttributeError("Não foi possível obter o schema da stream.")

    def getWarmupSize(self):
        if hasattr(self.threshold, "config") and hasattr(self.threshold.config, "warmupSize"):
            return int(self.threshold.config.warmupSize)

        return 0

    def getThresholdName(self):
        state = self.threshold.getState()

        if isinstance(state, dict) and "name" in state:
            return str(state["name"])

        return self.threshold.__class__.__name__

    def getNormalizerName(self):
        normalizerName = self.normalizer.__class__.__name__

        if normalizerName == "NoOnlineNormalizer":
            return "none"

        if normalizerName == "IncrementalZScoreNormalizer":
            return "incrementalZScore"

        return normalizerName

    def emptyMetricValues(self):
        return {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "cumulativeTp": self.metrics.tp,
            "cumulativeTn": self.metrics.tn,
            "cumulativeFp": self.metrics.fp,
            "cumulativeFn": self.metrics.fn,
            "cumulativeAccuracy": np.nan,
            "cumulativePrecision": np.nan,
            "cumulativeRecall": np.nan,
            "cumulativeSpecificity": np.nan,
            "cumulativeF1": np.nan,
            "cumulativeMcc": np.nan,
        }

    def resetComponents(self):
        self.threshold.reset()
        self.normalizer.reset()
        self.scoreMovingAverages.reset()
        self.metrics.reset()
        self.thresholdWarmupScores = []