from collections import deque

import numpy as np


class StreamPipeline:
    """Gera scores causais e executa a política de treinamento online.

    Com ``trainingStrategy=all`` o modelo aprende com todas as instâncias e os
    thresholds permanecem desacoplados para replay posterior. Com
    ``trainingStrategy=predictedNormal`` a avaliação escolhida é executada no
    próprio fluxo, pois sua predição define se a instância atual pode treinar o
    modelo e, consequentemente, altera os scores futuros.
    """

    def __init__(self, plan, datasetConfig, components, featureNames):
        self.plan = plan
        self.datasetConfig = datasetConfig
        self.components = components
        self.featureNames = list(featureNames)
        maximumWindow = max(plan.movingAverageWindows or [1])
        self.scoreHistory = deque(maxlen=maximumWindow)

    def run(self, stream, metadata):
        stream.restart()
        rows = []
        instanceId = 0
        while stream.has_more_instances():
            rawInstance = stream.next_instance()
            rows.append(self.processInstance(rawInstance, instanceId, metadata))
            instanceId += 1
        return rows

    def processInstance(self, rawInstance, instanceId, metadata):
        rawValues = np.asarray(rawInstance.x, dtype=np.float64)
        extractedValues = self.components["featureExtractor"].transform(rawValues)
        normalizedValues = self.components["normalizer"].transform(extractedValues)
        smoothedValues = self.components["featureSmoother"].transform(normalizedValues)
        modelInstance = self.createInstance(rawInstance, smoothedValues)

        rawScore = float(self.components["model"].score_instance(modelInstance))
        trueLabel = int(rawInstance.y_index)
        isAttack = int(trueLabel != self.datasetConfig.normalClassIndex)
        self.scoreHistory.append(rawScore)

        row = self.createRow(
            instanceId=instanceId,
            metadata=metadata,
            trueLabel=trueLabel,
            isAttack=isAttack,
            rawScore=rawScore,
            normalizedValues=normalizedValues,
        )

        trainingDecision = self.evaluateTrainingDecision(
            row=row,
            instanceId=instanceId,
            trueLabel=trueLabel,
            isAttack=isAttack,
        )
        feedbackComponents = trainingDecision.pop("feedbackComponents", None)
        shouldTrain = self.components["trainingStrategy"].shouldTrain(
            prediction=trainingDecision.get("trainingPrediction"),
            thresholdReady=trainingDecision.get("trainingThresholdReady", False),
            isWarmup=bool(row["isWarmup"]),
            trueLabel=trueLabel,
            isAttack=isAttack,
        )
        row.update(trainingDecision)
        row["trainingAllowed"] = int(bool(shouldTrain))
        row["wasTrained"] = int(self.trainModel(modelInstance) if shouldTrain else False)

        self.components["featureExtractor"].update(rawValues)
        if self.shouldUpdateNormalizer(isAttack):
            self.components["normalizer"].update(extractedValues)
        self.components["featureSmoother"].update(normalizedValues)
        self.updateTrainingDecision(trainingDecision, trueLabel, feedbackComponents)
        return row

    def evaluateTrainingDecision(self, row, instanceId, trueLabel, isAttack):
        feedbackEvaluation = self.components.get("feedbackEvaluation")
        feedbackComponents = self.components.get("feedbackComponents")
        if feedbackEvaluation is None or feedbackComponents is None:
            return {
                "trainingFeedbackEvaluation": None,
                "trainingScore": np.nan,
                "trainingThreshold": np.nan,
                "trainingThresholdReady": False,
                "trainingPrediction": np.nan,
            }

        scoreColumn = feedbackEvaluation.scoreColumn
        if scoreColumn not in row:
            raise ValueError(
                f"A coluna {scoreColumn} não está disponível durante o treinamento online. "
                f"Disponíveis: {list(row)}"
            )
        sourceScore = float(row[scoreColumn])
        smoothScore = float(feedbackComponents["scoreSmoother"].transform(sourceScore))
        thresholdValue = float(feedbackComponents["threshold"].getThreshold())
        isWarmup = instanceId < int(self.plan.warmup)
        thresholdReady = bool(feedbackComponents["threshold"].isReady()) and not isWarmup
        prediction = int(
            feedbackComponents["decision"].predict(
                smoothScore,
                thresholdValue,
                thresholdReady,
            )
        )
        return {
            "trainingFeedbackEvaluation": feedbackEvaluation.name,
            "trainingSourceScore": sourceScore,
            "trainingScore": smoothScore,
            "trainingThreshold": thresholdValue,
            "trainingThresholdReady": bool(thresholdReady),
            "trainingPrediction": prediction,
            "feedbackComponents": feedbackComponents,
        }

    def updateTrainingDecision(self, trainingDecision, trueLabel, feedbackComponents):
        if feedbackComponents is None:
            return
        sourceScore = float(trainingDecision["trainingSourceScore"])
        smoothScore = float(trainingDecision["trainingScore"])
        thresholdValue = float(trainingDecision["trainingThreshold"])
        prediction = int(trainingDecision["trainingPrediction"])
        feedbackComponents["scoreSmoother"].update(sourceScore)
        feedbackComponents["decision"].update(
            smoothScore,
            thresholdValue,
            prediction,
            int(trueLabel),
        )
        feedbackComponents["threshold"].update(smoothScore)

    def trainModel(self, modelInstance):
        try:
            self.components["model"].train(modelInstance)
            return True
        except ValueError:
            return False

    def shouldUpdateNormalizer(self, isAttack):
        policy = self.plan.normalizerUpdatePolicy
        if policy == "none":
            return False
        if policy == "all":
            return True
        return int(isAttack) == 0

    def createInstance(self, rawInstance, values):
        try:
            from capymoa.instance import LabeledInstance
        except ImportError as error:
            raise ImportError("CapyMOA é necessário para executar o StreamPipeline.") from error
        return LabeledInstance.from_array(
            schema=rawInstance.schema,
            x=np.asarray(values, dtype=np.float64),
            y_index=int(rawInstance.y_index),
        )

    def createRow(self, instanceId, metadata, trueLabel, isAttack, rawScore, normalizedValues):
        targetNames = metadata.get("targetNames", [])
        labelName = (
            str(targetNames[trueLabel])
            if 0 <= int(trueLabel) < len(targetNames)
            else str(trueLabel)
        )
        row = {
            "runId": metadata["runId"],
            "scoreArtifactId": metadata["scoreArtifactId"],
            "dataset": self.datasetConfig.name,
            "instanceId": int(instanceId),
            "trueLabel": int(trueLabel),
            "labelName": labelName,
            "isAttack": int(isAttack),
            "modelCode": metadata["modelCode"],
            "modelConfig": metadata["modelConfig"],
            "modelName": self.components["modelName"],
            "normalizer": metadata["normalizer"],
            "normalizerUpdatePolicy": self.plan.normalizerUpdatePolicy,
            "trainingStrategy": metadata["trainingStrategy"],
            "warmup": int(self.plan.warmup),
            "isWarmup": int(instanceId < int(self.plan.warmup)),
            "runSeed": int(metadata["runSeed"]),
            "rawScore": float(rawScore),
        }
        scoreValues = list(self.scoreHistory)
        for windowSize in self.plan.movingAverageWindows:
            recentScores = scoreValues[-max(1, int(windowSize)):]
            row[f"scoreMa{int(windowSize)}"] = float(np.mean(recentScores))

        if self.plan.output.saveNormalizedFeatures:
            for featureIndex, featureName in enumerate(self.featureNames):
                if featureIndex < len(normalizedValues):
                    row[str(featureName)] = float(normalizedValues[featureIndex])
        return row
