from collections import deque

import numpy as np
from capymoa.instance import LabeledInstance


class StreamPipeline:
    def __init__(self, config, components, featureNames):
        self.config = config
        self.components = components
        self.featureNames = list(featureNames)
        self.scoreHistory = deque(maxlen=max(config.movingAverageWindows or [1]))

    def run(self, stream, runId, modelCode, normalizerName):
        stream.restart()
        rows = []
        instanceId = 0

        while stream.has_more_instances():
            rawInstance = stream.next_instance()
            row = self.processInstance(
                rawInstance=rawInstance,
                runId=runId,
                instanceId=instanceId,
                modelCode=modelCode,
                normalizerName=normalizerName,
            )
            rows.append(row)
            instanceId += 1

        return rows

    def processInstance(self, rawInstance, runId, instanceId, modelCode, normalizerName):
        rawValues = np.asarray(rawInstance.x, dtype=np.float64)
        selectedValues = self.components["featureSelector"].transform(rawValues)
        extractedValues = self.components["featureExtractor"].transform(selectedValues)
        normalizedValues = self.components["normalizer"].transform(extractedValues)
        smoothedValues = self.components["featureSmoother"].transform(normalizedValues)
        modelInstance = self.createInstance(rawInstance, smoothedValues)

        rawScore = float(self.components["model"].score_instance(modelInstance))
        smoothScore = float(self.components["scoreSmoother"].transform(rawScore))
        thresholdValue = float(self.components["thresholdStrategy"].getThreshold())
        thresholdReady = bool(self.components["thresholdStrategy"].isReady())
        prediction = int(self.components["decisionStrategy"].predict(
            smoothScore,
            thresholdValue,
            thresholdReady,
        ))
        trueLabel = int(rawInstance.y_index)
        attackLabel = int(trueLabel != self.config.normalClassIndex)

        self.scoreHistory.append(smoothScore)
        row = self.createRow(
            runId=runId,
            instanceId=instanceId,
            modelCode=modelCode,
            normalizerName=normalizerName,
            trueLabel=trueLabel,
            attackLabel=attackLabel,
            rawScore=rawScore,
            smoothScore=smoothScore,
            prediction=prediction,
            thresholdValue=thresholdValue,
            thresholdReady=thresholdReady,
            normalizedValues=normalizedValues,
        )

        if self.components["trainingStrategy"].shouldTrain(prediction, trueLabel):
            try:
                self.components["model"].train(modelInstance)
            except ValueError:
                pass

        self.updateComponents(
            rawValues=rawValues,
            selectedValues=selectedValues,
            extractedValues=extractedValues,
            normalizedValues=normalizedValues,
            rawScore=rawScore,
            smoothScore=smoothScore,
            attackLabel=attackLabel,
            trueLabel=trueLabel,
            prediction=prediction,
            thresholdValue=thresholdValue,
        )
        return row

    def updateComponents(self, rawValues, selectedValues, extractedValues, normalizedValues, rawScore, smoothScore, attackLabel, trueLabel, prediction, thresholdValue):
        self.components["featureSelector"].update(rawValues)
        self.components["featureExtractor"].update(selectedValues)

        if self.shouldUpdateNormalizer(attackLabel):
            self.components["normalizer"].update(extractedValues)

        self.components["featureSmoother"].update(normalizedValues)
        self.components["scoreSmoother"].update(rawScore)
        self.components["decisionStrategy"].update(
            smoothScore,
            thresholdValue,
            prediction,
            trueLabel,
        )
        self.components["thresholdStrategy"].update(smoothScore)

    def shouldUpdateNormalizer(self, attackLabel):
        policyName = self.config.normalizerUpdatePolicy
        if policyName == "none":
            return False
        if policyName == "all":
            return True
        return attackLabel == 0

    def createInstance(self, rawInstance, values):
        return LabeledInstance.from_array(
            schema=rawInstance.schema,
            x=np.asarray(values, dtype=np.float64),
            y_index=int(rawInstance.y_index),
        )

    def createRow(self, runId, instanceId, modelCode, normalizerName, trueLabel, attackLabel, rawScore, smoothScore, prediction, thresholdValue, thresholdReady, normalizedValues):
        row = {
            "runId": runId,
            "dataset": self.config.datasetName,
            "instanceId": instanceId,
            "trueLabel": trueLabel,
            "isAttack": attackLabel,
            "modelCode": modelCode,
            "modelName": self.components["modelName"],
            "normalizer": normalizerName,
            "normalizerUpdatePolicy": self.config.normalizerUpdatePolicy,
            "trainingStrategy": self.config.trainingStrategy.name,
            "rawScore": rawScore,
            "score": smoothScore,
            "prediction": prediction,
            "thresholdStrategy": self.config.thresholdStrategy.name,
            "threshold": thresholdValue,
            "thresholdReady": thresholdReady,
        }

        scoreValues = list(self.scoreHistory)
        for windowSize in self.config.movingAverageWindows:
            recentScores = scoreValues[-max(1, int(windowSize)):]
            row[f"scoreMa{windowSize}"] = float(np.mean(recentScores))

        if self.config.saveNormalizedFeatures:
            for featureIndex, featureName in enumerate(self.featureNames):
                if featureIndex < len(normalizedValues):
                    row[str(featureName)] = float(normalizedValues[featureIndex])
        return row
