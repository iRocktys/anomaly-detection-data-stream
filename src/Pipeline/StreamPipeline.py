from collections import deque

import numpy as np


class StreamPipeline:
    """Gera scores causais. Thresholds e decisões não pertencem a esta etapa."""

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

        if self.components["trainingStrategy"].shouldTrain(trueLabel, isAttack):
            try:
                self.components["model"].train(modelInstance)
            except ValueError:
                pass

        self.components["featureExtractor"].update(rawValues)
        if self.shouldUpdateNormalizer(isAttack):
            self.components["normalizer"].update(extractedValues)
        self.components["featureSmoother"].update(normalizedValues)
        return row

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
            "trainingStrategy": self.plan.trainingStrategy.name,
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
