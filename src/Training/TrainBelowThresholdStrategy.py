class TrainBelowThresholdStrategy:
    name = "belowThreshold"

    def __init__(self, trainDuringWarmup=True):
        self.trainDuringWarmup = bool(trainDuringWarmup)

    def shouldTrain(self, prediction=None, thresholdReady=False, isWarmup=False):
        if isWarmup or not thresholdReady:
            return self.trainDuringWarmup

        if prediction is None:
            raise ValueError("A estratégia abaixo do threshold requer uma predição.")

        return int(prediction) == 0