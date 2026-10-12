class TrainBelowThresholdStrategy:
    name = "belowThreshold"

    def __init__(self, trainDuringWarmup=True):
        # Configura se o detector também deve treinar antes de o threshold estar disponível.
        self.trainDuringWarmup = bool(trainDuringWarmup)

    def shouldTrain(self, prediction=None, thresholdReady=False, isWarmup=False):
        # Treina somente instâncias consideradas normais depois da fase de warm-up.
        if isWarmup or not thresholdReady:
            return self.trainDuringWarmup
        if prediction is None:
            raise ValueError("The below-threshold strategy requires a prediction.")
        return int(prediction) == 0
