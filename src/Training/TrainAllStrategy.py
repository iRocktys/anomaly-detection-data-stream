class TrainAllStrategy:
    name = "all"

    def shouldTrain(self, prediction=None, thresholdReady=False, isWarmup=False):
        return True