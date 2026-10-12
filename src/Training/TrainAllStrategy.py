class TrainAllStrategy:
    name = "all"

    def shouldTrain(self, prediction=None, thresholdReady=False, isWarmup=False):
        # Decide se a instância atual pode ser utilizada no treinamento incremental.
        return True
