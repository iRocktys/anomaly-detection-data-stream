class TrainAllStrategy:
    name = "all"

    def shouldTrain(self, prediction=None, thresholdReady=False, isWarmup=False):
        # Autoriza o treinamento do detector em todas as instâncias da stream.
        return True
