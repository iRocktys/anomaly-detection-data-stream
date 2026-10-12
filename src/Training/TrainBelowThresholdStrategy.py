class TrainBelowThresholdStrategy:
    name = "belowThreshold"

    def __init__(self, trainDuringWarmup=True):
        # Inicializa o estado e as dependências necessárias para a execução da classe.
        self.trainDuringWarmup = bool(trainDuringWarmup)

    def shouldTrain(self, prediction=None, thresholdReady=False, isWarmup=False):
        # Decide se a instância atual pode ser utilizada no treinamento incremental.
        if isWarmup or not thresholdReady:
            return self.trainDuringWarmup

        if prediction is None:
            raise ValueError("The below-threshold strategy requires a prediction.")

        return int(prediction) == 0
