class TrainAllStrategy:
    def shouldTrain(self, prediction, trueLabel):
        return True


class TrainNormalPredictionStrategy:
    def shouldTrain(self, prediction, trueLabel):
        return int(prediction) == 0


class NoTrainingStrategy:
    def shouldTrain(self, prediction, trueLabel):
        return False


class TrainingStrategies:
    @staticmethod
    def createStrategy(config):
        strategyName = str(config.name).strip().lower()
        if strategyName == "all":
            return TrainAllStrategy()
        if strategyName in {"normalprediction", "normalonly"}:
            return TrainNormalPredictionStrategy()
        if strategyName == "none":
            return NoTrainingStrategy()
        raise ValueError(f"Estratégia de treinamento desconhecida: {config.name}")
