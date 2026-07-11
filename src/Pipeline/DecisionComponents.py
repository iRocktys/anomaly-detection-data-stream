class ThresholdDecisionStrategy:
    def predict(self, score, threshold, thresholdReady=True):
        if not thresholdReady:
            return 0
        return int(float(score) > float(threshold))

    def update(self, score, threshold, prediction, trueLabel=None):
        return None


class DecisionComponents:
    @staticmethod
    def createStrategy(config):
        strategyName = str(config.name).strip().lower()
        if strategyName in {"threshold", "binary"}:
            return ThresholdDecisionStrategy()
        raise ValueError(f"Estratégia de decisão desconhecida: {config.name}")
