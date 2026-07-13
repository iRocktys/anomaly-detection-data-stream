class ThresholdDecisionStrategy:
    def predict(self, score, threshold, thresholdReady=True):
        if not thresholdReady:
            return 0
        return int(float(score) > float(threshold))

    def update(self, score, threshold, prediction, trueLabel=None):
        return None

    def reset(self):
        return None
