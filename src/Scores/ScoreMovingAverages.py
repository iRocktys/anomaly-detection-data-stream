from collections import deque


class ScoreMovingAverages:
    def __init__(self, windowSizes=None):
        self.windowSizes = self.validateWindowSizes(windowSizes)
        self.reset()

    def validateWindowSizes(self, windowSizes):
        if windowSizes is None:
            return []

        validatedSizes = sorted(set(int(windowSize) for windowSize in windowSizes))

        if any(windowSize <= 0 for windowSize in validatedSizes):
            raise ValueError("Os tamanhos das janelas devem ser maiores que zero.")

        return validatedSizes

    def calculate(self, score):
        score = float(score)
        self.history.append(score)
        averages = {}

        for windowSize in self.windowSizes:
            values = list(self.history)[-windowSize:]
            averages[f"scoreMa{windowSize}"] = sum(values) / len(values)

        return averages

    def reset(self):
        maximumWindow = max(self.windowSizes, default=1)
        self.history = deque(maxlen=maximumWindow)