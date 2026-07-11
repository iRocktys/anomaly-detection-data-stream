import numpy as np

class Smoothing:
    def causal_moving_average(values, window):
        values = list(values)

        if len(values) == 0:
            return 0.0

        window = max(1, int(window))
        recent = values[-window:]

        return float(np.mean(recent))

    def build_causal_moving_average_series(values, window):
        values = list(values)

        if len(values) == 0:
            return []

        window = max(1, int(window))

        return [
            float(np.mean(values[max(0, i - window + 1):i + 1]))
            for i in range(len(values))
        ]