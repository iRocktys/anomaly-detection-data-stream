import numpy as np


class NoFeatureExtractor:
    def transform(self, values):
        return np.asarray(values, dtype=np.float64)

    def update(self, values):
        return None

    def reset(self):
        return None
