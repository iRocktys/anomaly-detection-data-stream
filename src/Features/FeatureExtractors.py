import numpy as np


class NoFeatureExtractor:
    def transform(self, values):
        return np.asarray(values, dtype=np.float64)

    def update(self, values):
        return None


class FeatureExtractors:
    @staticmethod
    def createExtractor(config):
        extractorName = str(config.name).strip().lower()
        if extractorName == "none":
            return NoFeatureExtractor()
        raise ValueError(f"Extrator de features desconhecido: {config.name}")
