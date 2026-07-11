import numpy as np


class SelectedFeatureSelector:
    def transform(self, values):
        return np.asarray(values, dtype=np.float64)

    def update(self, values):
        return None


class FeatureSelectors:
    @staticmethod
    def createSelector(config):
        selectorName = str(config.name).strip().lower()
        if selectorName in {"selected", "all", "none"}:
            return SelectedFeatureSelector()
        raise ValueError(f"Seletor de features desconhecido: {config.name}")
