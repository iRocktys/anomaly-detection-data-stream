import numpy as np

from src.Data.OnlineImputers import IncrementalMeanImputer
from src.Data.OnlineNormalizers import BaseOnlineNormalizer, IncrementalZScoreNormalizer


class NoOnlineImputer:
    name = "none"

    def transform(self, values):
        # Retorna os valores sem imputação e rejeita qualquer entrada não finita.
        prepared_values = np.asarray(values, dtype=np.float64).copy()
        if np.any(~np.isfinite(prepared_values)):
            raise ValueError("The 'none' imputer received missing or non-finite values.")
        return prepared_values

    def update(self, values):
        # Mantém a interface incremental sem modificar estado.
        return None

    def reset(self):
        # Mantém a interface incremental sem estado interno.
        return None


class NoOnlineNormalizer(BaseOnlineNormalizer):
    name = "none"

    def transform(self, values):
        # Retorna os valores sem normalização depois de validar que são finitos.
        return self.prepareValues(values)


class ImputerRegistry:
    factories = {
        "incrementalmean": IncrementalMeanImputer,
        "none": NoOnlineImputer,
    }

    @classmethod
    def create(cls, name, parameters=None):
        # Cria o imputador configurado a partir de um nome textual normalizado.
        key = str(name).strip().lower().replace("_", "").replace("-", "")
        if key not in cls.factories:
            raise ValueError(f"Unknown imputer: {name}. Available: {sorted(cls.factories)}")
        return cls.factories[key](**dict(parameters or {}))


class NormalizerRegistry:
    factories = {
        "incrementalzscore": IncrementalZScoreNormalizer,
        "none": NoOnlineNormalizer,
    }

    @classmethod
    def create(cls, name, parameters=None):
        # Cria o normalizador configurado a partir de um nome textual normalizado.
        key = str(name).strip().lower().replace("_", "").replace("-", "")
        if key not in cls.factories:
            raise ValueError(f"Unknown normalizer: {name}. Available: {sorted(cls.factories)}")
        return cls.factories[key](**dict(parameters or {}))
