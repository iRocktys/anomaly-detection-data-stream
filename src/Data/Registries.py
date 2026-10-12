import numpy as np

from src.Data.OnlineImputers import IncrementalMeanImputer
from src.Data.OnlineNormalizers import BaseOnlineNormalizer, IncrementalZScoreNormalizer


class NoOnlineImputer:
    name = "none"

    def transform(self, values):
        # Retorna os valores sem imputação e rejeita dados ausentes antes do modelo.
        prepared = np.asarray(values, dtype=np.float64).copy()
        if np.any(~np.isfinite(prepared)):
            raise ValueError("O imputador 'none' recebeu valores ausentes ou não finitos.")
        return prepared

    def update(self, values):
        # Mantém a interface incremental sem atualizar estado.
        return None

    def reset(self):
        # Mantém a interface incremental sem estado interno.
        return None


class NoOnlineNormalizer(BaseOnlineNormalizer):
    name = "none"

    def transform(self, values):
        # Retorna os valores sem normalização após validar finitude.
        return self.prepareValues(values)


class ImputerRegistry:
    factories = {
        "incrementalmean": IncrementalMeanImputer,
        "none": NoOnlineImputer,
    }

    @classmethod
    def create(cls, name, parameters=None):
        # Cria um imputador a partir de um nome textual e parâmetros opcionais.
        key = str(name).strip().lower().replace("_", "").replace("-", "")
        if key not in cls.factories:
            raise ValueError(f"Imputador desconhecido: {name}. Disponíveis: {sorted(cls.factories)}")
        return cls.factories[key](**dict(parameters or {}))


class NormalizerRegistry:
    factories = {
        "incrementalzscore": IncrementalZScoreNormalizer,
        "none": NoOnlineNormalizer,
    }

    @classmethod
    def create(cls, name, parameters=None):
        # Cria um normalizador a partir de um nome textual e parâmetros opcionais.
        key = str(name).strip().lower().replace("_", "").replace("-", "")
        if key not in cls.factories:
            raise ValueError(f"Normalizador desconhecido: {name}. Disponíveis: {sorted(cls.factories)}")
        return cls.factories[key](**dict(parameters or {}))
