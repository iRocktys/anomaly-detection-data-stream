from src.Training.TrainAllStrategy import TrainAllStrategy
from src.Training.TrainBelowThresholdStrategy import TrainBelowThresholdStrategy


class TrainingStrategyRegistry:
    factories = {
        "all": TrainAllStrategy,
        "belowthreshold": TrainBelowThresholdStrategy,
    }

    @classmethod
    def create(cls, name, parameters=None):
        # Cria a estratégia de treinamento selecionada pelo experimento.
        key = str(name).strip().lower().replace("_", "").replace("-", "")
        if key not in cls.factories:
            raise ValueError(f"Estratégia desconhecida: {name}. Disponíveis: {sorted(cls.factories)}")
        return cls.factories[key](**dict(parameters or {}))
