from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ModelDefinition:
    code: str
    displayName: str
    defaults: dict[str, Any]
    seedParameter: str
    loader: Callable[[], type]


class ModelRegistry:
    definitions = {
        "AIF": ModelDefinition(
            code="AIF",
            displayName="AdaptiveIsolationForest",
            defaults={
                "window_size": 256,
                "n_trees": 100,
                "height": None,
                "m_trees": 10,
                "weights": 0.5,
            },
            seedParameter="seed",
            loader=lambda: ModelRegistry.loadCapymoaClass("AdaptiveIsolationForest"),
        ),
        "HST": ModelDefinition(
            code="HST",
            displayName="HalfSpaceTrees",
            defaults={
                "CLI": None,
                "window_size": 250,
                "number_of_trees": 25,
                "max_depth": 15,
                "anomaly_threshold": 0.50,
                "size_limit": 0.10,
            },
            seedParameter="random_seed",
            loader=lambda: ModelRegistry.loadCapymoaClass("HalfSpaceTrees"),
        ),
        "AE": ModelDefinition(
            code="AE",
            displayName="Autoencoder",
            defaults={
                "hidden_layer": 2,
                "learning_rate": 0.5,
                "threshold": 0.60,
            },
            seedParameter="random_seed",
            loader=lambda: ModelRegistry.loadCapymoaClass("Autoencoder"),
        ),
        "SRHF": ModelDefinition(
            code="StreamRHF",
            displayName="StreamRHF",
            defaults={
                "max_height": 5,
                "num_trees": 100,
                "window_size": 20,
                "random_seed": 0,
            },
            seedParameter="random_seed",
            loader=lambda: ModelRegistry.loadCapymoaClass("StreamRHF"),
        ),
    }

    @staticmethod
    def loadCapymoaClass(className):
        try:
            from capymoa import anomaly
        except ImportError as error:
            raise ImportError(
                "CapyMOA não está instalado no ambiente. Instale-o antes de executar os modelos."
            ) from error
        return getattr(anomaly, className)

    @classmethod
    def normalizeCode(cls, code):
        resolvedCode = str(code).strip().upper()
        if resolvedCode not in cls.definitions:
            supported = ", ".join(sorted(cls.definitions))
            raise ValueError(f"Modelo desconhecido: {code}. Disponíveis: {supported}.")
        return resolvedCode

    @classmethod
    def validateParameters(cls, code, parameters):
        resolvedCode = cls.normalizeCode(code)
        definition = cls.definitions[resolvedCode]
        allowed = set(definition.defaults) | {definition.seedParameter, "schema"}
        unknown = sorted(set(parameters) - allowed)
        if unknown:
            raise ValueError(
                f"Parâmetros inválidos para {resolvedCode}: {unknown}. "
                f"Permitidos: {sorted(allowed - {'schema'})}."
            )

    @classmethod
    def create(cls, schema, modelConfig, runSeed):
        code = cls.normalizeCode(modelConfig.code)
        definition = cls.definitions[code]
        parameters = dict(definition.defaults)
        parameters.update(dict(modelConfig.parameters))
        cls.validateParameters(code, parameters)
        parameters["schema"] = schema
        parameters.setdefault(definition.seedParameter, int(runSeed))
        modelClass = definition.loader()
        try:
            model = modelClass(**parameters)
        except TypeError as error:
            raise TypeError(
                f"Falha ao criar {code} com parâmetros {parameters}. "
                "Confira a versão do CapyMOA e os nomes dos parâmetros."
            ) from error
        persistedParameters = {key: value for key, value in parameters.items() if key != "schema"}
        return definition.displayName, model, persistedParameters

    @classmethod
    def describe(cls):
        return {
            code: {
                "displayName": definition.displayName,
                "defaults": dict(definition.defaults),
                "seedParameter": definition.seedParameter,
            }
            for code, definition in cls.definitions.items()
        }
