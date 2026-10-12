from ProjectDefaults import getModelDefaults
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
            defaults=getModelDefaults("AIF"),
            seedParameter="seed",
            loader=lambda: ModelRegistry.loadCapymoaClass("AdaptiveIsolationForest"),
        ),
        "HST": ModelDefinition(
            code="HST",
            displayName="HalfSpaceTrees",
            defaults=getModelDefaults("HST"),
            seedParameter="random_seed",
            loader=lambda: ModelRegistry.loadCapymoaClass("HalfSpaceTrees"),
        ),
        "AE": ModelDefinition(
            code="AE",
            displayName="Autoencoder",
            defaults=getModelDefaults("AE"),
            seedParameter="random_seed",
            loader=lambda: ModelRegistry.loadCapymoaClass("Autoencoder"),
        ),
        "SRHF": ModelDefinition(
            code="StreamRHF",
            displayName="StreamRHF",
            defaults=getModelDefaults("SRHF"),
            seedParameter="random_seed",
            loader=lambda: ModelRegistry.loadCapymoaClass("StreamRHF"),
        ),
    }

    @staticmethod
    def loadCapymoaClass(className):
        # Carrega a classe correspondente diretamente do módulo disponível no CapyMOA.
        try:
            from capymoa import anomaly
        except ImportError as error:
            raise ImportError(
                "CapyMOA is not installed in the environment. Install it before running the models."
            ) from error
        return getattr(anomaly, className)

    @classmethod
    def normalizeCode(cls, code):
        # Normaliza e valida o código textual informado para o modelo.
        resolvedCode = str(code).strip().upper()
        if resolvedCode not in cls.definitions:
            supported = ", ".join(sorted(cls.definitions))
            raise ValueError(f"Unknown model: {code}. Available: {supported}.")
        return resolvedCode

    @classmethod
    def validateParameters(cls, code, parameters):
        # Valida se os parâmetros informados pertencem ao contrato do modelo selecionado.
        resolvedCode = cls.normalizeCode(code)
        definition = cls.definitions[resolvedCode]
        allowed = set(definition.defaults) | {definition.seedParameter, "schema"}
        unknown = sorted(set(parameters) - allowed)
        if unknown:
            raise ValueError(
                f"Invalid parameters for {resolvedCode}: {unknown}. "
                f"Allowed: {sorted(allowed - {'schema'})}."
            )

    @classmethod
    def create(cls, schema, modelConfig, runSeed):
        # Cria a implementação registrada usando o nome e os parâmetros informados.
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
                f"Failed to create {code} with parameters {parameters}. "
                "Check the installed CapyMOA version and parameter names."
            ) from error
        persistedParameters = {key: value for key, value in parameters.items() if key != "schema"}
        return definition.displayName, model, persistedParameters

    @classmethod
    def describe(cls):
        # Retorna uma descrição tabular das implementações registradas e seus parâmetros padrão.
        return {
            code: {
                "displayName": definition.displayName,
                "defaults": dict(definition.defaults),
                "seedParameter": definition.seedParameter,
            }
            for code, definition in cls.definitions.items()
        }
