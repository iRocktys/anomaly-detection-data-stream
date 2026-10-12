from dataclasses import dataclass
from typing import Any, Callable

from ProjectDefaults import getClassifierDefaults


@dataclass(frozen=True)
class ClassifierDefinition:
    code: str
    displayName: str
    defaults: dict[str, Any]
    seedParameter: str
    loader: Callable[[], type]


class ClassifierRegistry:
    definitions = {
        "ARF": ClassifierDefinition(
            code="ARF",
            displayName="AdaptiveRandomForestClassifier",
            defaults=getClassifierDefaults("ARF"),
            seedParameter="random_seed",
            loader=lambda: ClassifierRegistry.loadCapymoaClass("AdaptiveRandomForestClassifier"),
        ),
        "HAT": ClassifierDefinition(
            code="HAT",
            displayName="HoeffdingAdaptiveTree",
            defaults=getClassifierDefaults("HAT"),
            seedParameter="random_seed",
            loader=lambda: ClassifierRegistry.loadCapymoaClass("HoeffdingAdaptiveTree"),
        ),
        "HT": ClassifierDefinition(
            code="HT",
            displayName="HoeffdingTree",
            defaults=getClassifierDefaults("HT"),
            seedParameter="random_seed",
            loader=lambda: ClassifierRegistry.loadCapymoaClass("HoeffdingTree"),
        ),
        "NB": ClassifierDefinition(
            code="NB",
            displayName="NaiveBayes",
            defaults=getClassifierDefaults("NB"),
            seedParameter="random_seed",
            loader=lambda: ClassifierRegistry.loadCapymoaClass("NaiveBayes"),
        ),
    }

    @staticmethod
    def loadCapymoaClass(className):
        # Carrega a classe do classificador diretamente do módulo oficial do CapyMOA.
        try:
            from capymoa import classifier
        except ImportError as error:
            raise ImportError("CapyMOA is not installed in the environment.") from error
        return getattr(classifier, className)

    @classmethod
    def normalizeCode(cls, code):
        # Normaliza e valida o código textual do classificador solicitado.
        resolved_code = str(code).strip().upper()
        if resolved_code not in cls.definitions:
            supported = ", ".join(sorted(cls.definitions))
            raise ValueError(f"Unknown classifier: {code}. Available: {supported}.")
        return resolved_code

    @classmethod
    def validateParameters(cls, code, parameters):
        # Rejeita parâmetros que não pertencem ao contrato conhecido do classificador.
        resolved_code = cls.normalizeCode(code)
        definition = cls.definitions[resolved_code]
        allowed = set(definition.defaults) | {definition.seedParameter, "schema"}
        unknown = sorted(set(parameters) - allowed)
        if unknown:
            raise ValueError(f"Invalid parameters for {resolved_code}: {unknown}.")

    @classmethod
    def describe(cls):
        # Retorna os classificadores registrados e os respectivos parâmetros padrão.
        return {
            code: {
                "displayName": definition.displayName,
                "defaults": dict(definition.defaults),
                "seedParameter": definition.seedParameter,
            }
            for code, definition in cls.definitions.items()
        }
