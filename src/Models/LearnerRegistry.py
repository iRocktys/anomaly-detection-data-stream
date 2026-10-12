from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class LearnerDefinition:
    code: str
    displayName: str
    className: str
    defaults: dict[str, Any]
    seedParameter: str


class LearnerRegistry:
    definitions = {}
    capymoaModule = ""
    learnerKind = "learner"

    @classmethod
    def normalizeCode(cls, code):
        # Normaliza e valida o código textual do learner solicitado no registry concreto.
        resolved_code = str(code).strip().upper()
        if resolved_code not in cls.definitions:
            raise ValueError(f"Unknown {cls.learnerKind}: {code}. Available: {sorted(cls.definitions)}")
        return resolved_code

    @classmethod
    def validateParameters(cls, code, parameters):
        # Rejeita parâmetros que não pertencem ao contrato conhecido do learner selecionado.
        resolved_code = cls.normalizeCode(code)
        definition = cls.definitions[resolved_code]
        allowed = set(definition.defaults) | {definition.seedParameter, "schema"}
        unknown = sorted(set(parameters) - allowed)
        if unknown:
            raise ValueError(f"Invalid parameters for {resolved_code}: {unknown}.")

    @classmethod
    def create(cls, schema, code, parameters=None, runSeed=1):
        # Cria um learner CapyMOA registrado e devolve sua configuração efetivamente utilizada.
        resolved_code = cls.normalizeCode(code)
        definition = cls.definitions[resolved_code]
        resolved_parameters = dict(definition.defaults)
        resolved_parameters.update(dict(parameters or {}))
        resolved_parameters["schema"] = schema
        resolved_parameters.setdefault(definition.seedParameter, int(runSeed))
        cls.validateParameters(resolved_code, resolved_parameters)
        try:
            module = import_module(f"capymoa.{cls.capymoaModule}")
        except ImportError as error:
            raise ImportError("CapyMOA is not installed in the environment.") from error
        model_class = getattr(module, definition.className)
        try:
            model = model_class(**resolved_parameters)
        except TypeError as error:
            raise TypeError(f"Failed to create {resolved_code} with parameters {resolved_parameters}.") from error
        persisted_parameters = {key: value for key, value in resolved_parameters.items() if key != "schema"}
        return definition.displayName, model, persisted_parameters

    @classmethod
    def describe(cls):
        # Retorna os learners registrados e os respectivos parâmetros padrão persistíveis.
        return {
            code: {
                "displayName": definition.displayName,
                "defaults": dict(definition.defaults),
                "seedParameter": definition.seedParameter,
            }
            for code, definition in cls.definitions.items()
        }
