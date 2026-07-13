from dataclasses import dataclass
from typing import Callable

from src.Anomaly.Thresholds.FixedThreshold import FixedThreshold
from src.Anomaly.Thresholds.Incremental.DspotThreshold import DspotThreshold


@dataclass(frozen=True)
class ThresholdDefinition:
    name: str
    builder: Callable
    usesGlobalWarmup: bool = True
    aliases: tuple[str, ...] = ()


class ThresholdRegistry:
    """Registro extensível das técnicas de threshold disponíveis.

    Atualmente estão registrados o limiar fixo e o DSPOT. Para incluir uma
    nova técnica dinâmica posteriormente:

    1. adicione sua classe em ``src/Anomaly/Thresholds/Incremental``;
    2. importe a classe neste arquivo;
    3. adicione uma ``ThresholdDefinition`` em ``definitions``;
    4. selecione o novo nome em ``thresholdName`` no experimento.

    Técnicas com ``usesGlobalWarmup=True`` recebem automaticamente o warmup do
    ``ExperimentPlan``. Assim, cada novo threshold compartilha o mesmo período
    global de aquecimento sem aceitar um warmup independente na configuração.
    """

    definitions = {
        "fixed": ThresholdDefinition(
            name="fixed",
            builder=FixedThreshold,
            usesGlobalWarmup=False,
            aliases=("static", "constant"),
        ),
        "dspot": ThresholdDefinition(
            name="dspot",
            builder=DspotThreshold,
            usesGlobalWarmup=True,
            aliases=("dynamicspot",),
        ),
    }

    @staticmethod
    def normalizeName(name):
        return "".join(
            character
            for character in str(name or "").lower()
            if character.isalnum()
        )

    @classmethod
    def resolveDefinition(cls, name):
        normalizedName = cls.normalizeName(name)
        for key, definition in cls.definitions.items():
            names = {cls.normalizeName(key), cls.normalizeName(definition.name)}
            names.update(cls.normalizeName(alias) for alias in definition.aliases)
            if normalizedName in names:
                return key, definition
        available = ", ".join(sorted(cls.definitions))
        raise ValueError(
            f"Threshold desconhecido: {name}. Disponíveis atualmente: {available}."
        )

    @classmethod
    def validateConfig(cls, config):
        _, definition = cls.resolveDefinition(config.name)
        parameters = dict(config.parameters or {})
        if definition.usesGlobalWarmup:
            forbidden = {
                "warmup",
                "minimumSamples",
                "minimum_samples",
            } & set(parameters)
            if forbidden:
                raise ValueError(
                    "O aquecimento dos thresholds dinâmicos é definido pelo warmup "
                    f"global do experimento; remova {sorted(forbidden)}."
                )

    @classmethod
    def create(cls, config, globalWarmup):
        _, definition = cls.resolveDefinition(config.name)
        cls.validateConfig(config)
        parameters = dict(config.parameters or {})
        if definition.usesGlobalWarmup:
            parameters["warmup"] = int(globalWarmup)
        try:
            return definition.builder(**parameters)
        except TypeError as error:
            raise TypeError(
                f"Parâmetros inválidos para threshold/{config.name}: {parameters}."
            ) from error

    @classmethod
    def describe(cls):
        return {
            key: {
                "name": definition.name,
                "usesGlobalWarmup": definition.usesGlobalWarmup,
                "aliases": list(definition.aliases),
            }
            for key, definition in cls.definitions.items()
        }
