from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy

import ProjectDefaults as defaults


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    targetColumn: str = defaults.DEFAULT_TARGET_COLUMN
    selectedFeatures: tuple[str, ...] | None = defaults.DEFAULT_SELECTED_FEATURES
    removedFeatures: tuple[str, ...] | None = defaults.DEFAULT_REMOVED_FEATURES
    binaryLabel: bool = defaults.DEFAULT_BINARY_LABEL
    parameters: dict | None = None

    def __post_init__(self):
        # Normaliza o contrato de um dataset sem inferir nome lógico a partir do arquivo.
        if not str(self.name).strip():
            raise ValueError("O nome lógico do dataset não pode ser vazio.")
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "targetColumn", str(self.targetColumn).strip())
        for field_name in ("selectedFeatures", "removedFeatures"):
            values = getattr(self, field_name)
            if values is not None:
                object.__setattr__(self, field_name, tuple(str(v).strip() for v in values))
        object.__setattr__(self, "parameters", deepcopy(dict(self.parameters or {})))


def deepMerge(base, override):
    # Combina recursivamente parâmetros mantendo os defaults que não foram sobrescritos.
    result = deepcopy(base)
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deepMerge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolveParameters(parameters=None):
    # Resolve os parâmetros do experimento sobre os defaults técnicos do projeto.
    return deepMerge(defaults.getDefaultParameters(), parameters or {})


def resolveDatasets(datasets, defaultParameters=None):
    # Converte o dicionário informado no notebook em contratos explícitos de dataset.
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("datasets deve ser um dicionário não vazio.")
    dataset_defaults = resolveParameters(defaultParameters)["dataset"]
    resolved = []
    for name, definition in datasets.items():
        if isinstance(definition, (str, Path)):
            definition = {"path": definition}
        definition = dict(definition)
        if "path" not in definition:
            raise ValueError(f"Dataset '{name}' não possui path.")
        resolved.append(DatasetConfig(
            name=name,
            path=definition["path"],
            targetColumn=definition.get("targetColumn", dataset_defaults["targetColumn"]),
            selectedFeatures=definition.get("selectedFeatures", dataset_defaults["selectedFeatures"]),
            removedFeatures=definition.get("removedFeatures", dataset_defaults["removedFeatures"]),
            binaryLabel=definition.get("binaryLabel", dataset_defaults["binaryLabel"]),
            parameters=definition.get("parameters", {}),
        ))
    return tuple(resolved)
