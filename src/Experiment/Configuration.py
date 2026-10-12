from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

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
        # Normaliza o contrato de dataset mantendo somente configurações usadas no experimento binário.
        if not str(self.name).strip():
            raise ValueError("Dataset name cannot be empty.")
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "targetColumn", str(self.targetColumn).strip())
        for field_name in ("selectedFeatures", "removedFeatures"):
            values = getattr(self, field_name)
            if values is not None:
                object.__setattr__(self, field_name, tuple(str(value).strip() for value in values))
        if not bool(self.binaryLabel):
            raise ValueError("Only binary experiments are supported; binaryLabel must remain True.")
        object.__setattr__(self, "binaryLabel", True)
        object.__setattr__(self, "parameters", deepcopy(dict(self.parameters or {})))


def deepMerge(base, override):
    # Combina recursivamente parâmetros mantendo os valores base não sobrescritos.
    result = deepcopy(base)
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deepMerge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def mergeParameters(base, override=None, task="anomaly"):
    # Combina configurações sem herdar parâmetros incompatíveis quando modelo ou threshold são trocados.
    override = dict(override or {})
    result = deepMerge(base, override)
    resolved_task = str(task).strip().lower()
    model_override = dict(override.get("model", {}))
    if "code" in model_override:
        old_code = str(base.get("model", {}).get("code", "")).strip().upper()
        new_code = str(model_override["code"]).strip().upper()
        if new_code != old_code:
            if resolved_task == "classification":
                model_parameters = defaults.getClassifierDefaults(new_code)
            else:
                model_parameters = defaults.getModelDefaults(new_code)
            model_parameters.update(dict(model_override.get("parameters", {})))
            result["model"]["parameters"] = model_parameters
    if resolved_task == "anomaly":
        threshold_override = dict(override.get("threshold", {}))
        if "name" in threshold_override:
            old_name = str(base.get("threshold", {}).get("name", "")).strip().lower()
            new_name = str(threshold_override["name"]).strip().lower()
            if new_name != old_name:
                threshold_parameters = deepcopy(defaults.DEFAULT_DSPOT_PARAMETERS) if new_name == "dspot" else {}
                threshold_parameters.update(dict(threshold_override.get("parameters", {})))
                result["threshold"]["parameters"] = threshold_parameters
    return result


def resolveParameters(parameters=None, task="anomaly"):
    # Resolve parâmetros sobre os defaults da tarefa preservando o contrato original dos notebooks.
    resolved_task = str(task).strip().lower()
    if resolved_task == "anomaly":
        base = defaults.getDefaultParameters()
    elif resolved_task == "classification":
        base = defaults.getClassificationParameters()
    else:
        raise ValueError("Task must be 'anomaly' or 'classification'.")
    return mergeParameters(base, parameters or {}, task=resolved_task)


def resolveDatasets(datasets, defaultParameters=None, task="anomaly"):
    # Converte o dicionário informado no notebook em contratos explícitos de datasets binários.
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("datasets must be a non-empty dictionary.")
    dataset_defaults = resolveParameters(defaultParameters, task=task)["dataset"]
    resolved = []
    for name, definition in datasets.items():
        if isinstance(definition, (str, Path)):
            definition = {"path": definition}
        definition = dict(definition)
        if "path" not in definition:
            raise ValueError(f"Dataset '{name}' does not define a path.")
        resolved.append(DatasetConfig(
            name=name,
            path=definition["path"],
            targetColumn=definition.get("targetColumn", dataset_defaults["targetColumn"]),
            selectedFeatures=definition.get("selectedFeatures", dataset_defaults["selectedFeatures"]),
            removedFeatures=definition.get("removedFeatures", dataset_defaults["removedFeatures"]),
            binaryLabel=definition.get("binaryLabel", dataset_defaults.get("binaryLabel", True)),
            parameters=definition.get("parameters", {}),
        ))
    return tuple(resolved)
