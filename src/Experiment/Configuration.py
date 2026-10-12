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
    normalLabels: tuple[str, ...] = defaults.DEFAULT_NORMAL_LABELS
    parameters: dict | None = None

    def __post_init__(self):
        # Normaliza o contrato de dataset usado por treinamento e otimização binários.
        if not str(self.name).strip():
            raise ValueError("The logical dataset name cannot be empty.")
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "targetColumn", str(self.targetColumn).strip())
        for field_name in ("selectedFeatures", "removedFeatures", "normalLabels"):
            values = getattr(self, field_name)
            if values is not None:
                object.__setattr__(self, field_name, tuple(str(value).strip() for value in values))
        object.__setattr__(self, "parameters", deepcopy(dict(self.parameters or {})))


def deepMerge(base, override):
    # Combina recursivamente dicionários sem modificar os objetos recebidos.
    result = deepcopy(base)
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deepMerge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def mergeParameters(base, override=None, task="anomaly"):
    # Mescla parâmetros preservando defaults corretos quando o modelo ou threshold muda de estratégia.
    resolved_task = str(task).strip().lower()
    override = deepcopy(dict(override or {}))
    result = deepMerge(base, override)
    model_override = dict(override.get("model", {}))
    if model_override:
        model_code = str(result["model"]["code"]).strip().upper()
        base_code = str(base["model"]["code"]).strip().upper()
        if "code" in model_override and model_code != base_code:
            model_defaults = defaults.getAnomalyModelDefaults(model_code) if resolved_task == "anomaly" else defaults.getClassifierDefaults(model_code)
            result["model"]["parameters"] = deepMerge(model_defaults, model_override.get("parameters", {}))
        elif "parameters" in model_override:
            result["model"]["parameters"] = deepMerge(base["model"].get("parameters", {}), model_override["parameters"])
        result["model"]["code"] = model_code
    if resolved_task == "anomaly" and "threshold" in override:
        threshold_override = dict(override["threshold"])
        threshold_name = str(result["threshold"]["name"]).strip().lower()
        base_name = str(base["threshold"]["name"]).strip().lower()
        if "name" in threshold_override and threshold_name != base_name:
            threshold_defaults = deepcopy(defaults.DEFAULT_DSPOT_PARAMETERS) if threshold_name == "dspot" else {"value": 0.5} if threshold_name == "fixed" else {}
            result["threshold"]["parameters"] = deepMerge(threshold_defaults, threshold_override.get("parameters", {}))
        elif "parameters" in threshold_override:
            result["threshold"]["parameters"] = deepMerge(base["threshold"].get("parameters", {}), threshold_override["parameters"])
        result["threshold"]["name"] = threshold_name
    return result


def resolveParameters(parameters=None, task="anomaly"):
    # Resolve os parâmetros informados sobre os defaults da tarefa binária escolhida.
    base = defaults.getDefaultParameters(task)
    return mergeParameters(base, parameters or {}, task=task)


def resolveDatasets(datasets, defaultParameters=None, task="anomaly"):
    # Converte o dicionário do notebook em contratos explícitos de dataset binário.
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("datasets must be a non-empty dictionary.")
    dataset_defaults = (defaultParameters or resolveParameters(task=task))["dataset"]
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
            normalLabels=definition.get("normalLabels", dataset_defaults["normalLabels"]),
            parameters=definition.get("parameters", {}),
        ))
    return tuple(resolved)
