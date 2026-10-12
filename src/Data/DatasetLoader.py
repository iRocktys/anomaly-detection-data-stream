from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.Data.Processor import DataStreamProcessor


@dataclass(frozen=True)
class PreparedDataset:
    name: str
    path: Path
    stream: object
    featureNames: tuple[str, ...]
    labelNames: tuple[str, ...]
    binaryLabels: np.ndarray
    totalInstances: int
    attackInstances: int
    attackRatioPercent: float
    parameters: dict


class DatasetLoader:
    @staticmethod
    def load(dataset_config, logging=True):
        # Carrega um CSV uma única vez e o converte no contrato binário compartilhado pelos pipelines.
        dataframe = pd.read_csv(dataset_config.path, low_memory=False)
        processor = DataStreamProcessor(
            logging=logging,
            selected_features=dataset_config.selectedFeatures,
            removed_features=dataset_config.removedFeatures,
            normal_labels=dataset_config.normalLabels,
        )
        stream, feature_names, label_names, binary_labels = processor.create_stream(
            dataframe,
            target_label_col=dataset_config.targetColumn,
        )
        total_instances = int(len(binary_labels))
        attack_instances = int(np.sum(binary_labels))
        attack_ratio = float((attack_instances / total_instances) * 100.0) if total_instances else 0.0
        return PreparedDataset(
            name=dataset_config.name,
            path=Path(dataset_config.path),
            stream=stream,
            featureNames=tuple(feature_names),
            labelNames=tuple(label_names),
            binaryLabels=binary_labels,
            totalInstances=total_instances,
            attackInstances=attack_instances,
            attackRatioPercent=attack_ratio,
            parameters=dict(dataset_config.parameters or {}),
        )
