from pathlib import Path

import pandas as pd

import ProjectDefaults as defaults
from src.Data.DatasetLoader import DatasetLoader
from src.Data.Registries import ImputerRegistry, NormalizerRegistry
from src.Experiment.Configuration import mergeParameters, resolveDatasets, resolveParameters
from src.Experiment.Task import normalizeTask
from src.Pipeline.AnomalyDetectionPipeline import AnomalyDetectionPipeline
from src.Pipeline.ClassificationPipeline import ClassificationPipeline
from src.Pipeline.ResultManager import ResultManager
from src.Thresholds.ThresholdRegistry import ThresholdRegistry
from src.Training.StrategyRegistry import TrainingStrategyRegistry


class ExperimentRunner:
    def __init__(self, task="anomaly", outputRoot=None):
        # Inicializa o executor da tarefa binária e centraliza a raiz dos resultados experimentais.
        self.task = normalizeTask(task)
        task_directory = "AnomalyDetection" if self.task == "anomaly" else "Classification"
        self.outputRoot = Path(outputRoot or (defaults.DEFAULT_EXPERIMENT_OUTPUT_ROOT / task_directory))

    def run(self, experimentName, datasets, parameters=None):
        # Executa a mesma configuração sobre todos os datasets explicitamente fornecidos pelo notebook.
        experiment_name = str(experimentName).strip()
        if not experiment_name:
            raise ValueError("experimentName must be provided explicitly.")
        base_parameters = resolveParameters(parameters, task=self.task)
        dataset_configs = resolveDatasets(datasets, base_parameters, task=self.task)
        experiment_output = self.outputRoot / experiment_name
        results = {}
        summary_rows = []
        for index, dataset_config in enumerate(dataset_configs, start=1):
            effective_parameters = mergeParameters(base_parameters, dataset_config.parameters, task=self.task)
            print("\n" + "=" * 80)
            print(f"[{index:02d}/{len(dataset_configs):02d}] Experimento={experiment_name} | Dataset={dataset_config.name}")
            print(f"Arquivo: {dataset_config.path}")
            result = self._runDataset(dataset_config, effective_parameters, experiment_output)
            results[dataset_config.name] = result
            metrics = result["streamMetricsFrame"].iloc[0].to_dict()
            summary_rows.append({"experiment": experiment_name, "dataset": dataset_config.name, **metrics})
        return {
            "task": self.task,
            "experimentName": experiment_name,
            "outputDirectory": str(experiment_output),
            "results": results,
            "summary": pd.DataFrame(summary_rows),
        }

    def _runDataset(self, datasetConfig, parameters, outputDirectory):
        # Prepara o dataset uma única vez e delega a execução ao pipeline específico da tarefa.
        prepared_dataset = DatasetLoader.load(datasetConfig, logging=True)
        preprocessing = parameters["preprocessing"]
        execution = parameters["execution"]
        model = parameters["model"]
        common = {
            "stream": prepared_dataset.stream,
            "datasetName": prepared_dataset.name,
            "modelCode": model["code"],
            "labelNames": prepared_dataset.labelNames,
            "modelParameters": model.get("parameters", {}),
            "imputer": ImputerRegistry.create(preprocessing["imputer"], preprocessing.get("imputerParameters")),
            "normalizer": NormalizerRegistry.create(preprocessing["normalizer"], preprocessing.get("normalizerParameters")),
            "metricsWindowSize": int(execution["metricsWindowSize"]),
            "seed": int(execution["seed"]),
            "generatePlots": bool(execution["generatePlots"]),
            "initialWarmupSize": int(execution["initialWarmupSize"]),
            "resultManager": ResultManager(outputDirectory),
            "strictTrainingErrors": bool(execution.get("strictTrainingErrors", False)),
        }
        if self.task == "anomaly":
            threshold_config = parameters["threshold"]
            training_config = parameters["training"]
            pipeline = AnomalyDetectionPipeline(
                **common,
                threshold=ThresholdRegistry.create(threshold_config["name"], threshold_config.get("parameters", {})),
                trainingStrategy=TrainingStrategyRegistry.create(training_config["strategy"], training_config.get("parameters")),
                scoreWindowSizes=tuple(threshold_config.get("scoreWindowSizes", ())),
                thresholdScoreSource=threshold_config.get("scoreSource", "raw"),
            )
        else:
            pipeline = ClassificationPipeline(**common)
        result = pipeline.run()
        result["featureNames"] = prepared_dataset.featureNames
        result["binaryTargetNames"] = ("BENIGN", "ATTACK")
        return result

