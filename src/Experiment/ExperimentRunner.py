from pathlib import Path

import pandas as pd

import ProjectDefaults as defaults
from src.Anomaly.Thresholds.ThresholdRegistry import ThresholdRegistry
from src.Data.Processor import DataStreamProcessor
from src.Data.Registries import ImputerRegistry, NormalizerRegistry
from src.Experiment.Configuration import deepMerge, resolveDatasets, resolveParameters
from src.Pipeline.ResultManager import ResultManager
from src.Pipeline.TrainingPipeline import TrainingPipeline
from src.Training.StrategyRegistry import TrainingStrategyRegistry


class ExperimentRunner:
    def __init__(self, outputRoot=None):
        # Inicializa o executor genérico sem assumir datasets, cenários ou nome de experimento.
        self.outputRoot = Path(outputRoot or defaults.DEFAULT_OUTPUT_ROOT)

    def run(self, experimentName, datasets, parameters=None):
        # Executa o mesmo perfil experimental para todos os datasets explicitamente informados.
        experiment_name = str(experimentName).strip()
        if not experiment_name:
            raise ValueError("experimentName deve ser informado explicitamente.")
        base_parameters = resolveParameters(parameters)
        dataset_configs = resolveDatasets(datasets, base_parameters)
        experiment_output = self.outputRoot / experiment_name
        results = {}
        records = []
        for index, dataset in enumerate(dataset_configs, start=1):
            effective = deepMerge(base_parameters, dataset.parameters)
            print("\n" + "=" * 80)
            print(f"[{index:02d}/{len(dataset_configs):02d}] Experimento={experiment_name} | Dataset={dataset.name}")
            print(f"Arquivo: {dataset.path}")
            result = self._runDataset(dataset, effective, experiment_output)
            results[dataset.name] = result
            metrics = result["streamMetricsFrame"].iloc[0].to_dict()
            records.append({"experiment": experiment_name, "dataset": dataset.name, **metrics})
        return {
            "experimentName": experiment_name,
            "outputDirectory": str(experiment_output),
            "results": results,
            "summary": pd.DataFrame(records),
        }

    def _runDataset(self, dataset, parameters, outputDirectory):
        # Prepara o stream e delega treinamento, threshold, métricas e persistência ao pipeline.
        dataframe = pd.read_csv(dataset.path, low_memory=False)
        processor = DataStreamProcessor(
            logging=True,
            selected_features=dataset.selectedFeatures,
            removed_features=dataset.removedFeatures,
        )
        stream, target_names, feature_names, label_names = processor.create_stream(
            dataframe,
            target_label_col=dataset.targetColumn,
            binary_label=dataset.binaryLabel,
        )
        preprocessing = parameters["preprocessing"]
        training = parameters["training"]
        threshold_config = parameters["threshold"]
        execution = parameters["execution"]
        model = parameters["model"]
        threshold = ThresholdRegistry.create(
            threshold_config["name"],
            threshold_config.get("parameters", {}),
        )
        pipeline = TrainingPipeline(
            stream=stream,
            datasetName=dataset.name,
            modelCode=model["code"],
            threshold=threshold,
            labelNames=label_names,
            modelParameters=model.get("parameters", {}),
            imputer=ImputerRegistry.create(preprocessing["imputer"], preprocessing.get("imputerParameters")),
            normalizer=NormalizerRegistry.create(preprocessing["normalizer"], preprocessing.get("normalizerParameters")),
            trainingStrategy=TrainingStrategyRegistry.create(training["strategy"], training.get("parameters")),
            scoreWindowSizes=tuple(threshold_config.get("scoreWindowSizes", ())),
            metricsWindowSize=int(execution["metricsWindowSize"]),
            outputPath=outputDirectory,
            normalClassIndex=defaults.DEFAULT_NORMAL_CLASS_INDEX,
            seed=int(execution["seed"]),
            generatePlots=bool(execution["generatePlots"]),
            initialWarmupSize=int(execution["initialWarmupSize"]),
            thresholdScoreSource=threshold_config.get("scoreSource", "raw"),
            resultManager=ResultManager(outputDirectory),
            strictTrainingErrors=bool(execution.get("strictTrainingErrors", False)),
        )
        result = pipeline.run()
        result["featureNames"] = tuple(feature_names)
        result["targetNames"] = tuple(target_names)
        return result
