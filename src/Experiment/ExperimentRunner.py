from pathlib import Path

import pandas as pd

import ProjectDefaults as defaults
from src.Classification.Models import ClassifierRegistry
from src.Data.Processor import DataStreamProcessor
from src.Data.Registries import ImputerRegistry, NormalizerRegistry
from src.Experiment.Configuration import mergeParameters, resolveDatasets, resolveParameters
from src.Pipeline.ClassificationPipeline import ClassificationPipeline
from src.Pipeline.ResultManager import ResultManager
from src.Pipeline.TrainingPipeline import TrainingPipeline
from src.Thresholds.ThresholdRegistry import ThresholdRegistry
from src.Training.StrategyRegistry import TrainingStrategyRegistry


class ExperimentRunner:
    def __init__(self, outputRoot=None, task="anomaly"):
        # Inicializa o executor preservando detecção como tarefa padrão e permitindo classificação binária explícita.
        resolved_task = str(task).strip().lower()
        if resolved_task not in {"anomaly", "classification"}:
            raise ValueError("Task must be 'anomaly' or 'classification'.")
        self.task = resolved_task
        self.outputRoot = Path(outputRoot or defaults.DEFAULT_OUTPUT_ROOT)

    def run(self, experimentName, datasets, parameters=None):
        # Executa a mesma configuração experimental em todos os datasets explicitamente informados.
        experiment_name = str(experimentName).strip()
        if not experiment_name:
            raise ValueError("experimentName must be provided explicitly.")
        base_parameters = resolveParameters(parameters, task=self.task)
        dataset_configs = resolveDatasets(datasets, base_parameters, task=self.task)
        experiment_output = self.outputRoot / experiment_name
        results = {}
        records = []
        for index, dataset in enumerate(dataset_configs, start=1):
            effective = mergeParameters(base_parameters, dataset.parameters, task=self.task)
            print("\n" + "=" * 80)
            print(f"[{index:02d}/{len(dataset_configs):02d}] Experimento={experiment_name} | Dataset={dataset.name}")
            print(f"Arquivo: {dataset.path}")
            result = self._runDataset(dataset, effective, experiment_output)
            results[dataset.name] = result
            metrics = result["streamMetricsFrame"].iloc[0].to_dict()
            records.append({"experiment": experiment_name, "dataset": dataset.name, **metrics})
        return {
            "experimentName": experiment_name,
            "task": self.task,
            "outputDirectory": str(experiment_output),
            "results": results,
            "summary": pd.DataFrame(records),
        }

    def _runDataset(self, dataset, parameters, outputDirectory):
        # Prepara uma stream binária e delega a execução ao pipeline correspondente à tarefa selecionada.
        if not dataset.path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset.path}")
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
        execution = parameters["execution"]
        model = parameters["model"]
        imputer = ImputerRegistry.create(preprocessing["imputer"], preprocessing.get("imputerParameters"))
        normalizer = NormalizerRegistry.create(preprocessing["normalizer"], preprocessing.get("normalizerParameters"))
        result_manager = ResultManager(outputDirectory)
        if self.task == "classification":
            ClassifierRegistry.normalizeCode(model["code"])
            pipeline = ClassificationPipeline(
                stream=stream,
                datasetName=dataset.name,
                modelCode=model["code"],
                labelNames=label_names,
                modelParameters=model.get("parameters", {}),
                imputer=imputer,
                normalizer=normalizer,
                metricsWindowSize=int(execution["metricsWindowSize"]),
                outputPath=outputDirectory,
                seed=int(execution["seed"]),
                generatePlots=bool(execution["generatePlots"]),
                initialWarmupSize=int(execution.get("initialWarmupSize", 0)),
                resultManager=result_manager,
                strictTrainingErrors=bool(execution.get("strictTrainingErrors", False)),
            )
        else:
            threshold_config = parameters["threshold"]
            training = parameters["training"]
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
                imputer=imputer,
                normalizer=normalizer,
                trainingStrategy=TrainingStrategyRegistry.create(training["strategy"], training.get("parameters")),
                scoreWindowSizes=tuple(threshold_config.get("scoreWindowSizes", ())),
                metricsWindowSize=int(execution["metricsWindowSize"]),
                outputPath=outputDirectory,
                normalClassIndex=defaults.DEFAULT_NORMAL_CLASS_INDEX,
                seed=int(execution["seed"]),
                generatePlots=bool(execution["generatePlots"]),
                initialWarmupSize=int(execution["initialWarmupSize"]),
                thresholdScoreSource=threshold_config.get("scoreSource", "raw"),
                resultManager=result_manager,
                strictTrainingErrors=bool(execution.get("strictTrainingErrors", False)),
            )
        result = pipeline.run()
        result["featureNames"] = tuple(feature_names)
        result["targetNames"] = tuple(target_names)
        return result
