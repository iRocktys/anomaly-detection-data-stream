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
    # Inicializa o executor preservando detecção como tarefa padrão e permitindo classificação binária explícita.
    def __init__(self, outputRoot=None, task="anomaly"):
        resolved_task = str(task).strip().lower()
        if resolved_task not in {"anomaly", "classification"}:
            raise ValueError("Task must be 'anomaly' or 'classification'.")
        self.task = resolved_task
        self.outputRoot = Path(outputRoot or defaults.DEFAULT_OUTPUT_ROOT)

    # Executa a mesma configuração experimental em todos os datasets explicitamente informados.
    def run(self, experimentName, datasets, parameters=None):
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
            print("Stream | " + " | ".join(
                f"{name}={float(metrics[name]):.4f}"
                for name in ("accuracy", "precision", "recall", "f1", "mcc") if name in metrics
            ))
        return {
            "experimentName": experiment_name,
            "task": self.task,
            "outputDirectory": str(experiment_output),
            "results": results,
            "summary": pd.DataFrame(records),
        }

    # Prepara uma stream binária e delega a execução ao pipeline correspondente à tarefa selecionada.
    def _runDataset(self, dataset, parameters, outputDirectory):
        if not dataset.path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset.path}")
        dataframe = pd.read_csv(dataset.path, low_memory=False)
        processor = DataStreamProcessor(
            logging=False,
            selected_features=dataset.selectedFeatures,
            removed_features=dataset.removedFeatures,
        )
        stream, target_names, feature_names, label_names = processor.create_stream(
            dataframe,
            target_label_col=dataset.targetColumn,
            binary_label=dataset.binaryLabel,
        )
        pipeline = self.createPipeline(
            task=self.task, stream=stream, datasetName=dataset.name, labelNames=label_names,
            parameters=parameters, outputDirectory=outputDirectory,
            resultManager=ResultManager(outputDirectory),
            generatePlots=bool(parameters["execution"]["generatePlots"]),
        )
        result = pipeline.run()
        result["featureRemovalReport"] = processor.getFeatureRemovalReport()
        result["featureNames"] = tuple(feature_names)
        result["targetNames"] = tuple(target_names)
        return result

    @staticmethod
    def createPipeline(task, stream, datasetName, labelNames, parameters,
                       outputDirectory, resultManager, generatePlots):
        # Montagem única usada pelo experimento normal e por cada trial da otimização.
        if task not in {"anomaly", "classification"}:
            raise ValueError("Task must be 'anomaly' or 'classification'.")
        preprocessing, execution, model = (parameters[name] for name in ("preprocessing", "execution", "model"))
        common = dict(
            stream=stream, datasetName=datasetName, modelCode=model["code"], labelNames=labelNames,
            modelParameters=model.get("parameters", {}),
            imputer=ImputerRegistry.create(preprocessing["imputer"], preprocessing.get("imputerParameters")),
            normalizer=NormalizerRegistry.create(preprocessing["normalizer"], preprocessing.get("normalizerParameters")),
            metricsWindowSize=int(execution["metricsWindowSize"]), outputPath=outputDirectory,
            seed=int(execution["seed"]), generatePlots=bool(generatePlots),
            initialWarmupSize=int(execution.get("initialWarmupSize", 0)),
            resultManager=resultManager, strictTrainingErrors=bool(execution.get("strictTrainingErrors", False)),
        )
        if task == "classification":
            ClassifierRegistry.normalizeCode(model["code"])
            return ClassificationPipeline(**common)
        threshold, training = parameters["threshold"], parameters["training"]
        return TrainingPipeline(
            **common, threshold=ThresholdRegistry.create(threshold["name"], threshold.get("parameters", {})),
            trainingStrategy=TrainingStrategyRegistry.create(training["strategy"], training.get("parameters")),
            scoreWindowSizes=tuple(threshold.get("scoreWindowSizes", ())),
            normalClassIndex=defaults.DEFAULT_NORMAL_CLASS_INDEX,
            thresholdScoreSource=threshold.get("scoreSource", "raw"),
        )
