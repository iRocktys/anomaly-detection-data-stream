import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.Anomaly.Models import ModelRegistry
from src.Classification.Models import ClassifierRegistry
from src.Data.DatasetLoader import DatasetLoader
from src.Data.Registries import ImputerRegistry, NormalizerRegistry
from src.Experiment.Configuration import deepMerge, mergeParameters
from src.Optimization.AifSearchSpace import AifSearchSpace
from src.Optimization.ClassifierSearchSpace import ClassifierSearchSpace
from src.Optimization.DspotSearchSpace import DspotSearchSpace
from src.Optimization.OptimizationConfig import OptimizationConfig
from src.Pipeline.AnomalyDetectionPipeline import AnomalyDetectionPipeline
from src.Pipeline.ClassificationPipeline import ClassificationPipeline
from src.Pipeline.MetricsResultManager import MetricsResultManager
from src.Pipeline.ResultManager import ResultManager
from src.Thresholds.Incremental.DspotThreshold import DspotCalibrationError
from src.Thresholds.ThresholdRegistry import ThresholdRegistry
from src.Training.StrategyRegistry import TrainingStrategyRegistry


class OptunaStreamOptimizer:
    def __init__(self, config, dspotSearchSpace=None, aifSearchSpace=None, classifierSearchSpace=None):
        # Inicializa o motor Optuna comum às tarefas binárias sem acoplar persistência aos trials intermediários.
        if not isinstance(config, OptimizationConfig):
            raise TypeError("config must be an OptimizationConfig.")
        self.config = config
        self.dspotSearchSpace = dspotSearchSpace or DspotSearchSpace(config.dspotSearchSpace)
        self.aifSearchSpace = aifSearchSpace or AifSearchSpace(config.aifSearchSpace)
        self.classifierSearchSpace = classifierSearchSpace or ClassifierSearchSpace()

    def run(self):
        # Otimiza cada dataset, salva todos os trials e reexecuta somente a melhor configuração com artefatos completos.
        optuna = self.loadOptuna()
        task_name = "AnomalyDetection" if self.config.task == "anomaly" else "Classification"
        output_directory = self.config.outputRoot / self.normalizeName(self.config.experimentName) / task_name
        output_directory.mkdir(parents=True, exist_ok=True)
        best_results = {}
        summary_rows = []
        for dataset_config in self.config.datasetConfigs:
            prepared_dataset = DatasetLoader.load(dataset_config, logging=False)
            base_parameters = mergeParameters(self.config.effectiveParameters, dataset_config.parameters, task=self.config.task)
            model_code = self.getModelCode(base_parameters)
            print("\n" + "=" * 80)
            print(f"Otimização: {prepared_dataset.name} | Tarefa: {task_name} | Modelo: {model_code}")
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(self.config.optunaSeed)))
            self.optimizeStudy(study, prepared_dataset, base_parameters)
            dataset_directory = output_directory / model_code / self.normalizeName(prepared_dataset.name)
            dataset_directory.mkdir(parents=True, exist_ok=True)
            trials_path = dataset_directory / "trials.csv"
            study.trials_dataframe().to_csv(trials_path, index=False)
            best_parameters = dict(study.best_trial.user_attrs["resolvedParameters"])
            best_parameters_path = dataset_directory / "best_parameters.json"
            best_parameters_path.write_text(json.dumps(best_parameters, indent=2, ensure_ascii=False), encoding="utf-8")
            best_directory = dataset_directory / "Best"
            final_result = self.runPipeline(
                preparedDataset=prepared_dataset,
                parameters=best_parameters,
                resultManager=ResultManager(fixedRunDirectory=best_directory, overwriteFixedDirectory=True),
            )
            reexecution_f1 = float(final_result["streamMetricsFrame"].iloc[0]["f1"])
            best_results[prepared_dataset.name] = {
                "bestTrialNumber": int(study.best_trial.number),
                "optimizationF1": float(study.best_value),
                "reexecutionF1": reexecution_f1,
                "trialsPath": str(trials_path),
                "bestParametersPath": str(best_parameters_path),
                "bestDirectory": str(best_directory),
                "result": final_result,
            }
            summary_rows.append({
                "dataset": prepared_dataset.name,
                "task": self.config.task,
                "model": model_code,
                "bestTrialNumber": int(study.best_trial.number),
                "optimizationF1": float(study.best_value),
                "reexecutionF1": reexecution_f1,
                "trialsPath": str(trials_path),
                "bestParametersPath": str(best_parameters_path),
                "bestDirectory": str(best_directory),
            })
        summary = pd.DataFrame(summary_rows)
        summary_path = output_directory / "best_results.csv"
        summary.to_csv(summary_path, index=False)
        return {
            "task": self.config.task,
            "experimentName": self.config.experimentName,
            "outputDirectory": str(output_directory),
            "bestResults": best_results,
            "summary": summary,
            "summaryPath": str(summary_path),
        }

    def optimizeStudy(self, study, preparedDataset, baseParameters):
        # Continua a busca até completar a quantidade solicitada de trials válidos ou atingir o limite de tentativas.
        optuna = self.loadOptuna()
        target = int(self.config.nTrials)
        maximum_attempts = target * int(self.config.maxTrialAttemptsMultiplier)
        attempts = 0
        while self.countComplete(study, optuna) < target and attempts < maximum_attempts:
            study.optimize(lambda trial: self.objective(trial, preparedDataset, baseParameters), n_trials=1, catch=())
            attempts += 1
        completed = self.countComplete(study, optuna)
        if completed == 0:
            raise RuntimeError("No valid trial was completed.")
        if completed < target:
            print(f"Aviso: foram concluídos {completed} de {target} trials válidos após {attempts} tentativas.")

    def objective(self, trial, preparedDataset, baseParameters):
        # Constrói uma configuração candidata, executa a stream em memória e usa somente F1 como função objetivo.
        optuna = self.loadOptuna()
        try:
            resolved = self.suggestParameters(trial, baseParameters)
            trial.set_user_attr("resolvedParameters", resolved)
            result = self.runPipeline(preparedDataset, resolved, MetricsResultManager())
            value = float(result.streamMetrics["f1"])
            if not pd.notna(value):
                raise ValueError("Invalid F1 value in trial.")
            return value
        except (DspotCalibrationError, ValueError) as error:
            raise optuna.TrialPruned(str(error)) from error

    def suggestParameters(self, trial, baseParameters):
        # Aplica somente os espaços específicos da tarefa sobre a configuração base já resolvida.
        resolved = deepMerge(baseParameters, {})
        model_code = self.getModelCode(resolved)
        base_model = dict(resolved["model"].get("parameters", {}))
        fixed_model = dict(self.config.fixedModelParameters)
        if self.config.task == "anomaly":
            threshold_name = str(resolved["threshold"]["name"]).strip().lower()
            if threshold_name == "dspot":
                resolved["threshold"] = self.dspotSearchSpace.suggest(trial)
            if self.config.optimizeModelParameters and model_code == "AIF":
                model_warmup = int(resolved["execution"]["initialWarmupSize"]) - int(self.config.dspotSearchSpace.calibrationWindow)
                suggested_model = self.aifSearchSpace.suggest(trial, fixed_model, modelWarmupSize=model_warmup)
                resolved["model"]["parameters"] = deepMerge(base_model, suggested_model)
            else:
                resolved["model"]["parameters"] = deepMerge(base_model, fixed_model)
        else:
            if self.config.optimizeModelParameters:
                suggested_model = self.classifierSearchSpace.suggest(trial, model_code, fixed_model)
                resolved["model"]["parameters"] = deepMerge(base_model, suggested_model)
            else:
                resolved["model"]["parameters"] = deepMerge(base_model, fixed_model)
        return resolved

    def runPipeline(self, preparedDataset, parameters, resultManager):
        # Cria componentes novos para cada execução e delega ao pipeline binário correspondente à tarefa.
        preprocessing = parameters["preprocessing"]
        execution = parameters["execution"]
        model = parameters["model"]
        common = {
            "stream": preparedDataset.stream,
            "datasetName": preparedDataset.name,
            "modelCode": model["code"],
            "labelNames": preparedDataset.labelNames,
            "modelParameters": model.get("parameters", {}),
            "imputer": ImputerRegistry.create(preprocessing["imputer"], preprocessing.get("imputerParameters")),
            "normalizer": NormalizerRegistry.create(preprocessing["normalizer"], preprocessing.get("normalizerParameters")),
            "metricsWindowSize": int(execution["metricsWindowSize"]),
            "seed": int(execution["seed"]),
            "generatePlots": bool(execution.get("generatePlots", True)) if isinstance(resultManager, ResultManager) else False,
            "initialWarmupSize": int(execution["initialWarmupSize"]),
            "resultManager": resultManager,
            "strictTrainingErrors": bool(execution.get("strictTrainingErrors", False)),
        }
        if self.config.task == "anomaly":
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
        return pipeline.run()

    def getModelCode(self, parameters):
        # Normaliza o código do modelo com o registry apropriado à tarefa da otimização.
        code = parameters["model"]["code"]
        if self.config.task == "anomaly":
            return ModelRegistry.normalizeCode(code)
        return ClassifierRegistry.normalizeCode(code)

    @staticmethod
    def countComplete(study, optuna):
        # Conta apenas trials realmente concluídos para não confundir calibrações podadas com avaliações válidas.
        return sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)

    @staticmethod
    def normalizeName(value):
        # Converte um nome lógico em um componente ASCII seguro para caminhos de saída.
        ascii_name = unicodedata.normalize("NFKD", str(value).strip()).encode("ascii", "ignore").decode("ascii")
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_")
        if not safe_name:
            raise ValueError("Output path name cannot be empty.")
        return safe_name

    @staticmethod
    def loadOptuna():
        # Importa Optuna somente quando a rotina de otimização é efetivamente utilizada.
        try:
            import optuna
        except ImportError as error:
            raise ImportError("Optuna is not installed in the environment.") from error
        return optuna
