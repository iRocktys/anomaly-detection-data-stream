import json
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.Anomaly.Models import ModelRegistry
from src.Classification.Models import ClassifierRegistry
from src.Data.Processor import DataStreamProcessor
from src.Experiment.Configuration import deepMerge, mergeParameters
from src.Experiment.ExperimentRunner import ExperimentRunner
from src.Optimization.AifSearchSpace import AifSearchSpace
from src.Optimization.ClassifierSearchSpace import ClassifierSearchSpace
from src.Optimization.DspotSearchSpace import DspotSearchSpace
from src.Optimization.OptimizationConfig import OptimizationConfig, PreparedDataset
from src.Pipeline.MetricsResultManager import MetricsResultManager
from src.Pipeline.ResultManager import ResultManager
from src.Thresholds.Incremental.DspotThreshold import DspotCalibrationError


class OptunaStreamOptimizer:
    # Inicializa o otimizador preservando os argumentos originais e acrescentando suporte opcional a classificadores.
    def __init__(self, config, searchSpace=None, modelSearchSpace=None, classifierSearchSpace=None,
                 showFeatureRemovalReport=False):
        if not isinstance(config, OptimizationConfig):
            raise TypeError("config must be an OptimizationConfig.")
        if not isinstance(showFeatureRemovalReport, bool):
            raise ValueError("showFeatureRemovalReport deve ser booleano.")
        self.showFeatureRemovalReport = showFeatureRemovalReport
        self.featureRemovalReports = {}
        self.config = config
        self.parameters = config.effectiveParameters
        self.searchSpace = searchSpace or DspotSearchSpace(config.dspotSearchSpace)
        self.modelSearchSpace = modelSearchSpace or AifSearchSpace(config.aifSearchSpace)
        self.classifierSearchSpace = classifierSearchSpace or ClassifierSearchSpace()

    # Otimiza cada dataset, salva os trials e reexecuta somente a melhor configuração com artefatos completos.
    def run(self):
        optuna = self._loadOptuna()
        self.featureRemovalReports = {}
        prepared_datasets = [self._prepareDataset(dataset) for dataset in self.config.datasetConfigs]
        best_results = {}
        summary_rows = []
        for dataset in prepared_datasets:
            model_code = self._getModelCode(dataset.parameters)
            output_directory = self.config.outputRoot / self.config.experimentName / model_code
            dataset_directory = output_directory / self._normalizeName(dataset.name)
            dataset_directory.mkdir(parents=True, exist_ok=True)
            print("\n" + "=" * 80)
            print(f"Otimização: {dataset.name} | Modelo: {model_code} | Tarefa: {self.config.task}")
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=int(self.config.optunaSeed)),
            )
            self._optimizeStudy(study, dataset)
            trials_path = dataset_directory / "trials.csv"
            study.trials_dataframe().to_csv(trials_path, index=False)
            best_parameters = dict(study.best_trial.user_attrs["resolvedParameters"])
            best_parameters_path = dataset_directory / "best_parameters.json"
            best_parameters_path.write_text(
                json.dumps(self._jsonSafe(best_parameters), indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
            best_directory = dataset_directory / "best"
            final_result = self._createPipeline(
                dataset=dataset,
                parameters=best_parameters,
                resultManager=ResultManager(
                    outputPath=output_directory,
                    fixedRunDirectory=best_directory,
                    overwriteFixedDirectory=True,
                ),
                generatePlots=True,
            ).run()
            reexecution_f1 = float(final_result["streamMetricsFrame"].iloc[0]["f1"])
            best_results[dataset.name] = {
                "bestTrialNumber": int(study.best_trial.number),
                "optimizationF1": float(study.best_value),
                "reexecutionF1": reexecution_f1,
                "trialsPath": str(trials_path),
                "parametersPath": str(best_parameters_path),
                "bestDirectory": str(best_directory),
                "parameters": best_parameters,
                "result": final_result,
            }
            summary_rows.append({
                "dataset": dataset.name,
                "model": model_code,
                "task": self.config.task,
                "bestTrialNumber": int(study.best_trial.number),
                "optimizationF1": float(study.best_value),
                "reexecutionF1": reexecution_f1,
                "trialsPath": str(trials_path),
                "parametersPath": str(best_parameters_path),
                "bestDirectory": str(best_directory),
            })
            print(f"Melhor F1 da otimização: {float(study.best_value):.6f}")
            print(f"F1 da reexecução final: {reexecution_f1:.6f}")
        summary = pd.DataFrame(summary_rows)
        summary_path = self.config.outputRoot / self.config.experimentName / "best_results.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        return {
            "experimentName": self.config.experimentName,
            "task": self.config.task,
            "outputDirectory": str(summary_path.parent),
            "bestResults": best_results,
            "summary": summary,
            "summaryPath": str(summary_path),
            "featureRemovalReports": self.featureRemovalReports,
        }

    # Continua sugerindo configurações até completar a quantidade solicitada de trials válidos.
    def _optimizeStudy(self, study, dataset):
        target_complete = int(self.config.nTrials)
        maximum_attempts = target_complete * int(self.config.maxTrialAttemptsMultiplier)
        attempts = 0
        while self._countComplete(study) < target_complete and attempts < maximum_attempts:
            study.optimize(lambda trial: self._objective(trial, dataset), n_trials=1, gc_after_trial=True)
            attempts += 1
        complete_trials = self._countComplete(study)
        print(f"{dataset.name}: {complete_trials}/{target_complete} trials válidos em {attempts} tentativas.")
        if complete_trials == 0:
            raise RuntimeError(f"No valid trial was completed for '{dataset.name}'.")
        if complete_trials < target_complete:
            print("Aviso: o limite de tentativas foi atingido e o melhor resultado disponível será utilizado.")

    # Conta somente trials concluídos com F1 finito.
    def _countComplete(self, study):
        return sum(
            1
            for trial in study.trials
            if getattr(trial.state, "name", str(trial.state)) == "COMPLETE"
            and trial.value is not None
            and math.isfinite(float(trial.value))
        )

    # Executa um trial somente em memória e retorna o F1 binário como função objetivo.
    def _objective(self, trial, dataset):
        optuna = self._loadOptuna()
        try:
            parameters = self._suggestParameters(trial, dataset.parameters)
            trial.set_user_attr("resolvedParameters", self._jsonSafe(parameters))
            result = self._createPipeline(
                dataset=dataset,
                parameters=parameters,
                resultManager=MetricsResultManager(),
                generatePlots=False,
            ).run()
            f1 = float(result.streamMetrics["f1"])
            if not math.isfinite(f1):
                raise ValueError("The trial produced a non-finite F1 value.")
            return f1
        except (DspotCalibrationError, ValueError) as error:
            raise optuna.TrialPruned(str(error)) from error

    # Sugere apenas os parâmetros pertencentes à tarefa e mantém intacta a configuração restante.
    def _suggestParameters(self, trial, baseParameters):
        parameters = deepMerge(baseParameters, {})
        model_code = self._getModelCode(parameters)
        base_model = dict(parameters["model"].get("parameters", {}))
        fixed_model = dict(self.config.fixedModelParameters)
        if self.config.task == "classification":
            if self.config.optimizeModelParameters:
                suggested_model = self.classifierSearchSpace.suggest(trial, model_code, fixed_model)
                parameters["model"]["parameters"] = deepMerge(base_model, suggested_model)
            else:
                parameters["model"]["parameters"] = deepMerge(base_model, fixed_model)
            return parameters
        threshold_config = parameters["threshold"]
        if str(threshold_config["name"]).strip().lower() == "dspot":
            configuration = self.searchSpace.suggest(
                trial,
                imputerName=parameters["preprocessing"]["imputer"],
            )
            parameters["threshold"] = {
                "name": "dspot",
                "scoreSource": configuration.thresholdScoreSource,
                "scoreWindowSizes": list(configuration.scoreWindowSizes),
                "parameters": {
                    "driftDepth": int(configuration.driftDepth),
                    "calibrationSize": int(configuration.calibrationSize),
                    "initialQuantile": float(configuration.initialQuantile),
                    "risk": float(configuration.risk),
                    "refitEvery": int(configuration.refitEvery),
                    "optimizationStarts": int(configuration.optimizationStarts),
                    "tolerance": float(configuration.tolerance),
                },
            }
            calibration_window = configuration.calibrationWindow
        else:
            calibration_window = 0
        if self.config.optimizeModelParameters and model_code == "AIF":
            model_warmup = int(parameters["execution"]["initialWarmupSize"]) - int(calibration_window)
            suggested_model = self.modelSearchSpace.suggest(
                trial,
                fixedParameters=fixed_model,
                modelWarmupSize=model_warmup,
            )
            parameters["model"]["parameters"] = deepMerge(base_model, suggested_model)
        else:
            parameters["model"]["parameters"] = deepMerge(base_model, fixed_model)
        return parameters

    # Carrega o dataset uma única vez e mantém a stream reiniciável entre os trials.
    def _prepareDataset(self, dataset):
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
        self.featureRemovalReports[dataset.name] = processor.getFeatureRemovalReport()
        if self.showFeatureRemovalReport:
            processor.reportFeatureRemovals(f"Features removidas na otimização de {dataset.name}")
        attack_instances = sum(
            1
            for label in label_names
            if str(label).strip().upper() not in {"BENIGN", "NORMAL", "0"}
        )
        total_instances = len(label_names)
        effective_parameters = mergeParameters(self.parameters, dataset.parameters, task=self.config.task)
        return PreparedDataset(
            name=dataset.name,
            datasetPath=dataset.path,
            stream=stream,
            targetNames=tuple(target_names),
            featureNames=tuple(feature_names),
            labelNames=tuple(label_names),
            totalInstances=total_instances,
            attackInstances=attack_instances,
            attackRatioPercent=(100.0 * attack_instances / total_instances) if total_instances else 0.0,
            parameters=effective_parameters,
        )

    # Cria componentes novos para cada execução usando o mesmo pipeline empregado no treinamento normal.
    def _createPipeline(self, dataset, parameters, resultManager, generatePlots):
        return ExperimentRunner.createPipeline(
            task=self.config.task, stream=dataset.stream, datasetName=dataset.name,
            labelNames=dataset.labelNames, parameters=parameters,
            outputDirectory=self.config.outputRoot, resultManager=resultManager,
            generatePlots=generatePlots,
        )

    # Normaliza o código do modelo usando o registry adequado à tarefa corrente.
    def _getModelCode(self, parameters):
        code = parameters["model"]["code"]
        if self.config.task == "classification":
            return ClassifierRegistry.normalizeCode(code)
        return ModelRegistry.normalizeCode(code)

    # Sanitiza nomes lógicos antes de utilizá-los em diretórios de resultados.
    def _normalizeName(self, value):
        ascii_name = unicodedata.normalize("NFKD", str(value).strip()).encode("ascii", "ignore").decode("ascii")
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_")
        return safe_name or "dataset"

    # Converte caminhos, tuplas e estruturas aninhadas para tipos serializáveis em JSON.
    def _jsonSafe(self, value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [self._jsonSafe(item) for item in value]
        if isinstance(value, list):
            return [self._jsonSafe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._jsonSafe(item) for key, item in value.items()}
        return value

    @staticmethod
    # Importa Optuna somente quando a rotina de otimização é executada.
    def _loadOptuna():
        try:
            import optuna
        except ImportError as error:
            raise ImportError("Optuna is not installed in the environment.") from error
        return optuna
