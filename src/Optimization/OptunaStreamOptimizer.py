from dataclasses import replace
import json
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.Anomaly.Models import ModelRegistry
from Thresholds.Incremental.DspotThreshold import (
    DspotCalibrationError,
    DspotConfig,
    DspotThreshold,
)
from src.Data.Processor import DataStreamProcessor
from src.Data.Registries import (
    ImputerRegistry,
    NormalizerRegistry,
)
from src.Experiment.Configuration import deepMerge
from src.Optimization.AifSearchSpace import AifSearchSpace
from src.Optimization.DspotSearchSpace import DspotSearchSpace
from src.Optimization.OptimizationConfig import (
    OptimizationConfig,
    PreparedDataset,
    TrialConfiguration,
)
from src.Pipeline.MetricsResultManager import MetricsResultManager
from src.Pipeline.ResultManager import ResultManager
from src.Pipeline.TrainingPipeline import TrainingPipeline
from src.Training.StrategyRegistry import TrainingStrategyRegistry


class OptunaStreamOptimizer:
    def __init__(
        self,
        config,
        searchSpace=None,
        modelSearchSpace=None,
    ):
        # Inicializa a busca em memória e mantém F1 como única função objetivo.
        if not isinstance(
            config,
            OptimizationConfig,
        ):
            raise TypeError(
                "config deve ser OptimizationConfig."
            )

        self.config = config
        self.parameters = (
            config.effectiveParameters
        )

        self.modelCode = (
            ModelRegistry.normalizeCode(
                self.parameters[
                    "model"
                ][
                    "code"
                ]
            )
        )

        self.searchSpace = (
            searchSpace
            or DspotSearchSpace(
                config.dspotSearchSpace
            )
        )

        self.modelSearchSpace = (
            modelSearchSpace
        )

        if (
            config.optimizeModelParameters
            and self.modelSearchSpace is None
            and self.modelCode
            == "AIF"
        ):
            self.modelSearchSpace = (
                AifSearchSpace(
                    config.aifSearchSpace
                )
            )

    def run(self):
        # Otimiza cada dataset em memória e reexecuta somente o melhor trial para gerar os artefatos finais.
        optuna = self._loadOptuna()

        outputDirectory = (
            self.config.outputRoot
            / self.config.experimentName
            / self.modelCode
        )

        outputDirectory.mkdir(
            parents=True,
            exist_ok=True,
        )

        preparedDatasets = [
            self._prepareDataset(
                dataset
            )
            for dataset in self.config.datasetConfigs
        ]

        bestResults = {}
        summaryRows = []

        for dataset in preparedDatasets:
            print(
                "\n"
                + "=" * 80
            )

            print(
                f"Otimização: {dataset.name}"
            )

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=int(
                        self.config.optunaSeed
                    )
                ),
            )

            self._optimizeStudy(
                study,
                dataset,
            )

            bestTrial = study.best_trial

            finalResult = self._runBestTrial(
                dataset=dataset,
                bestTrial=bestTrial,
                outputDirectory=outputDirectory,
            )

            bestResults[
                dataset.name
            ] = finalResult

            metrics = (
                finalResult[
                    "result"
                ][
                    "streamMetricsFrame"
                ]
                .iloc[
                    0
                ]
                .to_dict()
            )

            summaryRows.append(
                {
                    "dataset": dataset.name,
                    "bestTrialNumber": int(
                        bestTrial.number
                    ),
                    "optimizationF1": float(
                        bestTrial.value
                    ),
                    "reexecutionF1": float(
                        metrics[
                            "f1"
                        ]
                    ),
                    "bestDirectory": finalResult[
                        "bestDirectory"
                    ],
                    **metrics,
                }
            )

        summary = pd.DataFrame(
            summaryRows
        )

        summaryPath = (
            outputDirectory
            / "best_results.csv"
        )

        summary.to_csv(
            summaryPath,
            index=False,
        )

        return {
            "experimentName": self.config.experimentName,
            "outputDirectory": str(
                outputDirectory
            ),
            "bestResults": bestResults,
            "summary": summary,
            "summaryPath": str(
                summaryPath
            ),
        }

    def _optimizeStudy(
        self,
        study,
        dataset,
    ):
        # Continua sugerindo configurações até completar nTrials válidos ou atingir o limite de tentativas.
        targetComplete = int(
            self.config.nTrials
        )

        maximumAttempts = (
            targetComplete
            * int(
                self.config.maxTrialAttemptsMultiplier
            )
        )

        attempts = 0

        while (
            self._countComplete(
                study
            )
            < targetComplete
            and attempts
            < maximumAttempts
        ):
            study.optimize(
                lambda trial: self._objective(
                    trial,
                    dataset,
                ),
                n_trials=1,
                gc_after_trial=True,
            )

            attempts += 1

        completeTrials = self._countComplete(
            study
        )

        print(
            f"{dataset.name}: "
            f"{completeTrials}/{targetComplete} trials válidos "
            f"em {attempts} tentativas."
        )

        if completeTrials == 0:
            raise RuntimeError(
                f"Nenhum trial válido foi concluído para '{dataset.name}'."
            )

        if completeTrials < targetComplete:
            print(
                "Aviso: o limite de tentativas foi atingido e o melhor "
                "resultado disponível será utilizado."
            )

    def _countComplete(
        self,
        study,
    ):
        # Conta somente trials concluídos com F1 finito.
        return sum(
            1
            for trial in study.trials
            if (
                getattr(
                    trial.state,
                    "name",
                    str(
                        trial.state
                    ),
                )
                == "COMPLETE"
                and trial.value is not None
                and math.isfinite(
                    float(
                        trial.value
                    )
                )
            )
        )

    def _objective(
        self,
        trial,
        dataset,
    ):
        # Executa uma configuração sem persistir artefatos e retorna exclusivamente o F1-score.
        optuna = self._loadOptuna()

        try:
            configuration = (
                self._suggestConfiguration(
                    trial,
                    dataset,
                )
            )

            trial.set_user_attr(
                "configuration",
                self._configurationToDict(
                    configuration
                ),
            )

            result = self._createPipeline(
                dataset=dataset,
                configuration=configuration,
                resultManager=MetricsResultManager(),
                generatePlots=False,
            ).run()

            f1 = float(
                result.streamMetrics[
                    "f1"
                ]
            )

            if not math.isfinite(
                f1
            ):
                raise ValueError(
                    "O trial produziu um F1 não finito."
                )

            return f1

        except DspotCalibrationError as error:
            raise optuna.TrialPruned(
                str(
                    error
                )
            ) from error

    def _suggestConfiguration(
        self,
        trial,
        dataset,
    ):
        # Sugere conjuntamente score, DSPOT e modelo respeitando os parâmetros efetivos do dataset.
        parameters = dataset.parameters

        imputerName = parameters[
            "preprocessing"
        ][
            "imputer"
        ]

        configuration = self.searchSpace.suggest(
            trial,
            imputerName=imputerName,
        )

        totalWarmup = int(
            parameters[
                "execution"
            ][
                "initialWarmupSize"
            ]
        )

        modelWarmup = (
            totalWarmup
            - configuration.calibrationWindow
        )

        if modelWarmup <= 0:
            raise ValueError(
                "O warm-up do modelo deve ser maior que zero."
            )

        if self.modelSearchSpace is not None:
            modelParameters = (
                self.modelSearchSpace.suggest(
                    trial,
                    fixedParameters=self.config.fixedModelParameters,
                    modelWarmupSize=modelWarmup,
                )
            )

        else:
            modelParameters = {
                **parameters[
                    "model"
                ].get(
                    "parameters",
                    {},
                ),
                **self.config.fixedModelParameters,
            }

        return replace(
            configuration,
            modelParameters=modelParameters,
        )

    def _prepareDataset(
        self,
        dataset,
    ):
        # Carrega o dataset uma única vez e resolve também suas sobrescritas específicas de parâmetros.
        if not dataset.path.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado: {dataset.path}"
            )

        dataframe = pd.read_csv(
            dataset.path,
            low_memory=False,
        )

        processor = DataStreamProcessor(
            logging=False,
            selected_features=dataset.selectedFeatures,
            removed_features=dataset.removedFeatures,
        )

        (
            stream,
            targetNames,
            featureNames,
            labelNames,
        ) = processor.create_stream(
            dataframe,
            target_label_col=dataset.targetColumn,
            binary_label=dataset.binaryLabel,
        )

        labels = (
            dataframe[
                dataset.targetColumn
            ]
            .astype(
                str
            )
            .str.strip()
            .str.upper()
        )

        attackInstances = int(
            (
                ~labels.isin(
                    [
                        "BENIGN",
                        "NORMAL",
                    ]
                )
            ).sum()
        )

        effectiveParameters = deepMerge(
            self.parameters,
            dataset.parameters,
        )

        return PreparedDataset(
            name=dataset.name,
            datasetPath=dataset.path,
            stream=stream,
            targetNames=tuple(
                targetNames
            ),
            featureNames=tuple(
                featureNames
            ),
            labelNames=tuple(
                labelNames
            ),
            totalInstances=len(
                dataframe
            ),
            attackInstances=attackInstances,
            attackRatioPercent=(
                100.0
                * attackInstances
                / len(
                    dataframe
                )
            )
            if len(
                dataframe
            )
            else 0.0,
            parameters=effectiveParameters,
        )

    def _runBestTrial(
        self,
        dataset,
        bestTrial,
        outputDirectory,
    ):
        # Reexecuta uma única vez o melhor trial com ResultManager para gerar CSVs e plots idênticos ao treinamento.
        configuration = self._configurationFromTrial(
            bestTrial
        )

        datasetDirectory = (
            outputDirectory
            / self._normalizeName(
                dataset.name
            )
        )

        bestDirectory = (
            datasetDirectory
            / "best"
        )

        resultManager = ResultManager(
            outputPath=outputDirectory,
            fixedRunDirectory=bestDirectory,
            overwriteFixedDirectory=True,
        )

        result = self._createPipeline(
            dataset=dataset,
            configuration=configuration,
            resultManager=resultManager,
            generatePlots=True,
        ).run()

        reexecutionF1 = float(
            result[
                "streamMetricsFrame"
            ][
                "f1"
            ].iloc[
                0
            ]
        )

        parameters = self._buildTrainingParameters(
            dataset,
            configuration,
        )

        parametersPayload = {
            "experimentName": self.config.experimentName,
            "dataset": dataset.name,
            "bestTrialNumber": int(
                bestTrial.number
            ),
            "optimizationF1": float(
                bestTrial.value
            ),
            "reexecutionF1": reexecutionF1,
            "parameters": parameters,
        }

        parametersPath = (
            bestDirectory
            / "best_parameters.json"
        )

        with parametersPath.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self._jsonSafe(
                    parametersPayload
                ),
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(
            f"Melhor F1 da otimização: {float(bestTrial.value):.6f}"
        )

        print(
            f"F1 da reexecução final: {reexecutionF1:.6f}"
        )

        print(
            f"Resultado final: {bestDirectory}"
        )

        return {
            "bestDirectory": str(
                bestDirectory
            ),
            "parametersPath": str(
                parametersPath
            ),
            "parameters": parameters,
            "optimizationF1": float(
                bestTrial.value
            ),
            "reexecutionF1": reexecutionF1,
            "result": result,
        }

    def _createPipeline(
        self,
        dataset,
        configuration,
        resultManager,
        generatePlots,
    ):
        # Cria o pipeline de um trial ou da reexecução final usando exatamente a mesma configuração.
        parameters = dataset.parameters

        threshold = DspotThreshold(
            DspotConfig(
                driftDepth=configuration.driftDepth,
                calibrationSize=configuration.calibrationSize,
                initialQuantile=configuration.initialQuantile,
                risk=configuration.risk,
                refitEvery=configuration.refitEvery,
                optimizationStarts=configuration.optimizationStarts,
                tolerance=configuration.tolerance,
            )
        )

        preprocessing = parameters[
            "preprocessing"
        ]

        training = parameters[
            "training"
        ]

        execution = parameters[
            "execution"
        ]

        return TrainingPipeline(
            stream=dataset.stream,
            datasetName=dataset.name,
            modelCode=self.modelCode,
            threshold=threshold,
            labelNames=dataset.labelNames,
            modelParameters=configuration.modelParameters,
            imputer=ImputerRegistry.create(
                configuration.imputerName,
                preprocessing.get(
                    "imputerParameters"
                ),
            ),
            normalizer=NormalizerRegistry.create(
                preprocessing[
                    "normalizer"
                ],
                preprocessing.get(
                    "normalizerParameters"
                ),
            ),
            trainingStrategy=TrainingStrategyRegistry.create(
                training[
                    "strategy"
                ],
                training.get(
                    "parameters"
                ),
            ),
            scoreWindowSizes=configuration.scoreWindowSizes,
            metricsWindowSize=int(
                execution[
                    "metricsWindowSize"
                ]
            ),
            outputPath=self.config.outputRoot,
            normalClassIndex=0,
            seed=int(
                execution[
                    "seed"
                ]
            ),
            generatePlots=bool(
                generatePlots
            ),
            initialWarmupSize=int(
                execution[
                    "initialWarmupSize"
                ]
            ),
            thresholdScoreSource=configuration.thresholdScoreSource,
            resultManager=resultManager,
            strictTrainingErrors=bool(
                execution.get(
                    "strictTrainingErrors",
                    False,
                )
            ),
        )

    def _configurationToDict(
        self,
        configuration,
    ):
        # Converte a configuração tipada do trial em valores simples armazenáveis pelo Optuna em memória.
        return {
            "imputerName": configuration.imputerName,
            "thresholdScoreSource": configuration.thresholdScoreSource,
            "scoreWindowSizes": list(
                configuration.scoreWindowSizes
            ),
            "driftDepth": int(
                configuration.driftDepth
            ),
            "calibrationSize": int(
                configuration.calibrationSize
            ),
            "initialQuantile": float(
                configuration.initialQuantile
            ),
            "risk": float(
                configuration.risk
            ),
            "refitEvery": int(
                configuration.refitEvery
            ),
            "optimizationStarts": int(
                configuration.optimizationStarts
            ),
            "tolerance": float(
                configuration.tolerance
            ),
            "modelParameters": dict(
                configuration.modelParameters
            ),
        }

    def _configurationFromTrial(
        self,
        trial,
    ):
        # Reconstrói a configuração vencedora armazenada no próprio trial em memória.
        values = trial.user_attrs.get(
            "configuration"
        )

        if not values:
            raise RuntimeError(
                "O melhor trial não possui a configuração necessária para reexecução."
            )

        return TrialConfiguration(
            imputerName=str(
                values[
                    "imputerName"
                ]
            ),
            thresholdScoreSource=str(
                values[
                    "thresholdScoreSource"
                ]
            ),
            scoreWindowSizes=tuple(
                int(
                    value
                )
                for value in values.get(
                    "scoreWindowSizes",
                    [],
                )
            ),
            driftDepth=int(
                values[
                    "driftDepth"
                ]
            ),
            calibrationSize=int(
                values[
                    "calibrationSize"
                ]
            ),
            initialQuantile=float(
                values[
                    "initialQuantile"
                ]
            ),
            risk=float(
                values[
                    "risk"
                ]
            ),
            refitEvery=int(
                values[
                    "refitEvery"
                ]
            ),
            optimizationStarts=int(
                values[
                    "optimizationStarts"
                ]
            ),
            tolerance=float(
                values[
                    "tolerance"
                ]
            ),
            modelParameters=dict(
                values[
                    "modelParameters"
                ]
            ),
        )

    def _buildTrainingParameters(
        self,
        dataset,
        configuration,
    ):
        # Produz o dicionário final diretamente reutilizável pelo ExperimentRunner.
        parameters = deepMerge(
            dataset.parameters,
            {
                "model": {
                    "code": self.modelCode,
                    "parameters": dict(
                        configuration.modelParameters
                    ),
                },
                "preprocessing": {
                    "imputer": configuration.imputerName,
                },
                "threshold": {
                    "name": "dspot",
                    "scoreSource": configuration.thresholdScoreSource,
                    "scoreWindowSizes": list(
                        configuration.scoreWindowSizes
                    ),
                    "parameters": {
                        "driftDepth": int(
                            configuration.driftDepth
                        ),
                        "calibrationSize": int(
                            configuration.calibrationSize
                        ),
                        "initialQuantile": float(
                            configuration.initialQuantile
                        ),
                        "risk": float(
                            configuration.risk
                        ),
                        "refitEvery": int(
                            configuration.refitEvery
                        ),
                        "optimizationStarts": int(
                            configuration.optimizationStarts
                        ),
                        "tolerance": float(
                            configuration.tolerance
                        ),
                    },
                },
            },
        )

        return parameters

    def _normalizeName(
        self,
        value,
    ):
        # Sanitiza o nome do dataset somente para a organização dos artefatos finais.
        asciiName = (
            unicodedata.normalize(
                "NFKD",
                str(
                    value
                ).strip(),
            )
            .encode(
                "ascii",
                "ignore",
            )
            .decode(
                "ascii"
            )
        )

        safeName = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            asciiName,
        ).strip(
            "_"
        )

        return (
            safeName
            or "dataset"
        )

    def _jsonSafe(
        self,
        value,
    ):
        # Converte tuplas, caminhos e estruturas aninhadas para tipos seguros em JSON.
        if isinstance(
            value,
            Path,
        ):
            return str(
                value
            )

        if isinstance(
            value,
            tuple,
        ):
            return [
                self._jsonSafe(
                    item
                )
                for item in value
            ]

        if isinstance(
            value,
            list,
        ):
            return [
                self._jsonSafe(
                    item
                )
                for item in value
            ]

        if isinstance(
            value,
            dict,
        ):
            return {
                str(
                    key
                ): self._jsonSafe(
                    item
                )
                for key, item in value.items()
            }

        return value

    @staticmethod
    def _loadOptuna():
        # Importa Optuna somente quando a otimização é realmente executada.
        try:
            import optuna

        except ImportError as error:
            raise ImportError(
                "Optuna não está instalado no ambiente."
            ) from error

        return optuna
