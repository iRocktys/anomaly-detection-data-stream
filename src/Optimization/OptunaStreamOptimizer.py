from dataclasses import asdict, replace
import hashlib
import json

import pandas as pd

from src.Anomaly.Models import ModelRegistry
from src.Anomaly.Thresholds.Incremental.DspotThreshold import (
    DspotCalibrationError,
    DspotConfig,
    DspotThreshold,
)
from src.Data.Processor import DataStreamProcessor
from src.Data.Registries import (
    ImputerRegistry,
    NormalizerRegistry,
)
from src.Optimization.AifSearchSpace import AifSearchSpace
from src.Optimization.DspotSearchSpace import DspotSearchSpace
from src.Optimization.OptimizationConfig import (
    OptimizationConfig,
    PreparedDataset,
)
from src.Optimization.OptimizationExporter import OptimizationExporter
from src.Pipeline.MetricsResultManager import MetricsResultManager
from src.Pipeline.TrainingPipeline import TrainingPipeline
from src.Training.StrategyRegistry import TrainingStrategyRegistry


class OptunaStreamOptimizer:
    def __init__(
        self,
        config,
        searchSpace=None,
        modelSearchSpace=None,
    ):
        # Inicializa a otimização conjunta do modelo, score e DSPOT usando F1 como único objetivo.
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
        # Executa trials até completar a quantidade solicitada de configurações válidas por dataset.
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

        databasePath = (
            outputDirectory
            / "optuna.db"
        ).resolve()

        storage = (
            "sqlite:///"
            + databasePath.as_posix()
        )

        exporter = OptimizationExporter(
            outputDirectory,
            self.config.topK,
        )

        studies = {}

        prepared = {
            dataset.name: self._prepareDataset(
                dataset
            )
            for dataset in self.config.datasetConfigs
        }

        for datasetName, dataset in prepared.items():
            studyName = self._studyName(
                datasetName
            )

            study = optuna.create_study(
                study_name=studyName,
                storage=storage,
                load_if_exists=True,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=int(
                        self.config.optunaSeed
                    )
                ),
            )

            self._validateStudy(
                study,
                dataset,
            )

            initialComplete = self._countComplete(
                study
            )

            targetComplete = (
                initialComplete
                + int(
                    self.config.nTrials
                )
            )

            maximumAttempts = (
                int(
                    self.config.nTrials
                )
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
                        exporter,
                    ),
                    n_trials=1,
                    gc_after_trial=True,
                )

                attempts += 1

            completedNow = (
                self._countComplete(
                    study
                )
                - initialComplete
            )

            print(
                f"{datasetName}: "
                f"{completedNow}/{self.config.nTrials} trials válidos concluídos "
                f"em {attempts} tentativas."
            )

            if (
                completedNow
                < int(
                    self.config.nTrials
                )
            ):
                print(
                    "Aviso: o limite de tentativas foi atingido; "
                    "a otimização seguirá com os trials válidos disponíveis."
                )

            studies[
                datasetName
            ] = study

        exports = exporter.export(
            studies
        )

        return {
            "databasePath": str(
                databasePath
            ),
            "studies": studies,
            **exports,
        }

    def _countComplete(self, study):
        # Conta somente trials COMPLETE com valor objetivo F1 disponível.
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
            )
        )

    def _studyName(self, datasetName):
        # Monta o nome técnico versionado sem modificar o nome do experimento informado pelo usuário.
        return (
            f"{self.config.experimentName}"
            f"__{datasetName}"
            f"__{self.modelCode}"
            f"__{self.config.studyVersion}"
        )

    def _validateStudy(self, study, dataset):
        # Impede retomar um estudo com protocolo, dataset ou espaço de busca diferente.
        payload = self._studyConfiguration(
            dataset
        )

        signature = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                default=str,
            ).encode(
                "utf-8"
            )
        ).hexdigest()

        storedSignature = (
            study.user_attrs.get(
                "configurationSignature"
            )
        )

        hasTrials = (
            len(
                study.trials
            )
            > 0
        )

        if (
            storedSignature is None
            and hasTrials
        ):
            raise ValueError(
                f"O estudo '{study.study_name}' já possui trials de outro protocolo. "
                "Use uma nova studyVersion."
            )

        if (
            storedSignature is not None
            and storedSignature
            != signature
        ):
            raise ValueError(
                f"O estudo '{study.study_name}' foi criado com outra configuração. "
                "Altere studyVersion para iniciar um estudo compatível."
            )

        if storedSignature is None:
            study.set_user_attr(
                "configurationSignature",
                signature,
            )

            study.set_user_attr(
                "configuration",
                payload,
            )

            study.set_user_attr(
                "experimentName",
                self.config.experimentName,
            )

            study.set_user_attr(
                "datasetKey",
                dataset.name,
            )

            study.set_user_attr(
                "modelCode",
                self.modelCode,
            )

            study.set_user_attr(
                "studyVersion",
                self.config.studyVersion,
            )

    def _studyConfiguration(self, dataset):
        # Constrói a assinatura completa do protocolo para impedir retomadas incompatíveis.
        return {
            "experimentName": self.config.experimentName,
            "studyVersion": self.config.studyVersion,
            "dataset": {
                "name": dataset.name,
                "path": str(
                    dataset.datasetPath.resolve()
                ),
                "size": dataset.datasetPath.stat().st_size,
                "modifiedNanoseconds": dataset.datasetPath.stat().st_mtime_ns,
                "featureNames": list(
                    dataset.featureNames
                ),
                "targetNames": list(
                    dataset.targetNames
                ),
            },
            "modelCode": self.modelCode,
            "parameters": self.parameters,
            "optunaSeed": int(
                self.config.optunaSeed
            ),
            "optimizeModelParameters": bool(
                self.config.optimizeModelParameters
            ),
            "fixedModelParameters": dict(
                self.config.fixedModelParameters
            ),
            "aifSearchSpace": asdict(
                self.config.aifSearchSpace
            ),
            "dspotSearchSpace": asdict(
                self.config.dspotSearchSpace
            ),
        }

    def _objective(self, trial, dataset, exporter):
        # Executa uma configuração completa e retorna exclusivamente o F1-score quando ela é válida.
        optuna = self._loadOptuna()

        try:
            configuration = (
                self._suggestConfiguration(
                    trial
                )
            )

            exporter.recordIdentity(
                trial,
                dataset,
                self.modelCode,
            )

            exporter.recordConfiguration(
                trial,
                configuration,
                self.parameters,
            )

            result = self._createPipeline(
                dataset,
                configuration,
            ).run()

            exporter.recordResult(
                trial,
                result,
            )

            return float(
                result.streamMetrics[
                    "f1"
                ]
            )

        except DspotCalibrationError as error:
            raise optuna.TrialPruned(
                str(
                    error
                )
            ) from error

    def _suggestConfiguration(self, trial):
        # Sugere conjuntamente score, parâmetros DSPOT e parâmetros AIF respeitando os warm-ups fixos.
        imputerName = self.parameters[
            "preprocessing"
        ][
            "imputer"
        ]

        configuration = self.searchSpace.suggest(
            trial,
            imputerName=imputerName,
        )

        totalWarmup = int(
            self.parameters[
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
            modelParameters = self.modelSearchSpace.suggest(
                trial,
                fixedParameters=self.config.fixedModelParameters,
                modelWarmupSize=modelWarmup,
            )

        else:
            modelParameters = {
                **self.parameters[
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

    def _prepareDataset(self, dataset):
        # Carrega o dataset uma única vez e cria a stream reutilizada pelos trials.
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
        )

    def _createPipeline(self, dataset, configuration):
        # Cria o pipeline leve de cada trial com 500 instâncias para AIF e 500 para DSPOT.
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

        preprocessing = self.parameters[
            "preprocessing"
        ]

        training = self.parameters[
            "training"
        ]

        execution = self.parameters[
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
            generatePlots=False,
            initialWarmupSize=int(
                execution[
                    "initialWarmupSize"
                ]
            ),
            thresholdScoreSource=configuration.thresholdScoreSource,
            resultManager=MetricsResultManager(),
            strictTrainingErrors=bool(
                execution.get(
                    "strictTrainingErrors",
                    False,
                )
            ),
        )

    @staticmethod
    def _loadOptuna():
        # Importa Optuna somente quando uma otimização é executada.
        try:
            import optuna

        except ImportError as error:
            raise ImportError(
                "Optuna não está instalado no ambiente."
            ) from error

        return optuna
