import numpy as np

from src.Anomaly.Models import ModelRegistry
from src.Data.OnlineImputers import IncrementalMeanImputer
from src.Data.OnlineNormalizers import IncrementalZScoreNormalizer
from src.Metrics.IncrementalMetrics import IncrementalMetrics
from src.Pipeline.ResultContracts import (
    PipelineRunContext,
    ResultManagerProtocol,
)
from src.Pipeline.ResultManager import ResultManager
from src.Pipeline.InstanceFactory import InstanceFactory
from src.Scores.ScoreMovingAverages import ScoreMovingAverages
from src.Training.TrainAllStrategy import TrainAllStrategy


class TrainingPipeline:
    def __init__(
        self,
        stream,
        datasetName,
        modelCode,
        threshold,
        labelNames,
        modelParameters=None,
        imputer=None,
        normalizer=None,
        trainingStrategy=None,
        scoreWindowSizes=None,
        metricsWindowSize=1000,
        outputPath="output",
        normalClassIndex=0,
        seed=1,
        generatePlots=True,
        initialWarmupSize=None,
        thresholdScoreSource="raw",
        resultManager=None,
        strictTrainingErrors=False,
    ):
        # Inicializa o estado e as dependências necessárias para a execução da classe.
        self.strictTrainingErrors = bool(strictTrainingErrors)
        self.stream = stream
        self.datasetName = str(datasetName)
        self.modelCode = ModelRegistry.normalizeCode(modelCode)
        self.threshold = threshold
        self.labelNames = list(labelNames)
        self.modelParameters = dict(modelParameters or {})
        self.imputer = (
            imputer
            if imputer is not None
            else IncrementalMeanImputer()
        )
        self.normalizer = (
            normalizer
            if normalizer is not None
            else IncrementalZScoreNormalizer()
        )
        self.trainingStrategy = (
            trainingStrategy
            if trainingStrategy is not None
            else TrainAllStrategy()
        )

        self.scoreMovingAverages = ScoreMovingAverages(
            scoreWindowSizes
        )
        self.thresholdScoreSource = (
            self.normalizeThresholdScoreSource(
                thresholdScoreSource
            )
        )
        self.thresholdScoreLabel = (
            self.getThresholdScoreLabel()
        )

        self.metrics = IncrementalMetrics()
        self.metricsWindowSize = max(
            1,
            int(metricsWindowSize),
        )
        self.resultManager = (
            resultManager
            if resultManager is not None
            else ResultManager(outputPath)
        )
        if not isinstance(self.resultManager, ResultManagerProtocol):
            raise TypeError(
                "resultManager must implement start(context), collect(row), and finish()."
            )
        self.normalClassIndex = int(
            normalClassIndex
        )
        self.seed = int(seed)
        self.generatePlots = bool(
            generatePlots
        )

        self.thresholdCalibrationWindowSize = (
            self.getThresholdCalibrationWindowSize()
        )
        self.initialWarmupSize = (
            self.resolveInitialWarmupSize(
                initialWarmupSize
            )
        )

        if self.initialWarmupSize >= len(self.labelNames):
            raise ValueError(
                "initialWarmupSize must leave at least one instance for evaluation."
            )

        self.thresholdCalibrationStart = (
            self.initialWarmupSize
            - self.thresholdCalibrationWindowSize
        )

        self.modelWarmupSize = (
            self.thresholdCalibrationStart
        )

        self.validateWarmupProtocol()

        self.thresholdWarmupScores = []
        self.modelName = None
        self.model = None

    def run(self):
        # Executa o fluxo principal da rotina e retorna os resultados produzidos.
        self.resetComponents()
        self.stream.restart()
        self.createModel()

        movingAverageColumns = tuple(
            f"scoreMa{windowSize}"
            for windowSize in self.scoreMovingAverages.windowSizes
        )
        self.resultManager.start(
            PipelineRunContext(
                datasetName=self.datasetName,
                modelCode=self.modelCode,
                metricsWindowSize=self.metricsWindowSize,
                movingAverageColumns=movingAverageColumns,
                generatePlots=self.generatePlots,
            )
        )
        instanceIndex = 0

        while self.stream.has_more_instances():
            if instanceIndex >= len(self.labelNames):
                raise RuntimeError(
                    "There is no original label for the current stream instance."
                )

            rawInstance = (
                self.stream.next_instance()
            )
            row = self.processInstance(
                rawInstance,
                instanceIndex,
            )
            self.resultManager.collect(row)
            instanceIndex += 1

        if instanceIndex != len(self.labelNames):
            raise RuntimeError(
                "The number of original labels differs from the number of stream instances."
            )

        return self.resultManager.finish()

    def createModel(self):
        # Cria o modelo configurado usando o schema e os parâmetros da execução.
        schema = self.getStreamSchema()
        definition = (
            ModelRegistry
            .definitions[self.modelCode]
        )

        parameters = dict(
            definition.defaults
        )
        parameters.update(
            self.modelParameters
        )
        parameters["schema"] = schema
        parameters.setdefault(
            definition.seedParameter,
            self.seed,
        )

        ModelRegistry.validateParameters(
            self.modelCode,
            parameters,
        )

        modelClass = definition.loader()
        self.model = modelClass(
            **parameters
        )
        self.modelName = (
            definition.displayName
        )

    def processInstance(
        self,
        rawInstance,
        instanceIndex,
    ):
        # Processa uma instância da stream aplicando o fluxo incremental configurado.
        rawValues = np.asarray(
            rawInstance.x,
            dtype=np.float64,
        )
        observedMask = np.isfinite(
            rawValues
        )

        imputedValues = (
            self.imputer.transform(
                rawValues
            )
        )

        if np.any(
            ~np.isfinite(imputedValues)
        ):
            raise RuntimeError(
                "The imputer produced non-finite values."
            )

        normalizedValues = (
            self.normalizer.transform(
                imputedValues
            )
        )

        if np.any(
            ~np.isfinite(normalizedValues)
        ):
            raise RuntimeError(
                "The normalizer produced non-finite values."
            )

        modelInstance = (
            self.createModelInstance(
                rawInstance,
                normalizedValues,
            )
        )

        rawScore = float(
            self.model.score_instance(
                modelInstance
            )
        )
        scoreAverages = (
            self.scoreMovingAverages
            .calculate(rawScore)
        )

        score = self.getThresholdScore(
            rawScore,
            scoreAverages,
        )

        thresholdReady = bool(
            self.threshold.isReady()
        )
        isWarmup = (
            instanceIndex
            < self.initialWarmupSize
        )
        evaluationReady = bool(
            thresholdReady
            and not isWarmup
        )

        thresholdValue = (
            float(
                self.threshold.getThreshold()
            )
            if thresholdReady
            else np.nan
        )

        predictedLabel = (
            int(score > thresholdValue)
            if evaluationReady
            else 0
        )

        trueLabel = int(
            rawInstance.y_index
        )
        isAttack = int(
            trueLabel
            != self.normalClassIndex
        )
        labelName = str(
            self.labelNames[instanceIndex]
        )

        shouldTrain = (
            self.trainingStrategy
            .shouldTrain(
                prediction=predictedLabel,
                thresholdReady=thresholdReady,
                isWarmup=isWarmup,
            )
        )

        wasTrained = (
            self.trainModel(
                modelInstance
            )
            if shouldTrain
            else False
        )

        metricValues = (
            self.metrics.update(
                isAttack,
                predictedLabel,
            )
            if evaluationReady
            else self.emptyMetricValues()
        )

        row = {
            "instanceId": int(
                instanceIndex
            ),
            "dataset": self.datasetName,
            "model": self.modelCode,
            "modelConfig": self.modelName,
            "imputer": self.getImputerName(),
            "normalizer": self.getNormalizerName(),
            "trainingStrategy": (
                self.trainingStrategy.name
            ),
            "thresholdStrategy": (
                self.getThresholdName()
            ),
            "thresholdScoreSource": (
                self.thresholdScoreSource
            ),
            "thresholdScoreLabel": (
                self.thresholdScoreLabel
            ),
            "evaluationName": (
                f"{self.modelCode}_"
                f"{self.getThresholdName()}_"
                f"{self.getThresholdScoreName()}_"
                f"{self.getImputerName()}_"
                f"{self.trainingStrategy.name}"
            ),
            "warmup": (
                self.initialWarmupSize
            ),
            "modelWarmup": (
                self.modelWarmupSize
            ),
            "thresholdWarmup": (
                self.thresholdCalibrationWindowSize
            ),
            "isWarmup": int(
                isWarmup
            ),
            "evaluationReady": int(
                evaluationReady
            ),
            "thresholdCalibrationWindow": (
                self.thresholdCalibrationWindowSize
            ),
            "thresholdCalibrationStart": (
                self.thresholdCalibrationStart
            ),
            "rawScore": rawScore,
            "score": score,
            **scoreAverages,
            "threshold": thresholdValue,
            "thresholdReady": int(
                thresholdReady
            ),
            "trueLabel": trueLabel,
            "labelName": labelName,
            "isAttack": isAttack,
            "predictedLabel": (
                predictedLabel
            ),
            "trainingAllowed": int(
                bool(shouldTrain)
            ),
            "wasTrained": int(
                bool(wasTrained)
            ),
            "missingFeatureCount": int(
                np.sum(~observedMask)
            ),
            **metricValues,
        }

        self.imputer.update(
            rawValues
        )
        self.normalizer.update(
            imputedValues,
            observedMask=observedMask,
        )
        self.updateThreshold(
            score,
            instanceIndex,
        )

        return row

    def updateThreshold(
        self,
        score,
        instanceIndex,
    ):
        # Atualiza o threshold incremental com o score produzido pela instância atual.
        if self.threshold.isReady():
            if (
                instanceIndex
                >= self.initialWarmupSize
            ):
                self.threshold.update(
                    score,
                    instanceIndex,
                )

            return

        if (
            self.thresholdCalibrationWindowSize
            == 0
        ):
            self.threshold.initialize([])
            return

        if (
            instanceIndex
            < self.thresholdCalibrationStart
        ):
            return

        if (
            instanceIndex
            >= self.initialWarmupSize
        ):
            raise RuntimeError(
                "The threshold was not initialized within the configured warm-up."
            )

        self.thresholdWarmupScores.append(
            float(score)
        )

        if (
            len(self.thresholdWarmupScores)
            == self.thresholdCalibrationWindowSize
        ):
            self.threshold.initialize(
                self.thresholdWarmupScores
            )

        elif (
            len(self.thresholdWarmupScores)
            > self.thresholdCalibrationWindowSize
        ):
            raise RuntimeError(
                "More scores were collected than the calibration-window size."
            )

    def trainModel(
        self,
        modelInstance,
    ):
        # Treina o modelo incrementalmente com a instância permitida pela estratégia atual.
        try:
            self.model.train(
                modelInstance
            )
            return True
        except ValueError:
            if self.strictTrainingErrors:
                raise
            return False

    def createModelInstance(
        self,
        rawInstance,
        values,
    ):
        # Reconstrói a instância rotulada usando a camada comum de compatibilidade com o CapyMOA.
        return InstanceFactory.createLabeledInstance(
            rawInstance,
            values,
        )

    def getStreamSchema(self):
        # Obtém o schema da stream sem depender de uma única versão da API do CapyMOA.
        return InstanceFactory.getStreamSchema(
            self.stream
        )

    def getThresholdCalibrationWindowSize(
        self,
    ):
        # Obtém o tamanho total da janela usada para calibrar o threshold.
        if (
            hasattr(
                self.threshold,
                "config",
            )
            and hasattr(
                self.threshold.config,
                "warmupSize",
            )
        ):
            return int(
                self.threshold
                .config
                .warmupSize
            )

        return 0

    def resolveInitialWarmupSize(
        self,
        initialWarmupSize,
    ):
        # Resolve e valida o tamanho do warm-up inicial utilizado pela execução.
        if initialWarmupSize is None:
            initialWarmupSize = (
                self.thresholdCalibrationWindowSize
            )

        initialWarmupSize = int(
            initialWarmupSize
        )

        if initialWarmupSize < 0:
            raise ValueError(
                "initialWarmupSize must be greater than or equal to zero."
            )

        if (
            initialWarmupSize
            < self.thresholdCalibrationWindowSize
        ):
            raise ValueError(
                "initialWarmupSize must be greater than or equal to the threshold calibration window "
                f"({self.thresholdCalibrationWindowSize})."
            )

        return initialWarmupSize

    def validateWarmupProtocol(self):
        # Valida o encaixe da janela do AIF na fase anterior à calibração do threshold.
        if self.modelWarmupSize < 0:
            raise ValueError(
                "The model-only warm-up cannot be negative."
            )

        if self.modelCode == "AIF":
            definition = (
                ModelRegistry
                .definitions[
                    "AIF"
                ]
            )

            windowSize = int(
                self.modelParameters.get(
                    "window_size",
                    definition.defaults[
                        "window_size"
                    ],
                )
            )

            if windowSize > self.modelWarmupSize:
                raise ValueError(
                    "The AIF window cannot exceed the model warm-up phase "
                    f"({self.modelWarmupSize})."
                )

    def normalizeThresholdScoreSource(
        self,
        thresholdScoreSource,
    ):
        # Normaliza o identificador da fonte de score usada pelo threshold.
        source = (
            str(thresholdScoreSource)
            .strip()
            .lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        if source in [
            "raw",
            "rawscore",
            "score",
        ]:
            return "raw"

        if source.startswith(
            "scorema"
        ):
            windowText = source.replace(
                "scorema",
                "",
                1,
            )

        elif source.startswith("ma"):
            windowText = source.replace(
                "ma",
                "",
                1,
            )

        elif source.isdigit():
            windowText = source

        else:
            raise ValueError(
                "Invalid thresholdScoreSource. Use 'raw', 'ma10', 'ma50', 'ma100', "
                "or another moving average present in scoreWindowSizes."
            )

        if not windowText.isdigit():
            raise ValueError(
                "Invalid thresholdScoreSource. The moving-average size must be an integer."
            )

        windowSize = int(
            windowText
        )

        if (
            windowSize
            not in self.scoreMovingAverages.windowSizes
        ):
            raise ValueError(
                f"Moving average {windowSize} is not available. Include {windowSize} in scoreWindowSizes."
            )

        return f"scoreMa{windowSize}"

    def getThresholdScore(
        self,
        rawScore,
        scoreAverages,
    ):
        # Obtém o score efetivamente utilizado pela estratégia de threshold.
        if (
            self.thresholdScoreSource
            == "raw"
        ):
            return float(rawScore)

        if (
            self.thresholdScoreSource
            not in scoreAverages
        ):
            raise RuntimeError(
                f"Series '{self.thresholdScoreSource}' was not calculated for the current instance."
            )

        return float(
            scoreAverages[
                self.thresholdScoreSource
            ]
        )

    def getThresholdScoreLabel(self):
        # Retorna o nome descritivo da fonte de score utilizada pelo threshold.
        if (
            self.thresholdScoreSource
            == "raw"
        ):
            return "Raw"

        windowSize = (
            self.thresholdScoreSource
            .replace(
                "scoreMa",
                "",
            )
        )

        return f"MA {windowSize}"

    def getThresholdScoreName(self):
        # Retorna o identificador normalizado da fonte de score do threshold.
        if (
            self.thresholdScoreSource
            == "raw"
        ):
            return "raw"

        windowSize = (
            self.thresholdScoreSource
            .replace(
                "scoreMa",
                "",
            )
        )

        return f"ma{windowSize}"

    def getThresholdName(self):
        # Retorna o nome padronizado da estratégia de threshold utilizada.
        state = (
            self.threshold.getState()
        )

        if (
            isinstance(state, dict)
            and "name" in state
        ):
            return str(
                state["name"]
            )

        return (
            self.threshold
            .__class__
            .__name__
        )

    def getNormalizerName(self):
        # Retorna o nome padronizado do normalizador utilizado na execução.
        normalizerName = (
            self.normalizer
            .__class__
            .__name__
        )

        if (
            normalizerName
            == "IncrementalZScoreNormalizer"
        ):
            return "incrementalZScore"

        return normalizerName

    def getImputerName(self):
        # Retorna o nome padronizado do imputador utilizado na execução.
        if hasattr(
            self.imputer,
            "name",
        ):
            return str(
                self.imputer.name
            )

        return (
            self.imputer
            .__class__
            .__name__
        )

    def emptyMetricValues(self):
        # Retorna métricas vazias para instâncias ainda não liberadas para avaliação.
        return {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "cumulativeTp": (
                self.metrics.tp
            ),
            "cumulativeTn": (
                self.metrics.tn
            ),
            "cumulativeFp": (
                self.metrics.fp
            ),
            "cumulativeFn": (
                self.metrics.fn
            ),
            "cumulativeAccuracy": np.nan,
            "cumulativePrecision": np.nan,
            "cumulativeRecall": np.nan,
            "cumulativeSpecificity": np.nan,
            "cumulativeF1": np.nan,
            "cumulativeMcc": np.nan,
        }

    def resetComponents(self):
        # Reinicia os componentes incrementais antes de uma nova execução completa.
        self.threshold.reset()
        self.imputer.reset()
        self.normalizer.reset()
        self.scoreMovingAverages.reset()
        self.metrics.reset()
        self.thresholdWarmupScores = []
