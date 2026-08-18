import numpy as np

from src.Anomaly.Models import ModelRegistry
from src.Data.OnlineImputers import ZeroOnlineImputer
from src.Data.OnlineNormalizers import NoOnlineNormalizer
from src.Metrics.IncrementalMetrics import IncrementalMetrics
from src.Pipeline.ResultManager import ResultManager
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
    ):
        self.stream = stream
        self.datasetName = str(datasetName)
        self.modelCode = ModelRegistry.normalizeCode(modelCode)
        self.threshold = threshold
        self.labelNames = list(labelNames)
        self.modelParameters = dict(modelParameters or {})
        self.imputer = (
            imputer
            if imputer is not None
            else ZeroOnlineImputer()
        )
        self.normalizer = (
            normalizer
            if normalizer is not None
            else NoOnlineNormalizer()
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
        self.resultManager = ResultManager(
            outputPath
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
                "initialWarmupSize deve ser menor que a quantidade de "
                "instâncias para que exista ao menos uma instância de avaliação."
            )

        self.thresholdCalibrationStart = (
            self.initialWarmupSize
            - self.thresholdCalibrationWindowSize
        )

        self.thresholdWarmupScores = []
        self.modelName = None
        self.model = None

    def run(self):
        self.resetComponents()
        self.stream.restart()
        self.createModel()

        rows = []
        instanceIndex = 0

        while self.stream.has_more_instances():
            if instanceIndex >= len(self.labelNames):
                raise RuntimeError(
                    "Não existe rótulo original para a instância atual da stream."
                )

            rawInstance = (
                self.stream.next_instance()
            )
            rows.append(
                self.processInstance(
                    rawInstance,
                    instanceIndex,
                )
            )
            instanceIndex += 1

        if instanceIndex != len(self.labelNames):
            raise RuntimeError(
                "A quantidade de rótulos originais é diferente da quantidade "
                "de instâncias da stream."
            )

        movingAverageColumns = [
            f"scoreMa{windowSize}"
            for windowSize
            in self.scoreMovingAverages.windowSizes
        ]

        return self.resultManager.save(
            rows,
            self.datasetName,
            self.modelCode,
            self.metricsWindowSize,
            movingAverageColumns,
            self.generatePlots,
        )

    def createModel(self):
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
                "O imputador produziu valores não finitos."
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
                "O normalizador produziu valores não finitos."
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
                "O threshold não foi inicializado dentro do warm-up configurado."
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
                "Foram coletados mais scores do que o tamanho da "
                "janela de calibração."
            )

    def trainModel(
        self,
        modelInstance,
    ):
        try:
            self.model.train(
                modelInstance
            )
            return True
        except ValueError:
            return False

    def createModelInstance(
        self,
        rawInstance,
        values,
    ):
        from capymoa.instance import LabeledInstance

        return LabeledInstance.from_array(
            schema=rawInstance.schema,
            x=np.asarray(
                values,
                dtype=np.float64,
            ),
            y_index=int(
                rawInstance.y_index
            ),
        )

    def getStreamSchema(self):
        if hasattr(
            self.stream,
            "get_schema",
        ):
            return (
                self.stream.get_schema()
            )

        if hasattr(
            self.stream,
            "schema",
        ):
            return self.stream.schema

        raise AttributeError(
            "Não foi possível obter o schema da stream."
        )

    def getThresholdCalibrationWindowSize(
        self,
    ):
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
        if initialWarmupSize is None:
            initialWarmupSize = (
                self.thresholdCalibrationWindowSize
            )

        initialWarmupSize = int(
            initialWarmupSize
        )

        if initialWarmupSize < 0:
            raise ValueError(
                "initialWarmupSize deve ser maior ou igual a zero."
            )

        if (
            initialWarmupSize
            < self.thresholdCalibrationWindowSize
        ):
            raise ValueError(
                "initialWarmupSize deve ser maior ou igual à janela de "
                "calibração do threshold "
                f"({self.thresholdCalibrationWindowSize})."
            )

        return initialWarmupSize

    def normalizeThresholdScoreSource(
        self,
        thresholdScoreSource,
    ):
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
                "thresholdScoreSource inválido. Use 'raw', 'ma10', "
                "'ma50', 'ma100' ou outra média presente em "
                "scoreWindowSizes."
            )

        if not windowText.isdigit():
            raise ValueError(
                "thresholdScoreSource inválido. O tamanho da média "
                "móvel deve ser inteiro."
            )

        windowSize = int(
            windowText
        )

        if (
            windowSize
            not in self.scoreMovingAverages.windowSizes
        ):
            raise ValueError(
                f"A média móvel {windowSize} não está disponível. "
                f"Inclua {windowSize} em scoreWindowSizes."
            )

        return f"scoreMa{windowSize}"

    def getThresholdScore(
        self,
        rawScore,
        scoreAverages,
    ):
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
                f"A série '{self.thresholdScoreSource}' não foi calculada "
                "para a instância atual."
            )

        return float(
            scoreAverages[
                self.thresholdScoreSource
            ]
        )

    def getThresholdScoreLabel(self):
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
        normalizerName = (
            self.normalizer
            .__class__
            .__name__
        )

        if (
            normalizerName
            == "NoOnlineNormalizer"
        ):
            return "none"

        if (
            normalizerName
            == "IncrementalZScoreNormalizer"
        ):
            return "incrementalZScore"

        return normalizerName

    def getImputerName(self):
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
        self.threshold.reset()
        self.imputer.reset()
        self.normalizer.reset()
        self.scoreMovingAverages.reset()
        self.metrics.reset()
        self.thresholdWarmupScores = []