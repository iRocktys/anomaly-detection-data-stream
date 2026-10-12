import numpy as np

from src.Anomaly.Models import ModelRegistry
from src.Metrics.IncrementalMetrics import IncrementalMetrics
from src.Pipeline.InstanceFactory import InstanceFactory
from src.Pipeline.BinaryPipelineSupport import BinaryPipelineSupport
from src.Pipeline.ResultContracts import PipelineRunContext, ResultManagerProtocol
from src.Pipeline.StreamPreprocessor import StreamPreprocessor
from src.Scores.ScoreMovingAverages import ScoreMovingAverages


class AnomalyDetectionPipeline(BinaryPipelineSupport):
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
        metricsWindowSize=100,
        seed=42,
        generatePlots=True,
        initialWarmupSize=1000,
        thresholdScoreSource="raw",
        resultManager=None,
        strictTrainingErrors=False,
    ):
        # Inicializa o pipeline binário de detecção com score, threshold e treinamento incremental.
        self.stream = stream
        self.datasetName = str(datasetName)
        self.modelCode = ModelRegistry.normalizeCode(modelCode)
        self.threshold = threshold
        self.labelNames = tuple(labelNames)
        self.modelParameters = dict(modelParameters or {})
        self.preprocessor = StreamPreprocessor(imputer, normalizer)
        self.trainingStrategy = trainingStrategy
        self.scoreMovingAverages = ScoreMovingAverages(scoreWindowSizes)
        self.metrics = IncrementalMetrics()
        self.metricsWindowSize = max(1, int(metricsWindowSize))
        self.seed = int(seed)
        self.generatePlots = bool(generatePlots)
        self.strictTrainingErrors = bool(strictTrainingErrors)
        self.resultManager = resultManager
        if not isinstance(self.resultManager, ResultManagerProtocol):
            raise TypeError("resultManager must implement start(context), collect(row), and finish().")
        self.thresholdCalibrationWindowSize = self.getThresholdCalibrationWindowSize()
        self.initialWarmupSize = self.resolveInitialWarmupSize(initialWarmupSize)
        self.thresholdCalibrationStart = self.initialWarmupSize - self.thresholdCalibrationWindowSize
        self.modelWarmupSize = self.thresholdCalibrationStart
        self.thresholdScoreSource = self.normalizeThresholdScoreSource(thresholdScoreSource)
        self.thresholdScoreLabel = self.getThresholdScoreLabel()
        self.thresholdWarmupScores = []
        self.modelName = None
        self.model = None
        self.validateWarmupProtocol()

    def run(self):
        # Executa toda a stream uma única vez e devolve o resultado produzido pelo gerenciador configurado.
        self.resetComponents()
        self.stream.restart()
        self.createModel()
        moving_average_columns = tuple(f"scoreMa{size}" for size in self.scoreMovingAverages.windowSizes)
        self.resultManager.start(PipelineRunContext(
            task="anomaly",
            datasetName=self.datasetName,
            modelCode=self.modelCode,
            metricsWindowSize=self.metricsWindowSize,
            movingAverageColumns=moving_average_columns,
            generatePlots=self.generatePlots,
        ))
        instance_index = 0
        while self.stream.has_more_instances():
            if instance_index >= len(self.labelNames):
                raise RuntimeError("The stream contains more instances than the binary label vector.")
            raw_instance = self.stream.next_instance()
            self.resultManager.collect(self.processInstance(raw_instance, instance_index))
            instance_index += 1
        if instance_index != len(self.labelNames):
            raise RuntimeError("The number of binary labels differs from the number of stream instances.")
        return self.resultManager.finish()

    def createModel(self):
        # Cria o detector registrado usando o schema da stream e a semente da execução.
        schema = InstanceFactory.getStreamSchema(self.stream)
        self.modelName, self.model, self.modelParameters = ModelRegistry.create(
            schema=schema,
            code=self.modelCode,
            parameters=self.modelParameters,
            runSeed=self.seed,
        )

    def processInstance(self, rawInstance, instanceIndex):
        # Processa uma instância no protocolo score-predição-treino-atualização sem vazamento temporal.
        raw_values = np.asarray(rawInstance.x, dtype=np.float64)
        normalized_values, imputed_values, observed_mask = self.preprocessor.transform(raw_values)
        model_instance = InstanceFactory.createLabeledInstance(rawInstance, normalized_values)
        raw_score = float(self.model.score_instance(model_instance))
        score_averages = self.scoreMovingAverages.calculate(raw_score)
        score = self.getThresholdScore(raw_score, score_averages)
        threshold_ready = bool(self.threshold.isReady())
        is_warmup = instanceIndex < self.initialWarmupSize
        evaluation_ready = bool(threshold_ready and not is_warmup)
        threshold_value = float(self.threshold.getThreshold()) if threshold_ready else np.nan
        predicted_label = int(score > threshold_value) if evaluation_ready else 0
        true_label = int(rawInstance.y_index)
        if true_label not in (0, 1):
            raise ValueError("The anomaly detection pipeline accepts only binary labels 0 and 1.")
        should_train = self.trainingStrategy.shouldTrain(
            prediction=predicted_label,
            thresholdReady=threshold_ready,
            isWarmup=is_warmup,
        )
        was_trained = self.trainModel(model_instance) if should_train else False
        metric_values = self.metrics.update(true_label, predicted_label) if evaluation_ready else self.emptyMetricValues()
        row = {
            "instanceId": int(instanceIndex),
            "task": "anomaly",
            "dataset": self.datasetName,
            "model": self.modelCode,
            "modelConfig": self.modelName,
            "imputer": self.preprocessor.getImputerName(),
            "normalizer": self.preprocessor.getNormalizerName(),
            "trainingStrategy": self.trainingStrategy.name,
            "thresholdStrategy": self.getThresholdName(),
            "thresholdScoreSource": self.thresholdScoreSource,
            "thresholdScoreLabel": self.thresholdScoreLabel,
            "evaluationName": f"{self.modelCode}_{self.getThresholdName()}_{self.getThresholdScoreName()}",
            "warmup": self.initialWarmupSize,
            "modelWarmup": self.modelWarmupSize,
            "thresholdWarmup": self.thresholdCalibrationWindowSize,
            "isWarmup": int(is_warmup),
            "evaluationReady": int(evaluation_ready),
            "thresholdCalibrationWindow": self.thresholdCalibrationWindowSize,
            "thresholdCalibrationStart": self.thresholdCalibrationStart,
            "rawScore": raw_score,
            "score": score,
            **score_averages,
            "threshold": threshold_value,
            "thresholdReady": int(threshold_ready),
            "trueLabel": true_label,
            "labelName": self.labelNames[instanceIndex],
            "isAttack": true_label,
            "predictedLabel": predicted_label,
            "trainingAllowed": int(bool(should_train)),
            "wasTrained": int(bool(was_trained)),
            "missingFeatureCount": int(np.sum(~observed_mask)),
            **metric_values,
        }
        self.preprocessor.update(raw_values, imputed_values, observed_mask)
        self.updateThreshold(score, instanceIndex)
        return row

    def updateThreshold(self, score, instanceIndex):
        # Atualiza ou calibra o threshold apenas depois de produzir a decisão da instância atual.
        if self.threshold.isReady():
            if instanceIndex >= self.initialWarmupSize:
                self.threshold.update(score, instanceIndex)
            return
        if self.thresholdCalibrationWindowSize == 0:
            self.threshold.initialize([])
            return
        if instanceIndex < self.thresholdCalibrationStart:
            return
        if instanceIndex >= self.initialWarmupSize:
            raise RuntimeError("The threshold was not initialized within the configured warm-up.")
        self.thresholdWarmupScores.append(float(score))
        if len(self.thresholdWarmupScores) == self.thresholdCalibrationWindowSize:
            self.threshold.initialize(self.thresholdWarmupScores)
        elif len(self.thresholdWarmupScores) > self.thresholdCalibrationWindowSize:
            raise RuntimeError("Calibration received more scores than the configured size.")

    def getThresholdCalibrationWindowSize(self):
        # Obtém a quantidade de scores consumidos pela calibração inicial do threshold.
        if hasattr(self.threshold, "config") and hasattr(self.threshold.config, "warmupSize"):
            return int(self.threshold.config.warmupSize)
        return 0

    def resolveInitialWarmupSize(self, initialWarmupSize):
        # Valida o warm-up total em relação ao período de calibração do threshold.
        warmup_size = int(initialWarmupSize or 0)
        if warmup_size < self.thresholdCalibrationWindowSize:
            raise ValueError("initialWarmupSize must be greater than or equal to the threshold calibration size.")
        if warmup_size >= len(self.labelNames):
            raise ValueError("initialWarmupSize must leave at least one instance for evaluation.")
        return warmup_size

    def validateWarmupProtocol(self):
        # Valida se a janela interna do AIF cabe integralmente no warm-up exclusivo do detector.
        if self.modelWarmupSize < 0:
            raise ValueError("The detector-only warm-up cannot be negative.")
        if self.modelCode == "AIF":
            default_window = ModelRegistry.definitions["AIF"].defaults["window_size"]
            window_size = int(self.modelParameters.get("window_size", default_window))
            if window_size > self.modelWarmupSize:
                raise ValueError(f"The AIF window ({window_size}) cannot exceed the detector warm-up ({self.modelWarmupSize}).")

    def normalizeThresholdScoreSource(self, thresholdScoreSource):
        # Normaliza a seleção entre score bruto e uma média móvel previamente configurada.
        source = str(thresholdScoreSource).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
        if source in {"raw", "rawscore", "score"}:
            return "raw"
        window_text = source.replace("scorema", "", 1) if source.startswith("scorema") else source.replace("ma", "", 1) if source.startswith("ma") else source
        if not window_text.isdigit():
            raise ValueError("thresholdScoreSource must be 'raw' or a moving average such as 'ma10'.")
        window_size = int(window_text)
        if window_size not in self.scoreMovingAverages.windowSizes:
            raise ValueError(f"Moving average {window_size} is not present in scoreWindowSizes.")
        return f"scoreMa{window_size}"

    def getThresholdScore(self, rawScore, scoreAverages):
        # Seleciona o score efetivamente entregue ao threshold na instância atual.
        if self.thresholdScoreSource == "raw":
            return float(rawScore)
        return float(scoreAverages[self.thresholdScoreSource])

    def getThresholdScoreLabel(self):
        # Produz um rótulo curto para a origem do score do threshold.
        if self.thresholdScoreSource == "raw":
            return "Raw"
        return f"MA {self.thresholdScoreSource.replace('scoreMa', '')}"

    def getThresholdScoreName(self):
        # Produz um nome seguro para persistir a origem do score em artefatos.
        if self.thresholdScoreSource == "raw":
            return "raw"
        return f"ma{self.thresholdScoreSource.replace('scoreMa', '')}"

    def getThresholdName(self):
        # Retorna o nome lógico do threshold informado pelo próprio estado do componente.
        state = self.threshold.getState()
        return str(state.get("name", self.threshold.__class__.__name__)) if isinstance(state, dict) else self.threshold.__class__.__name__

    def resetComponents(self):
        # Reinicia todos os estados incrementais para garantir uma execução independente.
        self.threshold.reset()
        self.preprocessor.reset()
        self.scoreMovingAverages.reset()
        self.metrics.reset()
        self.thresholdWarmupScores = []
