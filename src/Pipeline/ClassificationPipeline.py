import numpy as np

import ProjectDefaults as defaults
from src.Classification.Models import ClassifierRegistry
from src.Data.OnlineImputers import IncrementalMeanImputer
from src.Data.OnlineNormalizers import IncrementalZScoreNormalizer
from src.Metrics.IncrementalMetrics import IncrementalMetrics
from src.Pipeline.InstanceFactory import InstanceFactory
from src.Pipeline.ResultContracts import PipelineRunContext, ResultManagerProtocol
from src.Pipeline.ResultManager import ResultManager


class ClassificationPipeline:
    def __init__(
        self,
        stream,
        datasetName,
        modelCode,
        labelNames,
        modelParameters=None,
        imputer=None,
        normalizer=None,
        metricsWindowSize=100,
        outputPath="output",
        seed=1,
        generatePlots=True,
        initialWarmupSize=0,
        resultManager=None,
        strictTrainingErrors=False,
    ):
        # Inicializa o pipeline supervisionado binário mantendo o mesmo contrato de resultados da detecção.
        self.stream = stream
        self.datasetName = str(datasetName)
        self.modelCode = ClassifierRegistry.normalizeCode(modelCode)
        self.labelNames = list(labelNames)
        self.modelParameters = dict(modelParameters or {})
        self.imputer = imputer if imputer is not None else IncrementalMeanImputer()
        self.normalizer = normalizer if normalizer is not None else IncrementalZScoreNormalizer()
        self.metrics = IncrementalMetrics()
        self.metricsWindowSize = max(1, int(metricsWindowSize))
        self.seed = int(seed)
        self.generatePlots = bool(generatePlots)
        self.initialWarmupSize = int(defaults.DEFAULT_INITIAL_WARMUP_SIZE)
        self.resultManager = resultManager if resultManager is not None else ResultManager(outputPath)
        self.strictTrainingErrors = bool(strictTrainingErrors)
        if not isinstance(self.resultManager, ResultManagerProtocol):
            raise TypeError("resultManager must implement start(context), collect(row), and finish().")
        if self.initialWarmupSize >= len(self.labelNames):
            raise ValueError("initialWarmupSize must leave at least one instance for evaluation.")
        self.model = None
        self.modelName = None

    def run(self):
        # Executa a avaliação prequential binária predizendo cada instância antes do treinamento incremental.
        self.resetComponents()
        self.stream.restart()
        self.createModel()
        self.resultManager.start(
            PipelineRunContext(
                datasetName=self.datasetName,
                modelCode=self.modelCode,
                metricsWindowSize=self.metricsWindowSize,
                movingAverageColumns=(),
                generatePlots=self.generatePlots,
            )
        )
        instance_index = 0
        while self.stream.has_more_instances():
            if instance_index >= len(self.labelNames):
                raise RuntimeError("The stream contains more instances than the label vector.")
            raw_instance = self.stream.next_instance()
            self.resultManager.collect(self.processInstance(raw_instance, instance_index))
            instance_index += 1
        if instance_index != len(self.labelNames):
            raise RuntimeError("The number of labels differs from the number of stream instances.")
        return self.resultManager.finish()

    def createModel(self):
        # Cria o classificador registrado usando o schema original da stream e a semente configurada.
        schema = InstanceFactory.getStreamSchema(self.stream)
        definition = ClassifierRegistry.definitions[self.modelCode]
        parameters = dict(definition.defaults)
        parameters.update(self.modelParameters)
        parameters["schema"] = schema
        parameters.setdefault(definition.seedParameter, self.seed)
        ClassifierRegistry.validateParameters(self.modelCode, parameters)
        model_class = definition.loader()
        try:
            self.model = model_class(**parameters)
        except TypeError as error:
            raise TypeError(
                f"Failed to create {self.modelCode} with parameters {parameters}. Check the installed CapyMOA version."
            ) from error
        self.modelName = definition.displayName

    def processInstance(self, rawInstance, instanceIndex):
        # Processa uma instância no protocolo prequential com imputação e normalização causais.
        raw_values = np.asarray(rawInstance.x, dtype=np.float64)
        observed_mask = np.isfinite(raw_values)
        imputed_values = self.imputer.transform(raw_values)
        if np.any(~np.isfinite(imputed_values)):
            raise RuntimeError("The imputer produced non-finite values.")
        normalized_values = self.normalizer.transform(imputed_values)
        if np.any(~np.isfinite(normalized_values)):
            raise RuntimeError("The normalizer produced non-finite values.")
        model_instance = InstanceFactory.createLabeledInstance(rawInstance, normalized_values)
        raw_prediction = self.model.predict(model_instance)
        prediction_available = raw_prediction is not None and int(raw_prediction) in (0, 1)
        predicted_label = int(raw_prediction) if prediction_available else 0
        true_label = int(rawInstance.y_index)
        if true_label not in (0, 1):
            raise ValueError("The classification pipeline accepts only binary labels 0 and 1.")
        is_warmup = instanceIndex < self.initialWarmupSize
        evaluation_ready = bool(not is_warmup)
        metric_values = self.metrics.update(true_label, predicted_label) if evaluation_ready else self.emptyMetricValues()
        attack_probability = self.getAttackProbability(model_instance)
        was_trained = self.trainModel(model_instance)
        row = {
            "instanceId": int(instanceIndex),
            "dataset": self.datasetName,
            "model": self.modelCode,
            "modelConfig": self.modelName,
            "imputer": self.getImputerName(),
            "normalizer": self.getNormalizerName(),
            "trainingStrategy": "all",
            "thresholdStrategy": "none",
            "thresholdScoreSource": "none",
            "thresholdScoreLabel": "none",
            "evaluationName": f"{self.modelCode}_binary_prequential_{self.getImputerName()}",
            "warmup": self.initialWarmupSize,
            "modelWarmup": self.initialWarmupSize,
            "thresholdWarmup": 0,
            "isWarmup": int(is_warmup),
            "evaluationReady": int(evaluation_ready),
            "thresholdCalibrationWindow": 0,
            "thresholdCalibrationStart": self.initialWarmupSize,
            "thresholdReady": 1,
            "trueLabel": true_label,
            "labelName": str(self.labelNames[instanceIndex]),
            "isAttack": true_label,
            "predictedLabel": predicted_label,
            "predictionAvailable": int(prediction_available),
            "attackProbability": attack_probability,
            "trainingAllowed": 1,
            "wasTrained": int(bool(was_trained)),
            "missingFeatureCount": int(np.sum(~observed_mask)),
            **metric_values,
        }
        self.imputer.update(raw_values)
        self.normalizer.update(imputed_values, observedMask=observed_mask)
        return row

    def trainModel(self, modelInstance):
        # Treina o classificador com a instância atual após a predição prequential.
        try:
            self.model.train(modelInstance)
            return True
        except ValueError:
            if self.strictTrainingErrors:
                raise
            return False

    def getAttackProbability(self, modelInstance):
        # Obtém a probabilidade da classe ATTACK quando o classificador disponibilizar essa informação.
        try:
            probabilities = self.model.predict_proba(modelInstance)
        except (AttributeError, NotImplementedError, ValueError):
            return np.nan
        if probabilities is None or len(probabilities) < 2:
            return np.nan
        value = float(probabilities[1])
        return value if np.isfinite(value) else np.nan

    def getNormalizerName(self):
        # Retorna o nome padronizado do normalizador utilizado na execução.
        return getattr(self.normalizer, "name", self.normalizer.__class__.__name__)

    def getImputerName(self):
        # Retorna o nome padronizado do imputador utilizado na execução.
        return getattr(self.imputer, "name", self.imputer.__class__.__name__)

    def emptyMetricValues(self):
        # Retorna métricas vazias antes do início da avaliação prequential.
        return {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "cumulativeTp": self.metrics.tp,
            "cumulativeTn": self.metrics.tn,
            "cumulativeFp": self.metrics.fp,
            "cumulativeFn": self.metrics.fn,
            "cumulativeAccuracy": 0.0,
            "cumulativePrecision": 0.0,
            "cumulativeRecall": 0.0,
            "cumulativeSpecificity": 0.0,
            "cumulativeF1": 0.0,
            "cumulativeMcc": 0.0,
        }

    def resetComponents(self):
        # Reinicia stream, pré-processamento e métricas antes de cada execução completa.
        if hasattr(self.imputer, "reset"):
            self.imputer.reset()
        if hasattr(self.normalizer, "reset"):
            self.normalizer.reset()
        self.metrics.reset()
