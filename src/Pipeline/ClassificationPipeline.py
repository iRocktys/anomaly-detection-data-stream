import numpy as np

from src.Classification.Models import ClassifierRegistry
from src.Metrics.IncrementalMetrics import IncrementalMetrics
from src.Pipeline.InstanceFactory import InstanceFactory
from src.Pipeline.BinaryPipelineSupport import BinaryPipelineSupport
from src.Pipeline.ResultContracts import PipelineRunContext, ResultManagerProtocol
from src.Pipeline.StreamPreprocessor import StreamPreprocessor


class ClassificationPipeline(BinaryPipelineSupport):
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
        seed=42,
        generatePlots=True,
        initialWarmupSize=0,
        resultManager=None,
        strictTrainingErrors=False,
    ):
        # Inicializa o pipeline supervisionado binário no protocolo prequential prediz-depois-treina.
        self.stream = stream
        self.datasetName = str(datasetName)
        self.modelCode = ClassifierRegistry.normalizeCode(modelCode)
        self.labelNames = tuple(labelNames)
        self.modelParameters = dict(modelParameters or {})
        self.preprocessor = StreamPreprocessor(imputer, normalizer)
        self.metrics = IncrementalMetrics()
        self.metricsWindowSize = max(1, int(metricsWindowSize))
        self.seed = int(seed)
        self.generatePlots = bool(generatePlots)
        self.initialWarmupSize = max(0, int(initialWarmupSize))
        self.resultManager = resultManager
        self.strictTrainingErrors = bool(strictTrainingErrors)
        if not isinstance(self.resultManager, ResultManagerProtocol):
            raise TypeError("resultManager must implement start(context), collect(row), and finish().")
        if self.initialWarmupSize >= len(self.labelNames):
            raise ValueError("initialWarmupSize must leave at least one instance for evaluation.")
        self.modelName = None
        self.model = None

    def run(self):
        # Percorre a stream uma vez no protocolo prequential e materializa resultados binários.
        self.resetComponents()
        self.stream.restart()
        self.createModel()
        self.resultManager.start(PipelineRunContext(
            task="classification",
            datasetName=self.datasetName,
            modelCode=self.modelCode,
            metricsWindowSize=self.metricsWindowSize,
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
        # Cria o classificador registrado com o schema da stream e a semente da execução.
        schema = InstanceFactory.getStreamSchema(self.stream)
        self.modelName, self.model, self.modelParameters = ClassifierRegistry.create(
            schema=schema,
            code=self.modelCode,
            parameters=self.modelParameters,
            runSeed=self.seed,
        )

    def processInstance(self, rawInstance, instanceIndex):
        # Prediz a classe binária antes de treinar a mesma instância e atualizar o pré-processamento.
        raw_values = np.asarray(rawInstance.x, dtype=np.float64)
        normalized_values, imputed_values, observed_mask = self.preprocessor.transform(raw_values)
        model_instance = InstanceFactory.createLabeledInstance(rawInstance, normalized_values)
        raw_prediction = self.model.predict(model_instance)
        true_label = int(rawInstance.y_index)
        if true_label not in (0, 1):
            raise ValueError("The classification pipeline accepts only binary labels 0 and 1.")
        prediction_available = raw_prediction is not None and int(raw_prediction) in (0, 1)
        is_warmup = instanceIndex < self.initialWarmupSize
        evaluation_ready = bool(prediction_available and not is_warmup)
        predicted_label = int(raw_prediction) if prediction_available else 0
        metric_values = self.metrics.update(true_label, predicted_label) if evaluation_ready else self.emptyMetricValues()
        attack_probability = self.getAttackProbability(model_instance)
        was_trained = self.trainModel(model_instance)
        row = {
            "instanceId": int(instanceIndex),
            "task": "classification",
            "dataset": self.datasetName,
            "model": self.modelCode,
            "modelConfig": self.modelName,
            "imputer": self.preprocessor.getImputerName(),
            "normalizer": self.preprocessor.getNormalizerName(),
            "trainingStrategy": "prequentialAll",
            "evaluationName": f"{self.modelCode}_prequential",
            "warmup": self.initialWarmupSize,
            "isWarmup": int(is_warmup),
            "evaluationReady": int(evaluation_ready),
            "trueLabel": true_label,
            "labelName": self.labelNames[instanceIndex],
            "isAttack": true_label,
            "predictedLabel": predicted_label,
            "predictionAvailable": int(prediction_available),
            "attackProbability": attack_probability,
            "trainingAllowed": 1,
            "wasTrained": int(bool(was_trained)),
            "missingFeatureCount": int(np.sum(~observed_mask)),
            **metric_values,
        }
        self.preprocessor.update(raw_values, imputed_values, observed_mask)
        return row

    def getAttackProbability(self, modelInstance):
        # Obtém a probabilidade da classe ATTACK quando o classificador disponibiliza probabilidades válidas.
        try:
            probabilities = self.model.predict_proba(modelInstance)
        except (AttributeError, NotImplementedError, ValueError):
            return np.nan
        if probabilities is None or len(probabilities) < 2:
            return np.nan
        value = float(probabilities[1])
        return value if np.isfinite(value) else np.nan

    def resetComponents(self):
        # Reinicia pré-processamento e métricas antes de criar uma nova instância do classificador.
        self.preprocessor.reset()
        self.metrics.reset()
