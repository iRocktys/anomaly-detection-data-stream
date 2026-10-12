from copy import deepcopy

from src.Metrics.IncrementalMetrics import IncrementalMetrics
from src.Metrics.Metrics import Metrics
from src.Pipeline.ResultContracts import MetricsRunResult, PipelineRunContext


class MetricsResultManager:
    def __init__(self):
        # Inicializa o estado e as dependências necessárias para a execução da classe.
        self.context = None
        self.globalMetrics = IncrementalMetrics()
        self.windowMetrics = IncrementalMetrics()
        self.windows = []
        self.totalInstances = 0
        self.attackInstances = 0
        self.firstEvaluatedRow = None
        self.lastEvaluatedInstance = None
        self.windowStart = None

    def start(self, context: PipelineRunContext):
        # Inicia o estado necessário para uma nova execução ou coleta de resultados.
        self.context = context
        self.globalMetrics.reset()
        self.windowMetrics.reset()
        self.windows = []
        self.totalInstances = 0
        self.attackInstances = 0
        self.firstEvaluatedRow = None
        self.lastEvaluatedInstance = None
        self.windowStart = None

    def collect(self, row):
        # Coleta uma nova linha produzida durante o processamento da stream.
        if self.context is None:
            raise RuntimeError(
                "The result manager must be started before collection."
            )

        self.totalInstances += 1
        if not bool(row.get("evaluationReady", False)):
            return

        if self.firstEvaluatedRow is None:
            self.firstEvaluatedRow = deepcopy(row)

        instanceId = int(row["instanceId"])
        if self.windowStart is None:
            self.windowStart = instanceId

        trueLabel = int(row["isAttack"])
        predictedLabel = int(row["predictedLabel"])
        self.attackInstances += trueLabel
        self.globalMetrics.update(trueLabel, predictedLabel)
        self.windowMetrics.update(trueLabel, predictedLabel)
        self.lastEvaluatedInstance = instanceId

        if self.windowMetrics.instances >= self.context.metricsWindowSize:
            self._closeWindow(instanceId)

    def finish(self):
        # Finaliza o processamento atual e retorna os resultados consolidados.
        if self.context is None:
            raise RuntimeError("No execution has been started.")
        if self.firstEvaluatedRow is None:
            raise ValueError(
                "No instances are available after warm-up to calculate metrics."
            )
        if self.windowMetrics.instances:
            self._closeWindow(self.lastEvaluatedInstance)

        globalSnapshot = self.globalMetrics.snapshot()
        evaluatedInstances = int(globalSnapshot.pop("instances"))
        first = self.firstEvaluatedRow
        streamMetrics = {
            "dataset": str(first["dataset"]),
            "model": str(first["model"]),
            "modelConfig": str(first["modelConfig"]),
            "imputer": str(first["imputer"]),
            "normalizer": str(first["normalizer"]),
            "trainingStrategy": str(first["trainingStrategy"]),
            "thresholdStrategy": str(first["thresholdStrategy"]),
            "thresholdScoreSource": str(first["thresholdScoreSource"]),
            "thresholdScoreLabel": str(first["thresholdScoreLabel"]),
            "evaluationName": str(first["evaluationName"]),
            "warmup": int(first["warmup"]),
            "modelWarmup": int(
                first.get(
                    "modelWarmup",
                    first["thresholdCalibrationStart"],
                )
            ),
            "thresholdWarmup": int(
                first.get(
                    "thresholdWarmup",
                    first["thresholdCalibrationWindow"],
                )
            ),
            "thresholdCalibrationWindow": int(
                first["thresholdCalibrationWindow"]
            ),
            "thresholdCalibrationStart": int(
                first["thresholdCalibrationStart"]
            ),
            "totalInstances": int(self.totalInstances),
            "evaluationStart": int(first["instanceId"]),
            "evaluationEnd": int(self.lastEvaluatedInstance),
            "evaluatedInstances": evaluatedInstances,
            "benignInstances": evaluatedInstances - self.attackInstances,
            "attackInstances": int(self.attackInstances),
            "attackRatioPercent": Metrics.safeDivide(
                self.attackInstances,
                evaluatedInstances,
            )
            * 100.0,
            **globalSnapshot,
        }
        result = MetricsRunResult(
            streamMetrics=streamMetrics,
            windowMetrics=tuple(deepcopy(self.windows)),
        )
        self.context = None
        return result

    def _closeWindow(self, windowEnd):
        # Fecha a janela de métricas atual e armazena seus valores consolidados.
        windowSnapshot = self.windowMetrics.snapshot()
        cumulativeSnapshot = self.globalMetrics.snapshot()
        window = {
            "windowSize": int(self.context.metricsWindowSize),
            "windowIndex": len(self.windows),
            "windowStart": int(self.windowStart),
            "windowEnd": int(windowEnd),
            **windowSnapshot,
            "cumulativeInstances": cumulativeSnapshot["instances"],
            "cumulativeTp": cumulativeSnapshot["tp"],
            "cumulativeTn": cumulativeSnapshot["tn"],
            "cumulativeFp": cumulativeSnapshot["fp"],
            "cumulativeFn": cumulativeSnapshot["fn"],
            "cumulativeAccuracy": cumulativeSnapshot["accuracy"],
            "cumulativePrecision": cumulativeSnapshot["precision"],
            "cumulativeRecall": cumulativeSnapshot["recall"],
            "cumulativeSpecificity": cumulativeSnapshot["specificity"],
            "cumulativeF1": cumulativeSnapshot["f1"],
            "cumulativeMcc": cumulativeSnapshot["mcc"],
        }
        self.windows.append(window)
        self.windowMetrics.reset()
        self.windowStart = None
