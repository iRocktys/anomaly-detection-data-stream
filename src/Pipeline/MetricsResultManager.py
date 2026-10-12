from copy import deepcopy

from src.Metrics.IncrementalMetrics import IncrementalMetrics
from src.Metrics.Metrics import Metrics
from src.Pipeline.ResultContracts import MetricsRunResult, PipelineRunContext


class MetricsResultManager:
    def __init__(self):
        # Inicializa um coletor leve usado durante otimizações sem gravar artefatos por trial.
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
        # Reinicia o estado acumulado para uma nova avaliação em memória.
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
        # Atualiza métricas binárias somente para linhas marcadas como avaliáveis pelo pipeline.
        if self.context is None:
            raise RuntimeError("The result manager must be started before collection.")
        self.totalInstances += 1
        if not bool(row.get("evaluationReady", False)):
            return
        if self.firstEvaluatedRow is None:
            self.firstEvaluatedRow = deepcopy(row)
        instance_id = int(row["instanceId"])
        if self.windowStart is None:
            self.windowStart = instance_id
        true_label = int(row["isAttack"])
        predicted_label = int(row["predictedLabel"])
        self.attackInstances += true_label
        self.globalMetrics.update(true_label, predicted_label)
        self.windowMetrics.update(true_label, predicted_label)
        self.lastEvaluatedInstance = instance_id
        if self.windowMetrics.instances >= self.context.metricsWindowSize:
            self._closeWindow(instance_id)

    def finish(self):
        # Retorna métricas globais e janeladas sem persistir arquivos durante trials.
        if self.context is None:
            raise RuntimeError("No execution was started.")
        if self.firstEvaluatedRow is None:
            raise ValueError("No instances are available to calculate metrics.")
        if self.windowMetrics.instances:
            self._closeWindow(self.lastEvaluatedInstance)
        snapshot = self.globalMetrics.snapshot()
        evaluated_instances = int(snapshot.pop("instances"))
        first = self.firstEvaluatedRow
        stream_metrics = {
            "task": str(first.get("task", self.context.task)),
            "dataset": str(first["dataset"]),
            "model": str(first["model"]),
            "evaluationName": str(first["evaluationName"]),
            "warmup": int(first.get("warmup", 0)),
            "totalInstances": int(self.totalInstances),
            "evaluationStart": int(first["instanceId"]),
            "evaluationEnd": int(self.lastEvaluatedInstance),
            "evaluatedInstances": evaluated_instances,
            "benignInstances": evaluated_instances - self.attackInstances,
            "attackInstances": int(self.attackInstances),
            "attackRatioPercent": Metrics.safeDivide(self.attackInstances, evaluated_instances) * 100.0,
            **snapshot,
        }
        result = MetricsRunResult(streamMetrics=stream_metrics, windowMetrics=tuple(deepcopy(self.windows)))
        self.context = None
        return result

    def _closeWindow(self, windowEnd):
        # Fecha a janela corrente e anexa também as métricas acumuladas até seu final.
        window_snapshot = self.windowMetrics.snapshot()
        cumulative_snapshot = self.globalMetrics.snapshot()
        self.windows.append({
            "windowSize": int(self.context.metricsWindowSize),
            "windowIndex": len(self.windows),
            "windowStart": int(self.windowStart),
            "windowEnd": int(windowEnd),
            **window_snapshot,
            "cumulativeInstances": cumulative_snapshot["instances"],
            "cumulativeTp": cumulative_snapshot["tp"],
            "cumulativeTn": cumulative_snapshot["tn"],
            "cumulativeFp": cumulative_snapshot["fp"],
            "cumulativeFn": cumulative_snapshot["fn"],
            "cumulativeAccuracy": cumulative_snapshot["accuracy"],
            "cumulativePrecision": cumulative_snapshot["precision"],
            "cumulativeRecall": cumulative_snapshot["recall"],
            "cumulativeSpecificity": cumulative_snapshot["specificity"],
            "cumulativeF1": cumulative_snapshot["f1"],
            "cumulativeMcc": cumulative_snapshot["mcc"],
        })
        self.windowMetrics.reset()
        self.windowStart = None
