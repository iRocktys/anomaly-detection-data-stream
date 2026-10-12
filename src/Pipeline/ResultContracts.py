from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class PipelineRunContext:
    datasetName: str
    modelCode: str
    metricsWindowSize: int
    movingAverageColumns: tuple[str, ...] = ()
    generatePlots: bool = True


@dataclass(frozen=True)
class MetricsRunResult:
    streamMetrics: dict[str, Any]
    windowMetrics: tuple[dict[str, Any], ...] = field(
        default_factory=tuple
    )


@runtime_checkable
class ResultManagerProtocol(Protocol):
    def start(self, context: PipelineRunContext) -> None:
        # Inicia o estado necessário para uma nova execução ou coleta de resultados.
        ...

    def collect(self, row: dict[str, Any]) -> None:
        # Coleta uma nova linha produzida durante o processamento da stream.
        ...

    def finish(self) -> Any:
        # Finaliza o processamento atual e retorna os resultados consolidados.
        ...
