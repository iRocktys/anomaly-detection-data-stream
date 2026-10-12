from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class PipelineRunContext:
    task: str
    datasetName: str
    modelCode: str
    metricsWindowSize: int
    movingAverageColumns: tuple[str, ...] = ()
    generatePlots: bool = True


@dataclass(frozen=True)
class MetricsRunResult:
    streamMetrics: dict[str, Any]
    windowMetrics: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@runtime_checkable
class ResultManagerProtocol(Protocol):
    def start(self, context: PipelineRunContext) -> None:
        # Define o contrato de inicialização de um coletor de resultados.
        ...

    def collect(self, row: dict[str, Any]) -> None:
        # Define o contrato de coleta incremental de uma linha de resultado.
        ...

    def finish(self) -> Any:
        # Define o contrato de finalização e materialização dos resultados.
        ...
