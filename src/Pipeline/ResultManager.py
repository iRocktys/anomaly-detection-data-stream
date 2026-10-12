import json
import re
import shutil
import unicodedata
from pathlib import Path

import pandas as pd

from src.Pipeline.ResultContracts import PipelineRunContext
from src.Pipeline.ResultFrameBuilder import ResultFrameBuilder
from src.Plots.PlotManager import PlotManager


class ResultManager:
    def __init__(self, outputPath="output", fixedRunDirectory=None, overwriteFixedDirectory=False):
        # Inicializa o persistidor comum de resultados para detecção e classificação binárias.
        self.outputPath = Path(outputPath)
        self.fixedRunDirectory = Path(fixedRunDirectory) if fixedRunDirectory is not None else None
        self.overwriteFixedDirectory = bool(overwriteFixedDirectory)
        self.frameBuilder = ResultFrameBuilder()
        self.context = None
        self.rows = []

    def start(self, context: PipelineRunContext):
        # Inicia uma nova execução e descarta qualquer resultado acumulado anteriormente.
        self.context = context
        self.rows = []

    def collect(self, row):
        # Armazena uma linha produzida pelo pipeline para materialização ao final da stream.
        if self.context is None:
            raise RuntimeError("The result manager must be started before collection.")
        self.rows.append(dict(row))

    def finish(self):
        # Finaliza a execução atual e salva CSVs, metadados e gráficos em uma única estrutura.
        if self.context is None:
            raise RuntimeError("No execution was started.")
        context = self.context
        rows = self.rows
        self.context = None
        self.rows = []
        return self.save(rows, context)

    def save(self, rows, context):
        # Persiste os artefatos de uma execução sem duplicar regras entre tarefas.
        if not rows:
            raise ValueError("There are no results to save.")
        instance_frame = pd.DataFrame(rows)
        run_directory = self.createRunDirectory(context, instance_frame)
        plot_directory = run_directory / "plots"
        plot_directory.mkdir(parents=True, exist_ok=True)
        instance_path = run_directory / "instances.csv"
        window_path = run_directory / "windows.csv"
        stream_metrics_path = run_directory / "stream_metrics.csv"
        metadata_path = run_directory / "run_metadata.json"
        window_frame = self.frameBuilder.buildWindowFrame(instance_frame, context.metricsWindowSize)
        stream_metrics_frame = self.frameBuilder.buildStreamMetricsFrame(instance_frame)
        instance_frame.to_csv(instance_path, index=False)
        window_frame.to_csv(window_path, index=False)
        stream_metrics_frame.to_csv(stream_metrics_path, index=False)
        metadata = {
            "task": context.task,
            "dataset": context.datasetName,
            "model": context.modelCode,
            "metricsWindowSize": context.metricsWindowSize,
            "movingAverageColumns": list(context.movingAverageColumns),
            "columns": list(instance_frame.columns),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        plot_paths = {}
        if context.generatePlots:
            plot_paths = PlotManager().generate(
                task=context.task,
                instanceSource=instance_path,
                windowSource=window_path,
                outputDirectory=plot_directory,
                movingAverageColumns=context.movingAverageColumns,
                windowSize=context.metricsWindowSize,
            )
        return {
            "runDirectory": str(run_directory),
            "instancePath": str(instance_path),
            "windowPath": str(window_path),
            "streamMetricsPath": str(stream_metrics_path),
            "metadataPath": str(metadata_path),
            "plotDirectory": str(plot_directory),
            "plotPaths": plot_paths,
            "instanceFrame": instance_frame,
            "windowFrame": window_frame,
            "streamMetricsFrame": stream_metrics_frame,
        }

    def createRunDirectory(self, context, instanceFrame):
        # Cria uma pasta numerada estável ou reutiliza o diretório fixo solicitado pela otimização.
        if self.fixedRunDirectory is not None:
            run_directory = self.fixedRunDirectory
            if run_directory.exists() and self.overwriteFixedDirectory:
                shutil.rmtree(run_directory)
            run_directory.mkdir(parents=True, exist_ok=True)
            return run_directory
        model_directory = self.outputPath / self.normalizeName(context.modelCode)
        dataset_directory = model_directory / self.normalizeName(context.datasetName)
        dataset_directory.mkdir(parents=True, exist_ok=True)
        existing_numbers = []
        for path in dataset_directory.iterdir():
            if path.is_dir() and path.name.split("-", 1)[0].isdigit():
                existing_numbers.append(int(path.name.split("-", 1)[0]))
        descriptor = self.buildRunDescriptor(context.task, instanceFrame)
        run_name = f"{max(existing_numbers, default=0) + 1:03d}-{descriptor}"
        run_directory = dataset_directory / run_name
        run_directory.mkdir(parents=True, exist_ok=False)
        return run_directory

    def buildRunDescriptor(self, task, instanceFrame):
        # Monta um identificador curto de execução com os componentes relevantes de cada tarefa.
        parts = []
        if "trainingStrategy" in instanceFrame.columns:
            parts.append(self.normalizeName(instanceFrame["trainingStrategy"].iloc[0]))
        if str(task).lower() == "anomaly":
            for column in ("thresholdStrategy", "thresholdScoreSource"):
                if column in instanceFrame.columns:
                    parts.append(self.normalizeName(instanceFrame[column].iloc[0]))
        if "imputer" in instanceFrame.columns:
            parts.append(self.normalizeName(instanceFrame["imputer"].iloc[0]))
        return "-".join(parts) if parts else "run"

    def normalizeName(self, value):
        # Sanitiza nomes lógicos para utilização segura em caminhos de saída.
        ascii_name = unicodedata.normalize("NFKD", str(value).strip()).encode("ascii", "ignore").decode("ascii")
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_")
        if not safe_name:
            raise ValueError("The name used in the output path cannot be empty.")
        return safe_name
