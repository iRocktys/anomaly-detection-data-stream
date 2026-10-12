import re
import shutil
import unicodedata
from pathlib import Path

import pandas as pd

from src.Pipeline.ResultContracts import PipelineRunContext
from src.Pipeline.ResultFrameBuilder import ResultFrameBuilder
from src.Plots.Plots import Plots


class ResultManager:
    def __init__(
        self,
        outputPath="output",
        fixedRunDirectory=None,
        overwriteFixedDirectory=False,
    ):
        # Inicializa o gerenciador e permite opcionalmente salvar uma execução em um diretório final fixo.
        self.outputPath = Path(outputPath)
        self.fixedRunDirectory = (
            Path(fixedRunDirectory)
            if fixedRunDirectory is not None
            else None
        )
        self.overwriteFixedDirectory = bool(
            overwriteFixedDirectory
        )
        self.frameBuilder = ResultFrameBuilder()
        self.context = None
        self.rows = []

    def start(self, context: PipelineRunContext):
        # Inicia uma nova execução e limpa qualquer estado acumulado anteriormente.
        self.context = context
        self.rows = []

    def collect(self, row):
        # Armazena uma linha de resultado produzida durante o processamento incremental da stream.
        if self.context is None:
            raise RuntimeError(
                "The result manager must be started before collection."
            )

        self.rows.append(row)

    def finish(self):
        # Finaliza a execução atual e encaminha os resultados acumulados para persistência.
        if self.context is None:
            raise RuntimeError(
                "No execution has been started."
            )

        context = self.context
        rows = self.rows

        self.context = None
        self.rows = []

        return self.save(
            rows=rows,
            datasetName=context.datasetName,
            modelCode=context.modelCode,
            windowSize=context.metricsWindowSize,
            movingAverageColumns=context.movingAverageColumns,
            generatePlots=context.generatePlots,
        )

    def save(
        self,
        rows,
        datasetName,
        modelCode,
        windowSize,
        movingAverageColumns=None,
        generatePlots=True,
    ):
        # Salva instâncias, métricas janeladas, métricas globais e gráficos usando um único contrato de resultados.
        if not rows:
            raise ValueError(
                "There are no results to save."
            )

        instanceFrame = pd.DataFrame(
            rows
        )

        trainingName = self.normalizeTrainingName(
            instanceFrame[
                "trainingStrategy"
            ].iloc[0]
        )

        thresholdName = self.normalizeThresholdName(
            instanceFrame[
                "thresholdStrategy"
            ].iloc[0]
        )

        thresholdScoreName = self.normalizeThresholdScoreName(
            instanceFrame[
                "thresholdScoreSource"
            ].iloc[0]
        )

        imputerName = self.normalizeImputerName(
            instanceFrame[
                "imputer"
            ].iloc[0]
        )

        runDirectory = self.createRunDirectory(
            modelCode=modelCode,
            datasetName=datasetName,
            trainingName=trainingName,
            thresholdName=thresholdName,
            thresholdScoreName=(
                thresholdScoreName
                if thresholdName == "DSPOT"
                else None
            ),
            imputerName=imputerName,
        )

        plotDirectory = (
            runDirectory
            / "plots"
        )

        plotDirectory.mkdir(
            parents=True,
            exist_ok=True,
        )

        instancePath = (
            runDirectory
            / "instances.csv"
        )

        windowPath = (
            runDirectory
            / "windows.csv"
        )

        streamMetricsPath = (
            runDirectory
            / "stream_metrics.csv"
        )

        windowFrame = self.frameBuilder.buildWindowFrame(
            instanceFrame,
            windowSize,
        )

        streamMetricsFrame = (
            self.frameBuilder.buildStreamMetricsFrame(
                instanceFrame
            )
        )

        instanceFrame.to_csv(
            instancePath,
            index=False,
        )

        windowFrame.to_csv(
            windowPath,
            index=False,
        )

        streamMetricsFrame.to_csv(
            streamMetricsPath,
            index=False,
        )

        plotPaths = {}

        if generatePlots:
            plotPaths = self.generatePlots(
                instancePath=instancePath,
                windowPath=windowPath,
                plotDirectory=plotDirectory,
                movingAverageColumns=movingAverageColumns,
                windowSize=windowSize,
            )

        return {
            "runDirectory": str(
                runDirectory
            ),
            "instancePath": str(
                instancePath
            ),
            "windowPath": str(
                windowPath
            ),
            "streamMetricsPath": str(
                streamMetricsPath
            ),
            "plotDirectory": str(
                plotDirectory
            ),
            "plotPaths": plotPaths,
            "instanceFrame": instanceFrame,
            "windowFrame": windowFrame,
            "streamMetricsFrame": streamMetricsFrame,
        }

    def createRunDirectory(
        self,
        modelCode,
        datasetName,
        trainingName,
        thresholdName,
        thresholdScoreName=None,
        imputerName=None,
    ):
        # Cria a pasta normal numerada ou utiliza o diretório final fixo solicitado pela otimização.
        if self.fixedRunDirectory is not None:
            runDirectory = self.fixedRunDirectory

            if (
                runDirectory.exists()
                and self.overwriteFixedDirectory
            ):
                shutil.rmtree(
                    runDirectory
                )

            runDirectory.mkdir(
                parents=True,
                exist_ok=True,
            )

            return runDirectory

        modelDirectory = (
            self.outputPath
            / str(
                modelCode
            ).strip().upper()
        )

        modelDirectory.mkdir(
            parents=True,
            exist_ok=True,
        )

        datasetDirectory = (
            modelDirectory
            / self.normalizeDatasetName(
                datasetName
            )
        )

        datasetDirectory.mkdir(
            parents=True,
            exist_ok=True,
        )

        existingNumbers = []

        for path in datasetDirectory.iterdir():
            if not path.is_dir():
                continue

            prefix = path.name.split(
                "-",
                1,
            )[0]

            if prefix.isdigit():
                existingNumbers.append(
                    int(
                        prefix
                    )
                )

        directoryParts = [
            f"{max(existingNumbers, default=0) + 1:03d}",
            trainingName,
            thresholdName,
        ]

        if thresholdScoreName:
            directoryParts.append(
                thresholdScoreName
            )

        if imputerName:
            directoryParts.append(
                imputerName
            )

        runDirectory = (
            datasetDirectory
            / "-".join(
                directoryParts
            )
        )

        runDirectory.mkdir(
            parents=True,
            exist_ok=False,
        )

        return runDirectory

    def normalizeDatasetName(
        self,
        datasetName,
    ):
        # Sanitiza somente o nome lógico informado para o dataset sem aplicar aliases específicos de cenários.
        datasetStem = Path(
            str(
                datasetName
            ).strip()
        ).stem

        asciiName = (
            unicodedata.normalize(
                "NFKD",
                datasetStem,
            )
            .encode(
                "ascii",
                "ignore",
            )
            .decode(
                "ascii"
            )
        )

        safeName = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            asciiName,
        ).strip(
            "_"
        )

        if not safeName:
            raise ValueError(
                "The dataset name cannot be empty."
            )

        return safeName

    def normalizeTrainingName(
        self,
        trainingStrategy,
    ):
        # Normaliza o nome da estratégia de treinamento para uso consistente nos diretórios de saída.
        value = str(
            trainingStrategy
        ).strip().lower()

        if value == "all":
            return "ALL"

        if value in {
            "belowthreshold",
            "predicttrue",
            "predict_true",
        }:
            return "PREDICT_TRUE"

        return value.upper()

    def normalizeThresholdName(
        self,
        thresholdStrategy,
    ):
        # Normaliza o nome da estratégia de threshold para uso consistente nos artefatos de execução.
        value = str(
            thresholdStrategy
        ).strip().lower()

        if value == "fixed":
            return "FIXED"

        if value == "dspot":
            return "DSPOT"

        return value.upper()

    def normalizeThresholdScoreName(
        self,
        thresholdScoreSource,
    ):
        # Normaliza a origem do score usada pelo threshold para diferenciar score bruto e médias móveis.
        value = str(
            thresholdScoreSource
        ).strip()

        if value.lower() == "raw":
            return "RAW"

        if value.lower().startswith(
            "scorema"
        ):
            return (
                f"MA"
                f"{value[len('scoreMa'):]}"
            )

        if value.lower().startswith(
            "ma"
        ):
            return value.upper()

        return value.upper()

    def normalizeImputerName(
        self,
        imputer,
    ):
        # Normaliza o nome do imputador para manter nomes curtos e consistentes nas pastas de resultado.
        value = (
            str(
                imputer
            )
            .strip()
            .lower()
            .replace(
                "_",
                "",
            )
            .replace(
                "-",
                "",
            )
        )

        if value in {
            "incrementalmean",
            "mean",
            "media",
            "média",
        }:
            return "MEAN"

        return value.upper()

    def buildWindowFrame(
        self,
        instanceFrame,
        windowSize,
    ):
        # Constrói a tabela de métricas por janela a partir dos resultados individuais da stream.
        return self.frameBuilder.buildWindowFrame(
            instanceFrame,
            windowSize,
        )

    def buildStreamMetricsFrame(
        self,
        instanceFrame,
    ):
        # Constrói as métricas globais da execução a partir das instâncias avaliadas.
        return self.frameBuilder.buildStreamMetricsFrame(
            instanceFrame
        )

    def selectEvaluatedFrame(
        self,
        instanceFrame,
    ):
        # Retorna somente as instâncias pertencentes à região efetivamente avaliada da stream.
        return self.frameBuilder.selectEvaluatedFrame(
            instanceFrame
        )

    def generatePlots(
        self,
        instancePath,
        windowPath,
        plotDirectory,
        movingAverageColumns,
        windowSize,
    ):
        # Gera somente os gráficos compatíveis com as colunas produzidas pela tarefa atual.
        plotter = Plots()
        instanceFrame = pd.read_csv(instancePath, nrows=5)
        movingAverageColumns = list(movingAverageColumns or [])
        plotPaths = {}

        if {"rawScore", "score", "threshold"}.issubset(instanceFrame.columns):
            scoreLabels = [f"Moving Average ({column.replace('scoreMa', '')})" for column in movingAverageColumns]
            scoreColors = ["#5f86ad", "#f0a43a", "#6f2dbd", "#2a9d8f"]
            plotPaths["scores"] = plotter.plotScoreArtifact(
                instancePath,
                outputPath=plotDirectory / "scores.png",
                movingAverageColumns=movingAverageColumns,
                movingAverageLabels=scoreLabels,
                movingAverageColors=scoreColors,
                showWarmup=True,
                attackAlpha=0.30,
                legendColumns=8,
            )

        plotPaths["errors"] = plotter.plotWindowErrors(
            windowPath,
            attackSource=instancePath,
            outputPath=plotDirectory / "fp_fn_windows.png",
            windowSize=windowSize,
            attackAlpha=0.30,
            legendColumns=8,
        )
        plotPaths["metrics"] = plotter.plotWindowMetrics(
            windowPath,
            attackSource=instancePath,
            outputPath=plotDirectory / "metrics_windows.png",
            windowSize=windowSize,
            attackAlpha=0.30,
            legendColumns=8,
        )
        plotPaths["confusionMatrix"] = plotter.plotConfusionMatrix(
            instancePath,
            outputPath=plotDirectory / "confusion_matrix.png",
        )
        return plotPaths
