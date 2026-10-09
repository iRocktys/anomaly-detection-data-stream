import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.Pipeline.ResultContracts import PipelineRunContext
from src.Pipeline.ResultFrameBuilder import ResultFrameBuilder
from src.Plots.Plots import Plots


class ResultManager:
    # Inicializa o gerenciador responsável por armazenar métricas, artefatos e diretórios de cada execução.
    def __init__(self, outputPath="output"):
        self.outputPath = Path(outputPath)
        self.frameBuilder = ResultFrameBuilder()
        self.context = None
        self.rows = []

    # Inicia uma nova execução e limpa qualquer estado acumulado anteriormente.
    def start(self, context: PipelineRunContext):
        self.context = context
        self.rows = []

    # Armazena uma linha de resultado produzida durante o processamento incremental da stream.
    def collect(self, row):
        if self.context is None:
            raise RuntimeError(
                "O gerenciador de resultados deve ser iniciado antes da coleta."
            )

        self.rows.append(row)

    # Finaliza a execução atual e encaminha os resultados acumulados para persistência.
    def finish(self):
        if self.context is None:
            raise RuntimeError(
                "Nenhuma execução foi iniciada."
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

    # Salva resultados por instância, por janela e globais, além de gerar os gráficos configurados.
    def save(
        self,
        rows,
        datasetName,
        modelCode,
        windowSize,
        movingAverageColumns=None,
        generatePlots=True,
    ):
        if not rows:
            raise ValueError(
                "Não existem resultados para salvar."
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

    # Cria uma pasta numerada para a execução usando o nome lógico do dataset e a configuração aplicada.
    def createRunDirectory(
        self,
        modelCode,
        datasetName,
        trainingName,
        thresholdName,
        thresholdScoreName=None,
        imputerName=None,
    ):
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

    # Sanitiza somente o nome lógico informado para o dataset sem aplicar aliases específicos de cenários.
    def normalizeDatasetName(
        self,
        datasetName,
    ):
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
                "O nome do dataset não pode ser vazio."
            )

        return safeName

    # Normaliza o nome da estratégia de treinamento para uso consistente nos diretórios de saída.
    def normalizeTrainingName(
        self,
        trainingStrategy,
    ):
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

    # Normaliza o nome da estratégia de threshold para uso consistente nos artefatos de execução.
    def normalizeThresholdName(
        self,
        thresholdStrategy,
    ):
        value = str(
            thresholdStrategy
        ).strip().lower()

        if value == "fixed":
            return "FIXED"

        if value == "dspot":
            return "DSPOT"

        return value.upper()

    # Normaliza a origem do score usada pelo threshold para diferenciar score bruto e médias móveis.
    def normalizeThresholdScoreName(
        self,
        thresholdScoreSource,
    ):
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

        return value.upper()

    # Normaliza o nome do imputador para manter nomes curtos e consistentes nas pastas de resultado.
    def normalizeImputerName(
        self,
        imputer,
    ):
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

    # Constrói a tabela de métricas por janela a partir dos resultados individuais da stream.
    def buildWindowFrame(
        self,
        instanceFrame,
        windowSize,
    ):
        return self.frameBuilder.buildWindowFrame(
            instanceFrame,
            windowSize,
        )

    # Constrói as métricas globais da execução a partir das instâncias avaliadas.
    def buildStreamMetricsFrame(
        self,
        instanceFrame,
    ):
        return self.frameBuilder.buildStreamMetricsFrame(
            instanceFrame
        )

    # Retorna somente as instâncias pertencentes à região efetivamente avaliada da stream.
    def selectEvaluatedFrame(
        self,
        instanceFrame,
    ):
        return self.frameBuilder.selectEvaluatedFrame(
            instanceFrame
        )

    # Gera os gráficos de scores, erros e métricas janeladas associados à execução atual.
    def generatePlots(
        self,
        instancePath,
        windowPath,
        plotDirectory,
        movingAverageColumns,
        windowSize,
    ):
        plotter = Plots()

        movingAverageColumns = list(
            movingAverageColumns
            or []
        )

        scoreLabels = [
            (
                "Média móvel "
                f"({column.replace('scoreMa', '')})"
            )
            for column in movingAverageColumns
        ]

        scoreColors = [
            "#5f86ad",
            "#f0a43a",
            "#6f2dbd",
            "#2a9d8f",
        ]

        scorePath = plotter.plotScoreArtifact(
            instancePath,
            outputPath=(
                plotDirectory
                / "scores.png"
            ),
            movingAverageColumns=movingAverageColumns,
            movingAverageLabels=scoreLabels,
            movingAverageColors=scoreColors,
            showWarmup=True,
            attackAlpha=0.30,
            legendColumns=8,
        )

        errorPath = plotter.plotWindowErrors(
            windowPath,
            attackSource=instancePath,
            outputPath=(
                plotDirectory
                / "fp_fn_windows.png"
            ),
            windowSize=windowSize,
            attackAlpha=0.30,
            legendColumns=8,
        )

        metricsPath = plotter.plotWindowMetrics(
            windowPath,
            attackSource=instancePath,
            outputPath=(
                plotDirectory
                / "metrics_windows.png"
            ),
            windowSize=windowSize,
            attackAlpha=0.30,
            legendColumns=8,
        )

        return {
            "scores": scorePath,
            "errors": errorPath,
            "metrics": metricsPath,
        }
