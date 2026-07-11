import os
import unicodedata
from datetime import datetime

import pandas as pd

from src.Data.Processor import DataStreamProcessor
from src.Pipeline.ExperimentBuilder import ExperimentBuilder
from src.Pipeline.StreamPipeline import StreamPipeline


class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.builder = ExperimentBuilder()

    def runExperiment(self, runId=None):
        self.config.validate()
        runId = runId or datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(self.config.outputDirectory, exist_ok=True)
        generatedFiles = []

        for modelCode in self.config.modelCodes:
            for normalizerConfig in self.config.normalizers:
                stream, featureNames = self.createStream()

                if self.config.printSummary:
                    self.printRoundSummary(modelCode, normalizerConfig)

                components = self.builder.buildComponents(
                    config=self.config,
                    schema=stream.get_schema(),
                    modelCode=modelCode,
                    normalizerConfig=normalizerConfig,
                )
                pipeline = StreamPipeline(self.config, components, featureNames)
                rows = pipeline.run(stream, runId, modelCode, normalizerConfig.name)
                outputPath = self.saveRows(rows, modelCode, normalizerConfig.name)
                generatedFiles.append(outputPath)
        return generatedFiles

    def createStream(self):
        dataFrame = pd.read_csv(self.config.datasetPath)
        processor = DataStreamProcessor(
            logging=False,
            selected_features=self.config.selectedFeatures,
        )
        stream, targets, featureNames = processor.create_stream(
            df=dataFrame,
            target_label_col=self.config.targetLabelColumn,
            binary_label=False,
            normalize_method=None,
            threshold_var=None,
            threshold_corr=None,
            top_n_features=None,
            return_stream=True,
            extra_ignore_cols=self.config.ignoredColumns,
            imputation_method=self.config.imputationMethod,
        )
        return stream, featureNames

    def saveRows(self, rows, modelCode, normalizerName):
        fileName = f"{self.safeName(self.config.datasetName)}-{self.safeName(modelCode)}-{self.safeName(normalizerName)}.csv"
        outputPath = os.path.join(self.config.outputDirectory, fileName)
        pd.DataFrame(rows).to_csv(outputPath, index=False)
        return outputPath

    def printRoundSummary(self, modelCode, normalizerConfig):
        print("\n" + "=" * 72)
        print(f"Dataset: {self.config.datasetName}")
        print(f"Modelo: {modelCode}")
        print(f"Features selecionadas: {len(self.config.selectedFeatures)}")
        print(f"Seletor: {self.config.featureSelector.name}")
        print(f"Extrator: {self.config.featureExtractor.name}")
        print(f"Normalizador: {normalizerConfig.name} {normalizerConfig.parameters}")
        print(f"Suavização de features: {self.config.featureSmoother.name}")
        print(f"Suavização de score: {self.config.scoreSmoother.name}")
        print(f"Threshold: {self.config.thresholdStrategy.name} {self.config.thresholdStrategy.parameters}")
        print(f"Decisão: {self.config.decisionStrategy.name}")
        print(f"Treinamento: {self.config.trainingStrategy.name}")
        print(f"Atualização do normalizador: {self.config.normalizerUpdatePolicy}")
        print("=" * 72)

    def safeName(self, value):
        normalizedValue = unicodedata.normalize("NFKD", str(value))
        asciiValue = normalizedValue.encode("ascii", "ignore").decode("ascii")
        return "".join(character if character.isalnum() or character in {"-", "."} else "-" for character in asciiValue)
