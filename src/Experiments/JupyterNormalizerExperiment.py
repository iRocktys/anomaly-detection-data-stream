from pathlib import Path

from src.Experiments.NormalizerBehaviorExperiment import buildExperiment
from src.Pipeline.ExperimentRunner import ExperimentRunner


def runJupyterExperiment(
    datasetPath,
    selectedFeatures,
    datasetName="Adaptacao",
    outputDirectory="output/ExpNormalizers",
    modelCodes=None,
    rollingWindow=200,
    normalizerNames=None,
    normalizerUpdatePolicy="all",
    trainingStrategy="all",
    thresholdName="fixed",
    thresholdParameters=None,
    saveNormalizedFeatures=True,
    printSummary=True,
):
    """Executa todas as rodadas do experimento a partir de uma única célula.

    A função recebe apenas parâmetros simples, monta a configuração completa,
    executa as combinações de modelos e normalizadores e devolve os caminhos
    dos CSVs gerados.
    """
    resolvedDatasetPath = str(Path(datasetPath).expanduser())
    resolvedOutputDirectory = str(Path(outputDirectory).expanduser())

    experimentConfig = buildExperiment(
        datasetPath=resolvedDatasetPath,
        selectedFeatures=list(selectedFeatures),
        datasetName=datasetName,
        outputDirectory=resolvedOutputDirectory,
        modelCodes=list(modelCodes or ["AIF", "HST"]),
        rollingWindow=rollingWindow,
    )

    selectedNormalizerNames = set(normalizerNames or [
        "none",
        "incrementalMinMax",
        "incrementalZScore",
        "rollingMinMax",
        "rollingZScore",
    ])

    experimentConfig = experimentConfig.withChanges(
        normalizers=[
            normalizerConfig
            for normalizerConfig in experimentConfig.normalizers
            if normalizerConfig.name in selectedNormalizerNames
        ],
        normalizerUpdatePolicy=normalizerUpdatePolicy,
        trainingStrategy=experimentConfig.trainingStrategy.__class__(trainingStrategy),
        thresholdStrategy=experimentConfig.thresholdStrategy.__class__(
            thresholdName,
            dict(thresholdParameters or {"value": 0.5}),
        ),
        saveNormalizedFeatures=saveNormalizedFeatures,
        printSummary=printSummary,
    )

    generatedFiles = ExperimentRunner(experimentConfig).runExperiment()

    print("\nArquivos gerados:")
    for generatedFile in generatedFiles:
        print(generatedFile)

    return generatedFiles
