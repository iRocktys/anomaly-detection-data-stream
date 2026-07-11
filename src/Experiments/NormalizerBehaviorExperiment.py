import argparse

from src.Pipeline.ExperimentConfig import ComponentConfig, ExperimentConfig
from src.Pipeline.ExperimentRunner import ExperimentRunner


def buildExperiment(datasetPath, selectedFeatures, datasetName="Adaptacao", outputDirectory="output/ExpNormalizers", modelCodes=None, rollingWindow=200):
    return ExperimentConfig(
        datasetPath=datasetPath,
        datasetName=datasetName,
        outputDirectory=outputDirectory,
        selectedFeatures=selectedFeatures,
        modelCodes=modelCodes or ["AIF", "HST"],
        normalizers=[
            ComponentConfig("none"),
            ComponentConfig("incrementalMinMax"),
            ComponentConfig("incrementalZScore"),
            ComponentConfig("rollingMinMax", {"windowSize": rollingWindow}),
            ComponentConfig("rollingZScore", {"windowSize": rollingWindow}),
        ],
        featureSelector=ComponentConfig("selected"),
        featureExtractor=ComponentConfig("none"),
        featureSmoother=ComponentConfig("none"),
        scoreSmoother=ComponentConfig("none"),
        thresholdStrategy=ComponentConfig("fixed", {"value": 0.5}),
        decisionStrategy=ComponentConfig("threshold"),
        trainingStrategy=ComponentConfig("all"),
        normalizerUpdatePolicy="all",
        saveNormalizedFeatures=True,
        printSummary=True,
    )


def runExperiment(datasetPath, selectedFeatures, datasetName="Adaptacao", outputDirectory="output/ExpNormalizers", modelCodes=None, rollingWindow=200):
    experimentConfig = buildExperiment(
        datasetPath=datasetPath,
        selectedFeatures=selectedFeatures,
        datasetName=datasetName,
        outputDirectory=outputDirectory,
        modelCodes=modelCodes,
        rollingWindow=rollingWindow,
    )
    return ExperimentRunner(experimentConfig).runExperiment()


def parseArguments():
    parser = argparse.ArgumentParser(description="Executa o experimento com os cinco tipos de normalização online.")
    parser.add_argument("--dataset", required=True, help="Caminho do arquivo CSV.")
    parser.add_argument("--features", required=True, help="Features separadas por vírgula.")
    parser.add_argument("--datasetName", default="Adaptacao")
    parser.add_argument("--output", default="output/ExpNormalizers")
    parser.add_argument("--models", default="AIF,HST", help="Modelos separados por vírgula.")
    parser.add_argument("--rollingWindow", type=int, default=200)
    return parser.parse_args()


def main():
    arguments = parseArguments()
    selectedFeatures = [feature.strip() for feature in arguments.features.split(",") if feature.strip()]
    modelCodes = [model.strip().upper() for model in arguments.models.split(",") if model.strip()]
    generatedFiles = runExperiment(
        datasetPath=arguments.dataset,
        selectedFeatures=selectedFeatures,
        datasetName=arguments.datasetName,
        outputDirectory=arguments.output,
        modelCodes=modelCodes,
        rollingWindow=arguments.rollingWindow,
    )
    print("\nArquivos gerados:")
    for generatedFile in generatedFiles:
        print(generatedFile)


if __name__ == "__main__":
    main()
