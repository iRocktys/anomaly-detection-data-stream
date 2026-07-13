import argparse

from src.Experiments.JupyterNormalizerExperiment import JupyterExperiment


def buildExperiment(
    datasetPath,
    selectedFeatures,
    datasetName="Adaptacao",
    outputDirectory="output/ExpNormalizers",
    modelCodes=None,
    modelParameters=None,
    rollingWindow=200,
    runSeeds=None,
    warmup=200,
):
    experiment = JupyterExperiment(
        datasetPath=datasetPath,
        selectedFeatures=selectedFeatures,
        datasetName=datasetName,
        outputDirectory=outputDirectory,
        runSeeds=runSeeds or [1],
        warmup=warmup,
        saveNormalizedFeatures=True,
        printSummary=True,
    )
    resolvedParameters = dict(modelParameters or {})
    for modelCode in modelCodes or ["AIF", "HST"]:
        code = str(modelCode).strip().upper()
        experiment.addModel(code, resolvedParameters.get(code, {}))
    experiment.addNormalizer("none")
    experiment.addNormalizer("incrementalMinMax")
    experiment.addNormalizer("incrementalZScore")
    experiment.addNormalizer("rollingMinMax", {"windowSize": rollingWindow})
    experiment.addNormalizer("rollingZScore", {"windowSize": rollingWindow})
    experiment.addThresholdEvaluation(
        name="dspot",
        thresholdName="dspot",
        thresholdParameters={},
    )
    return experiment


def runExperiment(**options):
    return buildExperiment(**options).run()


def parseArguments():
    parser = argparse.ArgumentParser(
        description="Preset reprodutível para comparar os cinco normalizadores online."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--datasetName", default="Adaptacao")
    parser.add_argument("--output", default="output/ExpNormalizers")
    parser.add_argument("--models", default="AIF,HST")
    parser.add_argument("--rollingWindow", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=200)
    return parser.parse_args()


def main():
    arguments = parseArguments()
    result = runExperiment(
        datasetPath=arguments.dataset,
        selectedFeatures=[value.strip() for value in arguments.features.split(",") if value.strip()],
        datasetName=arguments.datasetName,
        outputDirectory=arguments.output,
        modelCodes=[value.strip().upper() for value in arguments.models.split(",") if value.strip()],
        rollingWindow=arguments.rollingWindow,
        warmup=arguments.warmup,
    )
    print(result.summary())


if __name__ == "__main__":
    main()
