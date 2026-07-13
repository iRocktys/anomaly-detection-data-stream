from pathlib import Path

from src.Pipeline.ExperimentConfig import (
    ComponentConfig,
    DatasetConfig,
    ExperimentPlan,
    ModelConfig,
    OutputConfig,
    ThresholdEvaluationConfig,
)


class JupyterExperiment:
    """Fachada única para montar, gerar scores e avaliar thresholds no notebook."""

    supportedModels = {"AIF", "HST", "AE"}

    def __init__(
        self,
        datasetPath=None,
        selectedFeatures=None,
        datasetName="dataset",
        outputDirectory="output/Experiments",
        targetLabelColumn="Label",
        ignoredColumns=None,
        imputationMethod="0",
        normalClassIndex=0,
        warmup=200,
        movingAverageWindows=None,
        runSeeds=None,
        runSeed=None,
        saveNormalizedFeatures=False,
        saveWindowMetrics=True,
        printSummary=True,
    ):
        self.datasets = []
        self.models = []
        self.normalizers = []
        self.thresholdEvaluations = []
        self.featureExtractor = ComponentConfig("none")
        self.featureSmoother = ComponentConfig("none")
        self.trainingStrategy = ComponentConfig("all")
        self.normalizerUpdatePolicy = "all"
        self.warmup = int(warmup)
        self.movingAverageWindows = list(movingAverageWindows or [3, 5, 10, 50, 100])
        resolvedSeeds = runSeeds if runSeeds is not None else [1 if runSeed is None else runSeed]
        self.runSeeds = [int(seed) for seed in resolvedSeeds]
        self.output = OutputConfig(
            directory=str(Path(outputDirectory).expanduser()),
            saveNormalizedFeatures=bool(saveNormalizedFeatures),
            saveWindowMetrics=bool(saveWindowMetrics),
            printSummary=bool(printSummary),
        )
        self.defaultThreshold = ComponentConfig("dspot")
        self.defaultScoreSmoother = ComponentConfig("none")
        self.defaultDecision = ComponentConfig("threshold")
        self.defaultEvaluationEnabled = False

        if datasetPath is not None:
            self.addDataset(
                datasetPath,
                selectedFeatures or [],
                datasetName,
                targetLabelColumn,
                ignoredColumns,
                imputationMethod,
                normalClassIndex,
            )

    def addDataset(
        self,
        path,
        selectedFeatures,
        name="dataset",
        targetLabelColumn="Label",
        ignoredColumns=None,
        imputationMethod="0",
        normalClassIndex=0,
    ):
        self.datasets.append(DatasetConfig(
            path=str(Path(path).expanduser()),
            selectedFeatures=list(selectedFeatures),
            name=str(name),
            targetLabelColumn=str(targetLabelColumn),
            ignoredColumns=list(ignoredColumns or [
                "Source IP",
                "Source Port",
                "Destination IP",
                "Destination Port",
                "Protocol",
                "Inbound",
            ]),
            imputationMethod=str(imputationMethod),
            normalClassIndex=int(normalClassIndex),
        ))
        return self

    def clearDatasets(self):
        self.datasets = []
        return self

    def addModel(self, code, parameters=None, name=None):
        resolvedCode = str(code).strip().upper()
        if resolvedCode not in self.supportedModels:
            raise ValueError(f"Modelo desconhecido: {code}.")
        self.models.append(ModelConfig(resolvedCode, dict(parameters or {}), name))
        return self

    def clearModels(self):
        self.models = []
        return self

    def addNormalizer(self, name="none", parameters=None):
        self.normalizers.append(self.createComponent(name, parameters))
        return self

    def clearNormalizers(self):
        self.normalizers = []
        return self

    def addThresholdEvaluation(
        self,
        name,
        thresholdName="dspot",
        thresholdParameters=None,
        scoreSmootherName="none",
        scoreSmootherParameters=None,
        scoreColumn="rawScore",
        metricsWindow=1000,
        evaluateOnlyReady=True,
        saveThresholdState=False,
        decisionName="threshold",
        decisionParameters=None,
    ):
        self.thresholdEvaluations.append(ThresholdEvaluationConfig(
            name=str(name),
            threshold=self.createComponent(thresholdName, thresholdParameters),
            scoreSmoother=self.createComponent(scoreSmootherName, scoreSmootherParameters),
            decisionStrategy=self.createComponent(decisionName, decisionParameters),
            scoreColumn=str(scoreColumn),
            metricsWindow=int(metricsWindow),
            evaluateOnlyReady=bool(evaluateOnlyReady),
            saveThresholdState=bool(saveThresholdState),
        ))
        return self

    def clearThresholdEvaluations(self):
        self.thresholdEvaluations = []
        return self

    def setFeatureSelector(self, name="selected", parameters=None):
        normalizedName = str(name).replace("_", "").strip().lower()
        if normalizedName not in {"selected", "all", "none"}:
            raise ValueError(
                "A seleção de features agora pertence ao DatasetConfig.selectedFeatures; "
                "não existem seletores adicionais no fluxo."
            )
        return self

    def setFeatureExtractor(self, name="none", parameters=None):
        self.featureExtractor = self.createComponent(name, parameters)
        return self

    def setFeatureSmoother(self, name="none", parameters=None):
        self.featureSmoother = self.createComponent(name, parameters)
        return self

    def setScoreSmoother(self, name="none", parameters=None):
        self.defaultScoreSmoother = self.createComponent(name, parameters)
        self.defaultEvaluationEnabled = True
        return self

    def setThreshold(self, name="dspot", parameters=None):
        self.defaultThreshold = self.createComponent(name, parameters)
        self.defaultEvaluationEnabled = True
        return self

    def setDecisionStrategy(self, name="threshold", parameters=None):
        self.defaultDecision = self.createComponent(name, parameters)
        self.defaultEvaluationEnabled = True
        return self

    def setTrainingStrategy(self, name="all", parameters=None):
        """Seleciona como o modelo será atualizado durante o fluxo.

        ``all`` é o padrão e treina com todas as instâncias. ``predictedNormal``
        treina durante o warmup e, depois, somente com instâncias classificadas
        como normais pela avaliação de threshold indicada em ``evaluationName``.
        """
        aliases = {
            "all": "all",
            "trainall": "all",
            "predictednormal": "predictedNormal",
            "normalprediction": "predictedNormal",
            "predictednormalonly": "predictedNormal",
        }
        normalizedName = "".join(
            character
            for character in str(name or "").lower()
            if character.isalnum()
        )
        if normalizedName not in aliases:
            raise ValueError("Estratégia deve ser all ou predictedNormal.")
        self.trainingStrategy = self.createComponent(
            aliases[normalizedName],
            parameters,
        )
        if aliases[normalizedName] == "predictedNormal":
            self.defaultEvaluationEnabled = True
        return self

    def setWarmup(self, warmup):
        self.warmup = int(warmup)
        if self.warmup < 20:
            raise ValueError("warmup deve ser maior ou igual a 20.")
        return self

    def setNormalizerUpdatePolicy(self, policyName="all"):
        aliases = {
            "all": "all",
            "none": "none",
            "normalOnly": "oracleNormal",
            "oracleNormal": "oracleNormal",
        }
        if policyName not in aliases:
            raise ValueError("Política deve ser all, oracleNormal ou none.")
        self.normalizerUpdatePolicy = aliases[policyName]
        return self

    def setRunSeeds(self, runSeeds):
        self.runSeeds = [int(seed) for seed in runSeeds]
        return self

    def setOutputOptions(
        self,
        outputDirectory=None,
        saveNormalizedFeatures=None,
        saveWindowMetrics=None,
        printSummary=None,
    ):
        self.output = OutputConfig(
            directory=str(Path(outputDirectory or self.output.directory).expanduser()),
            saveNormalizedFeatures=(
                self.output.saveNormalizedFeatures
                if saveNormalizedFeatures is None
                else bool(saveNormalizedFeatures)
            ),
            saveWindowMetrics=(
                self.output.saveWindowMetrics
                if saveWindowMetrics is None
                else bool(saveWindowMetrics)
            ),
            printSummary=(
                self.output.printSummary
                if printSummary is None
                else bool(printSummary)
            ),
        )
        return self

    def buildPlan(self, includeDefaultEvaluation=False):
        evaluations = list(self.thresholdEvaluations)
        if includeDefaultEvaluation and not evaluations:
            evaluations.append(ThresholdEvaluationConfig(
                name="default",
                threshold=self.defaultThreshold,
                scoreSmoother=self.defaultScoreSmoother,
                decisionStrategy=self.defaultDecision,
            ))
        plan = ExperimentPlan(
            datasets=list(self.datasets),
            models=list(self.models),
            normalizers=list(self.normalizers),
            thresholdEvaluations=evaluations,
            featureExtractor=self.featureExtractor,
            featureSmoother=self.featureSmoother,
            trainingStrategy=self.trainingStrategy,
            normalizerUpdatePolicy=self.normalizerUpdatePolicy,
            warmup=self.warmup,
            movingAverageWindows=list(self.movingAverageWindows),
            runSeeds=list(self.runSeeds),
            output=self.output,
        )
        plan.validate()
        return plan

    def describe(self):
        return self.buildPlan(includeDefaultEvaluation=self.defaultEvaluationEnabled).toDict()

    def run(self, runId=None):
        from src.Pipeline.ExperimentRunner import ExperimentRunner
        return ExperimentRunner(self.buildPlan(includeDefaultEvaluation=True)).runExperiment(runId=runId)

    def runScores(self, runId=None):
        from src.Pipeline.ExperimentRunner import ExperimentRunner
        includeEvaluation = (
            "".join(
                character
                for character in str(self.trainingStrategy.name).lower()
                if character.isalnum()
            )
            in {"predictednormal", "normalprediction", "predictednormalonly"}
        )
        return ExperimentRunner(
            self.buildPlan(includeDefaultEvaluation=includeEvaluation)
        ).runScores(runId=runId)

    def evaluateScores(self, scoreFiles, runId=None):
        from src.Pipeline.ExperimentRunner import ExperimentRunner
        return ExperimentRunner(self.buildPlan(includeDefaultEvaluation=True)).evaluateScores(
            scoreFiles,
            runId=runId,
        )

    @staticmethod
    def createComponent(name, parameters=None):
        if isinstance(name, ComponentConfig):
            if parameters:
                raise ValueError("Não informe parameters com ComponentConfig.")
            return ComponentConfig(name.name, dict(name.parameters))
        if isinstance(name, dict):
            if parameters:
                raise ValueError("Não informe parameters com dicionário de componente.")
            return ComponentConfig(str(name["name"]), dict(name.get("parameters", {})))
        resolvedName = str(name).strip()
        if not resolvedName:
            raise ValueError("Nome do componente não pode ser vazio.")
        return ComponentConfig(resolvedName, dict(parameters or {}))


def createJupyterExperiment(*args, **kwargs):
    return JupyterExperiment(*args, **kwargs)


def runJupyterExperiment(
    datasetPath,
    selectedFeatures,
    datasetName="Adaptacao",
    outputDirectory="output/Experiments",
    modelCodes=None,
    modelParameters=None,
    normalizerNames=None,
    normalizerParameters=None,
    rollingWindow=200,
    warmup=200,
    normalizerUpdatePolicy="all",
    trainingStrategy="all",
    trainingParameters=None,
    featureSelector="selected",
    featureSelectorParameters=None,
    featureExtractor="none",
    featureExtractorParameters=None,
    featureSmoother="none",
    featureSmootherParameters=None,
    scoreSmoother="none",
    scoreSmootherParameters=None,
    thresholdName="dspot",
    thresholdParameters=None,
    decisionStrategy="threshold",
    decisionParameters=None,
    movingAverageWindows=None,
    runSeeds=None,
    runSeed=1,
    thresholdEvaluations=None,
    saveNormalizedFeatures=True,
    saveWindowMetrics=True,
    printSummary=True,
    runId=None,
):
    experiment = JupyterExperiment(
        datasetPath=datasetPath,
        selectedFeatures=selectedFeatures,
        datasetName=datasetName,
        outputDirectory=outputDirectory,
        warmup=warmup,
        movingAverageWindows=movingAverageWindows,
        runSeeds=runSeeds if runSeeds is not None else [runSeed],
        saveNormalizedFeatures=saveNormalizedFeatures,
        saveWindowMetrics=saveWindowMetrics,
        printSummary=printSummary,
    )
    resolvedModelParameters = dict(modelParameters or {})
    for modelCode in modelCodes or ["AIF", "HST"]:
        code = str(modelCode).strip().upper()
        experiment.addModel(code, resolvedModelParameters.get(code, {}))

    resolvedNormalizerParameters = dict(normalizerParameters or {})
    for normalizerName in normalizerNames or [
        "none",
        "incrementalMinMax",
        "incrementalZScore",
        "rollingMinMax",
        "rollingZScore",
    ]:
        parameters = dict(resolvedNormalizerParameters.get(normalizerName, {}))
        if normalizerName in {"rollingMinMax", "rollingZScore"}:
            parameters.setdefault("windowSize", rollingWindow)
        experiment.addNormalizer(normalizerName, parameters)

    experiment.setFeatureSelector(featureSelector, featureSelectorParameters)
    experiment.setFeatureExtractor(featureExtractor, featureExtractorParameters)
    experiment.setFeatureSmoother(featureSmoother, featureSmootherParameters)
    experiment.setTrainingStrategy(trainingStrategy, trainingParameters)
    experiment.setNormalizerUpdatePolicy(normalizerUpdatePolicy)

    if thresholdEvaluations:
        for evaluation in thresholdEvaluations:
            experiment.addThresholdEvaluation(**evaluation)
    else:
        experiment.addThresholdEvaluation(
            name="default",
            thresholdName=thresholdName,
            thresholdParameters=thresholdParameters or {},
            scoreSmootherName=scoreSmoother,
            scoreSmootherParameters=scoreSmootherParameters,
            decisionName=decisionStrategy,
            decisionParameters=decisionParameters,
        )
    return experiment.run(runId=runId)
