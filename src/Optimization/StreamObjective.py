from src.Pipeline.MetricsResultManager import MetricsResultManager


class StreamObjective:
    metricNames = (
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "tp",
        "tn",
        "fp",
        "fn",
        "evaluatedInstances",
        "benignInstances",
        "attackInstances",
        "attackRatioPercent",
    )

    def __init__(
        self,
        scenario,
        searchSpace,
        pipelineFactory,
        modelProfile,
    ):
        self.scenario = scenario
        self.searchSpace = searchSpace
        self.pipelineFactory = pipelineFactory
        self.modelProfile = modelProfile

    def __call__(self, trial):
        trialConfiguration = self.searchSpace.suggest(trial)
        resultManager = MetricsResultManager()
        pipeline = self.pipelineFactory.create(
            scenario=self.scenario,
            trialConfiguration=trialConfiguration,
            resultManager=resultManager,
        )
        result = pipeline.run()
        metrics = result.streamMetrics

        trial.set_user_attr("scenario", self.scenario.name)
        trial.set_user_attr("dataset", self.scenario.datasetName)
        trial.set_user_attr("modelCode", self.modelProfile.code)
        trial.set_user_attr(
            "modelParameters",
            dict(self.modelProfile.parameters),
        )
        trial.set_user_attr(
            "thresholdScoreSource",
            trialConfiguration.thresholdScoreSource,
        )
        trial.set_user_attr(
            "calibrationSize",
            trialConfiguration.calibrationSize,
        )
        trial.set_user_attr(
            "optimizationStarts",
            trialConfiguration.optimizationStarts,
        )
        trial.set_user_attr("tolerance", trialConfiguration.tolerance)

        for name in self.metricNames:
            value = metrics[name]
            if isinstance(value, int):
                value = int(value)
            else:
                value = float(value)
            trial.set_user_attr(name, value)

        trial.set_user_attr(
            "windowCounts",
            [
                [
                    int(window["windowIndex"]),
                    int(window["windowStart"]),
                    int(window["windowEnd"]),
                    int(window["tp"]),
                    int(window["tn"]),
                    int(window["fp"]),
                    int(window["fn"]),
                ]
                for window in result.windowMetrics
            ],
        )
        return float(metrics["f1"])
