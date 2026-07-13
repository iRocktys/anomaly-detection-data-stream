from pathlib import Path

import pandas as pd

from src.Pipeline.ExperimentBuilder import ExperimentBuilder
from src.Pipeline.ResultManager import ResultManager
from src.Pipeline.StreamPipeline import StreamPipeline
from src.Pipeline.ThresholdReplayEvaluator import ThresholdReplayEvaluator


class ExperimentRunner:
    def __init__(self, plan):
        self.plan = plan
        self.builder = ExperimentBuilder()
        self.evaluator = ThresholdReplayEvaluator(self.builder)

    def runExperiment(self, runId=None):
        self.plan.validate()
        manager = ResultManager(self.plan, runId=runId, mode="full")
        result = manager.createResult()
        try:
            scoreArtifacts = self.generateScores(manager, result)
            self.evaluateArtifacts(scoreArtifacts, manager, result)
            return manager.finish(result)
        except Exception as error:
            manager.finish(result, status="failed", error=error)
            raise

    def runScores(self, runId=None):
        scorePlan = self.plan.scorePlan()
        scorePlan.validate()
        manager = ResultManager(scorePlan, runId=runId, mode="scores")
        result = manager.createResult()
        try:
            self.generateScores(manager, result, plan=scorePlan)
            return manager.finish(result)
        except Exception as error:
            manager.finish(result, status="failed", error=error)
            raise

    def evaluateScores(self, scoreFiles, runId=None):
        self.plan.validate()
        if not self.plan.thresholdEvaluations:
            raise ValueError("Adicione ao menos uma avaliação de threshold antes do replay.")
        manager = ResultManager(self.plan, runId=runId, mode="thresholdReplay")
        result = manager.createResult()
        try:
            artifacts = [self.readScoreArtifact(path) for path in scoreFiles]
            self.evaluateArtifacts(artifacts, manager, result)
            return manager.finish(result)
        except Exception as error:
            manager.finish(result, status="failed", error=error)
            raise

    def generateScores(self, manager, result, plan=None):
        activePlan = plan or self.plan
        artifacts = []
        for datasetConfig in activePlan.datasets:
            dataFrame = pd.read_csv(datasetConfig.resolvedPath())
            for modelConfig in activePlan.models:
                for normalizerConfig in activePlan.normalizers:
                    for runSeed in activePlan.runSeeds:
                        stream, targetNames, featureNames = self.createStream(datasetConfig, dataFrame)
                        components = self.builder.buildScoreComponents(
                            plan=activePlan,
                            schema=stream.get_schema(),
                            modelConfig=modelConfig,
                            normalizerConfig=normalizerConfig,
                            runSeed=runSeed,
                        )
                        configurationHash = ResultManager.configurationHash({
                            "dataset": datasetConfig.toDict(),
                            "model": modelConfig.toDict(),
                            "normalizer": normalizerConfig.toDict(),
                            "featureExtractor": activePlan.featureExtractor.toDict(),
                            "featureSmoother": activePlan.featureSmoother.toDict(),
                            "trainingStrategy": activePlan.trainingStrategy.toDict(),
                            "normalizerUpdatePolicy": activePlan.normalizerUpdatePolicy,
                            "runSeed": int(runSeed),
                        })
                        artifactId = ResultManager.buildArtifactId(
                            datasetConfig.name,
                            modelConfig.resolvedName(),
                            normalizerConfig.name,
                            f"seed-{runSeed}",
                            configurationHash,
                        )
                        metadata = {
                            "runId": manager.runId,
                            "scoreArtifactId": artifactId,
                            "dataset": datasetConfig.name,
                            "datasetPath": datasetConfig.resolvedPath(),
                            "targetNames": list(targetNames),
                            "featureNames": list(featureNames),
                            "modelCode": modelConfig.resolvedCode(),
                            "modelConfig": modelConfig.resolvedName(),
                            "modelParameters": components["modelParameters"],
                            "normalizer": normalizerConfig.name,
                            "normalizerParameters": dict(normalizerConfig.parameters),
                            "runSeed": int(runSeed),
                        }
                        self.printScoreSummary(activePlan, metadata)
                        rows = StreamPipeline(
                            activePlan,
                            datasetConfig,
                            components,
                            featureNames,
                        ).run(stream, metadata)
                        scoreFrame = pd.DataFrame(rows)
                        scorePath = manager.saveScores(
                            scoreFrame,
                            artifactId,
                            metadata,
                            result,
                        )
                        artifacts.append({
                            "path": scorePath,
                            "artifactId": artifactId,
                            "metadata": metadata,
                        })
        return artifacts

    def evaluateArtifacts(self, scoreArtifacts, manager, result):
        for scoreArtifact in scoreArtifacts:
            scoreFrame = pd.read_csv(scoreArtifact["path"])
            for evaluationConfig in self.plan.thresholdEvaluations:
                evaluationHash = ResultManager.configurationHash(evaluationConfig.toDict())
                evaluationId = ResultManager.buildArtifactId(
                    scoreArtifact["artifactId"],
                    evaluationConfig.name,
                    evaluationHash,
                )
                self.printEvaluationSummary(scoreArtifact, evaluationConfig)
                evaluationFrame, summary, windowMetrics = self.evaluator.evaluateFrame(
                    scoreFrame,
                    evaluationConfig,
                    evaluationId=evaluationId,
                )
                metadata = {
                    "sourceScorePath": str(scoreArtifact["path"]),
                    "sourceScoreArtifactId": scoreArtifact["artifactId"],
                    "evaluation": evaluationConfig.toDict(),
                }
                manager.saveEvaluation(
                    evaluationFrame,
                    evaluationId,
                    metadata,
                    result,
                )
                summary.update({
                    "runId": manager.runId,
                    "sourceScorePath": str(scoreArtifact["path"]),
                    "sourceScoreArtifactId": scoreArtifact["artifactId"],
                })
                manager.saveMetrics(
                    summary,
                    windowMetrics,
                    evaluationId,
                    metadata,
                    result,
                )

    def createStream(self, datasetConfig, dataFrame):
        try:
            from src.Data.Processor import DataStreamProcessor
        except ImportError as error:
            raise ImportError(
                "As dependências de processamento e o CapyMOA devem estar instalados para gerar scores."
            ) from error
        processor = DataStreamProcessor(
            logging=False,
            selected_features=datasetConfig.selectedFeatures,
        )
        stream, targetNames, featureNames = processor.create_stream(
            df=dataFrame.copy(),
            target_label_col=datasetConfig.targetLabelColumn,
            binary_label=False,
            normalize_method=None,
            threshold_var=None,
            threshold_corr=None,
            top_n_features=None,
            return_stream=True,
            extra_ignore_cols=datasetConfig.ignoredColumns,
            imputation_method=datasetConfig.imputationMethod,
        )
        return stream, targetNames, featureNames

    def readScoreArtifact(self, scorePath):
        path = Path(scorePath)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de score não encontrado: {scorePath}")
        frame = pd.read_csv(path, nrows=1)
        artifactId = (
            str(frame.iloc[0]["scoreArtifactId"])
            if not frame.empty and "scoreArtifactId" in frame.columns
            else path.stem
        )
        metadata = {
            "external": True,
            "sourcePath": str(path),
        }
        return {"path": str(path), "artifactId": artifactId, "metadata": metadata}

    def printScoreSummary(self, plan, metadata):
        if not plan.output.printSummary:
            return
        print(
            f"[scores] dataset={metadata['dataset']} model={metadata['modelConfig']} "
            f"normalizer={metadata['normalizer']} seed={metadata['runSeed']}"
        )

    def printEvaluationSummary(self, scoreArtifact, evaluationConfig):
        if not self.plan.output.printSummary:
            return
        print(
            f"[threshold] score={scoreArtifact['artifactId']} "
            f"evaluation={evaluationConfig.name} threshold={evaluationConfig.threshold.name}"
        )
