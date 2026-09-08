"""Persistência dos trials, ranking e exportação das métricas de otimização."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import shutil
import math
from pathlib import Path

import pandas as pd

from src.Metrics.Metrics import Metrics


class OptimizationExporter:
    globalMetricNames = (
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

    def __init__(self, outputDirectory, topK=10):
        self.outputDirectory = Path(outputDirectory)
        self.topK = max(1, int(topK))

    def preservePreviousExports(self, studies):
        """Copia exportações de outra busca antes de reutilizar os caminhos públicos."""
        files = [self.outputDirectory / name for name in ("trials.csv", "top10_windows.csv")]
        existing = [path for path in files if path.exists()]
        if not existing:
            return None
        currentNames = {study.study_name for study in studies.values()}
        compatible = True
        for path in existing:
            try:
                frame = pd.read_csv(path, usecols=["studyName"])
                names = set(frame["studyName"].dropna().astype(str))
                compatible = compatible and names <= currentNames
            except (ValueError, pd.errors.EmptyDataError):
                compatible = False
        if compatible:
            return None
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = self.outputDirectory / "archive" / f"previous_export_{timestamp}"
        directory.mkdir(parents=True, exist_ok=False)
        for path in existing:
            shutil.copy2(path, directory / path.name)
        print(f"CSVs anteriores preservados em: {directory}")
        return str(directory)

    def recordIdentity(self, trial, scenario, modelCode):
        for name, value in {
            "scenario": scenario.name, "dataset": scenario.datasetName,
            "modelCode": modelCode, "studyName": trial.study.study_name,
            "samplerSeed": trial.study.user_attrs.get("lastSamplerSeed"),
        }.items():
            trial.set_user_attr(name, value)

    def recordConfiguration(self, trial, configuration, optimizationConfig):
        effective = asdict(configuration)
        # Mesmo comportamento, mesma assinatura, independentemente de aliases.
        source = configuration.thresholdScoreSource
        if source != "raw":
            source = "scoreMa" + str(source).replace("scoreMa", "").replace("ma", "")
        effective["thresholdScoreSource"] = source
        effective.update({
            "normalizer": optimizationConfig.normalizerName,
            "normalizerParameters": optimizationConfig.normalizerParameters,
            "initialWarmupSize": optimizationConfig.initialWarmupSize,
            "metricsWindowSize": optimizationConfig.metricsWindowSize,
            "trainingStrategy": "all",
        })
        trial.set_user_attr("effectiveConfiguration", effective)
        trial.set_user_attr("modelParameters", dict(configuration.modelParameters))
        for name in ("thresholdScoreSource", "calibrationSize", "optimizationStarts", "tolerance"):
            trial.set_user_attr(name, effective[name])
        for name in ("initialWarmupSize", "metricsWindowSize", "normalizer"):
            trial.set_user_attr(name, effective[name])

    def recordResult(self, trial, result):
        metrics = result.streamMetrics
        if not math.isfinite(float(metrics["f1"])):
            raise ValueError("O trial produziu um F1 não finito.")
        countNames = {"tp", "tn", "fp", "fn", "evaluatedInstances", "benignInstances", "attackInstances"}
        for name in self.globalMetricNames:
            value = int(metrics[name]) if name in countNames else float(metrics[name])
            trial.set_user_attr(name, value)
        trial.set_user_attr("windowCounts", [
            [int(window[name]) for name in ("windowIndex", "windowStart", "windowEnd", "tp", "tn", "fp", "fn")]
            for window in result.windowMetrics
        ])

    def export(self, studies):
        self.outputDirectory.mkdir(parents=True, exist_ok=True)
        trialRows = []
        windowRows = []

        for scenario, study in studies.items():
            ranking = self._rankTrials(study.trials)
            topTrials = [
                trial
                for trial in study.trials
                if trial.number in ranking
                and ranking[trial.number] <= self.topK
            ]
            bestSoFar = -math.inf

            for trial in sorted(study.trials, key=lambda item: item.number):
                if self._isComplete(trial):
                    bestSoFar = max(bestSoFar, float(trial.value))
                row = self._trialRow(
                    scenario=scenario,
                    studyName=study.study_name,
                    trial=trial,
                    rank=ranking.get(trial.number),
                    bestSoFar=(
                        bestSoFar if math.isfinite(bestSoFar) else None
                    ),
                )
                trialRows.append(row)

            for trial in sorted(
                topTrials,
                key=lambda item: ranking[item.number],
            ):
                windowRows.extend(
                    self._windowRows(
                        scenario=scenario,
                        trial=trial,
                        rank=ranking[trial.number],
                    )
                )

        trialsPath = self.outputDirectory / "trials.csv"
        windowsPath = self.outputDirectory / "top10_windows.csv"
        trialFrame = pd.DataFrame(trialRows)
        windowFrame = pd.DataFrame(windowRows)
        # CSVs vazios continuam legíveis pelo notebook (inclusive se todos falharem).
        if trialFrame.empty:
            trialFrame = pd.DataFrame(columns=[
                "scenario", "studyName", "trialNumber", "state", "objectiveF1", "bestF1SoFar",
                "rank", "isTop10", "dataset", "modelCode", *self.globalMetricNames,
            ])
        if windowFrame.empty:
            windowFrame = pd.DataFrame(columns=[
                "scenario", "studyName", "modelCode", "rank", "trialNumber", "globalF1",
                "windowIndex", "windowStart", "windowEnd", "tp", "tn", "fp", "fn", "f1",
                "cumulativeInstances", "cumulativeF1",
            ])
        self._writeCsv(trialFrame, trialsPath)
        self._writeCsv(windowFrame, windowsPath)
        return {
            "trialsPath": str(trialsPath),
            "top10WindowsPath": str(windowsPath),
        }

    def _rankTrials(self, trials):
        completed = [trial for trial in trials if self._isComplete(trial)]
        ordered = sorted(completed, key=self._rankingKey)
        ranking = {}
        signatures = set()
        rank = 0

        for trial in ordered:
            signature = json.dumps(
                trial.user_attrs.get("effectiveConfiguration", {
                    "params": trial.params,
                    "modelParameters": trial.user_attrs.get("modelParameters", {}),
                }), sort_keys=True, default=str,
            )
            if signature in signatures:
                continue
            signatures.add(signature)
            rank += 1
            ranking[trial.number] = rank
        return ranking

    def _rankingKey(self, trial):
        attrs = trial.user_attrs
        return (
            -float(trial.value),
            -float(attrs.get("recall", 0.0)),
            -float(attrs.get("precision", 0.0)),
            int(attrs.get("fn", 10**30)),
            int(attrs.get("fp", 10**30)),
            int(trial.number),
        )

    def _isComplete(self, trial):
        return (
            getattr(trial.state, "name", str(trial.state)) == "COMPLETE"
            and trial.value is not None
            and math.isfinite(float(trial.value))
        )

    def _trialRow(self, scenario, studyName, trial, rank, bestSoFar):
        attrs = trial.user_attrs
        row = {
            "scenario": scenario,
            "studyName": studyName,
            "trialNumber": int(trial.number),
            "state": getattr(trial.state, "name", str(trial.state)),
            "objectiveF1": (
                float(trial.value) if self._isComplete(trial) else None
            ),
            "bestF1SoFar": bestSoFar,
            "rank": rank,
            "isTop10": bool(rank is not None and rank <= self.topK),
            "dataset": attrs.get("dataset"),
            "modelCode": attrs.get("modelCode"),
            "thresholdScoreSource": attrs.get("thresholdScoreSource"),
            "calibrationSize": attrs.get("calibrationSize"),
            "optimizationStarts": attrs.get("optimizationStarts"),
            "tolerance": attrs.get("tolerance"),
            "initialWarmupSize": attrs.get("initialWarmupSize"),
            "metricsWindowSize": attrs.get("metricsWindowSize"),
            "normalizer": attrs.get("normalizer"),
            "samplerSeed": attrs.get("samplerSeed"),
            "errorType": attrs.get("errorType"),
            "errorMessage": attrs.get("errorMessage"),
        }
        for name in self.globalMetricNames:
            row[name] = attrs.get(name)
        for name, value in sorted(trial.params.items()):
            row[f"param_{name}"] = value
        for name, value in sorted(attrs.get("modelParameters", {}).items()):
            row[f"model_{name}"] = value
        return row

    def _windowRows(self, scenario, trial, rank):
        rows = []
        cumulative = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        parameterColumns = {
            f"param_{name}": value
            for name, value in sorted(trial.params.items())
        }
        fixedModelColumns = {
            f"model_{name}": value
            for name, value in sorted(
                trial.user_attrs.get("modelParameters", {}).items()
            )
        }

        for values in trial.user_attrs.get("windowCounts", []):
            (
                windowIndex,
                windowStart,
                windowEnd,
                tp,
                tn,
                fp,
                fn,
            ) = values
            windowMetrics = Metrics.fromCounts(tp, tn, fp, fn)
            cumulative["tp"] += int(tp)
            cumulative["tn"] += int(tn)
            cumulative["fp"] += int(fp)
            cumulative["fn"] += int(fn)
            cumulativeMetrics = Metrics.fromCounts(**cumulative)
            rows.append(
                {
                    "scenario": scenario,
                    "studyName": trial.user_attrs.get("studyName"),
                    "modelCode": trial.user_attrs.get("modelCode"),
                    "dataset": trial.user_attrs.get("dataset"),
                    "rank": int(rank),
                    "trialNumber": int(trial.number),
                    "globalF1": float(trial.value),
                    "thresholdScoreSource": trial.user_attrs.get(
                        "thresholdScoreSource"
                    ),
                    "calibrationSize": trial.user_attrs.get(
                        "calibrationSize"
                    ),
                    "optimizationStarts": trial.user_attrs.get(
                        "optimizationStarts"
                    ),
                    "tolerance": trial.user_attrs.get("tolerance"),
                    "windowIndex": int(windowIndex),
                    "windowStart": int(windowStart),
                    "windowEnd": int(windowEnd),
                    **windowMetrics,
                    "cumulativeInstances": cumulativeMetrics["instances"],
                    "cumulativeTp": cumulativeMetrics["tp"],
                    "cumulativeTn": cumulativeMetrics["tn"],
                    "cumulativeFp": cumulativeMetrics["fp"],
                    "cumulativeFn": cumulativeMetrics["fn"],
                    "cumulativeAccuracy": cumulativeMetrics["accuracy"],
                    "cumulativePrecision": cumulativeMetrics["precision"],
                    "cumulativeRecall": cumulativeMetrics["recall"],
                    "cumulativeSpecificity": cumulativeMetrics["specificity"],
                    "cumulativeF1": cumulativeMetrics["f1"],
                    "cumulativeMcc": cumulativeMetrics["mcc"],
                    **parameterColumns,
                    **fixedModelColumns,
                }
            )
        return rows

    def _writeCsv(self, frame, path):
        temporaryPath = path.with_name(f"{path.name}.tmp")
        frame.to_csv(temporaryPath, index=False)
        temporaryPath.replace(path)
