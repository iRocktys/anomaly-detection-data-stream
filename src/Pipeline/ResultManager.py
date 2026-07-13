import hashlib
import json
import os
import platform
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class ExperimentResult:
    runId: str
    runDirectory: str
    manifestPath: str
    scoreFiles: list[str] = field(default_factory=list)
    evaluationFiles: list[str] = field(default_factory=list)
    metricFiles: list[str] = field(default_factory=list)
    windowMetricFiles: list[str] = field(default_factory=list)

    def summary(self):
        return {
            "runId": self.runId,
            "runDirectory": self.runDirectory,
            "manifestPath": self.manifestPath,
            "scoreFiles": list(self.scoreFiles),
            "evaluationFiles": list(self.evaluationFiles),
            "metricFiles": list(self.metricFiles),
            "windowMetricFiles": list(self.windowMetricFiles),
        }


class ResultManager:
    def __init__(self, plan, runId=None, mode="full"):
        self.plan = plan
        self.mode = mode
        self.runId = self.resolveRunId(runId)
        self.runDirectory = Path(plan.output.resolvedDirectory()) / self.runId
        self.scoreDirectory = self.runDirectory / "scores"
        self.evaluationDirectory = self.runDirectory / "evaluations"
        self.metricDirectory = self.runDirectory / "metrics"
        self.plotDirectory = self.runDirectory / "plots"
        for directory in [
            self.scoreDirectory,
            self.evaluationDirectory,
            self.metricDirectory,
            self.plotDirectory,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        self.manifestPath = self.runDirectory / "manifest.json"
        self.manifest = {
            "schemaVersion": "2.0",
            "runId": self.runId,
            "mode": self.mode,
            "status": "running",
            "startedAt": datetime.now().isoformat(timespec="seconds"),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "plan": plan.toDict(),
            "artifacts": {
                "scores": [],
                "evaluations": [],
                "metrics": [],
                "windowMetrics": [],
            },
        }
        self.writeManifest()

    def createResult(self):
        return ExperimentResult(
            runId=self.runId,
            runDirectory=str(self.runDirectory),
            manifestPath=str(self.manifestPath),
        )

    def saveScores(self, frame, artifactId, metadata, result):
        path = self.scoreDirectory / f"{self.safeName(artifactId)}.csv"
        frame.to_csv(path, index=False)
        result.scoreFiles.append(str(path))
        self.register("scores", path, metadata)
        return str(path)

    def saveEvaluation(self, frame, artifactId, metadata, result):
        path = self.evaluationDirectory / f"{self.safeName(artifactId)}.csv"
        frame.to_csv(path, index=False)
        result.evaluationFiles.append(str(path))
        self.register("evaluations", path, metadata)
        return str(path)

    def saveMetrics(self, summary, windowFrame, artifactId, metadata, result):
        summaryPath = self.metricDirectory / f"{self.safeName(artifactId)}-summary.json"
        with open(summaryPath, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2, default=self.jsonDefault)
        result.metricFiles.append(str(summaryPath))
        self.register("metrics", summaryPath, metadata)

        if self.plan.output.saveWindowMetrics:
            windowPath = self.metricDirectory / f"{self.safeName(artifactId)}-windows.csv"
            windowFrame.to_csv(windowPath, index=False)
            result.windowMetricFiles.append(str(windowPath))
            self.register("windowMetrics", windowPath, metadata)
        return str(summaryPath)

    def finish(self, result, status="completed", error=None):
        self.manifest["status"] = status
        self.manifest["finishedAt"] = datetime.now().isoformat(timespec="seconds")
        if error is not None:
            self.manifest["error"] = str(error)
        self.writeManifest()
        return result

    def register(self, category, path, metadata):
        self.manifest["artifacts"][category].append({
            "path": os.path.relpath(path, self.runDirectory),
            "metadata": metadata,
        })
        self.writeManifest()

    def writeManifest(self):
        with open(self.manifestPath, "w", encoding="utf-8") as file:
            json.dump(self.manifest, file, ensure_ascii=False, indent=2, default=self.jsonDefault)

    def resolveRunId(self, runId):
        base = self.safeName(runId or datetime.now().strftime("%Y%m%d-%H%M%S"))
        outputDirectory = Path(self.plan.output.resolvedDirectory())
        candidate = base
        index = 2
        while (outputDirectory / candidate).exists():
            candidate = f"{base}-{index}"
            index += 1
        return candidate

    @staticmethod
    def buildArtifactId(*parts):
        cleanParts = [ResultManager.safeName(part) for part in parts if str(part).strip()]
        raw = "|".join(str(part) for part in parts)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
        return "-".join(cleanParts + [digest])


    @staticmethod
    def configurationHash(value):
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=ResultManager.jsonDefault)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def safeName(value):
        normalized = unicodedata.normalize("NFKD", str(value))
        asciiValue = normalized.encode("ascii", "ignore").decode("ascii")
        clean = "".join(
            character if character.isalnum() or character in {"-", "."} else "-"
            for character in asciiValue
        )
        return "-".join(part for part in clean.split("-") if part) or "artifact"

    @staticmethod
    def jsonDefault(value):
        if hasattr(value, "item"):
            return value.item()
        return str(value)
