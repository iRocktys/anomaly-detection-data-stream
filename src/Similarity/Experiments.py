import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.Similarity.OneVsAll import SimilarityOneVsAll
from src.Similarity.Preprocessing import SimilarityPreprocessor
from src.Similarity.SimilarityAnalyzer import SimilarityAnalyzer
from src.Plots.SimilarityDistanceHeatmapPlot import SimilarityDistanceHeatmapPlot
from src.Plots.SimilarityDendrogramPlot import SimilarityDendrogramPlot


class SimilarityExperimentRunner:
    """Executa da preparação à similaridade; nunca recria cuts ou o Stream."""

    def run(self, datasetPath, outputDirectory, representation, selectionMethod="correlation",
            extractionMode="union", featuresPerAttack=10, minAttackOccurrences=2,
            selectedFeatures=None, downsample="random", nearConstantRatio=0.999,
            correlationThreshold=0.995, removeZeroColumns=True, removeAllMissingColumns=True,
            removeConstantColumns=True, imputation="median", normalization="standard",
            randomSeed=42, nEstimators=300, nJobs=-1, similarQuantile=0.33, distinctQuantile=0.67):
        if representation not in ("mixed_traffic", "attack_only", "stream"):
            raise ValueError("Unknown representation.")
        if selectionMethod not in (None, "correlation", "random_forest"):
            raise ValueError("Unknown selection method.")
        if representation == "stream" and downsample is not None:
            raise ValueError("Stream must use downsample=None to preserve sequential extraction.")
        if not 0 <= similarQuantile < distinctQuantile <= 1:
            raise ValueError("Require 0 <= similarQuantile < distinctQuantile <= 1.")
        # Subdiretório por estratégia: trocar union/recurrence nunca sobrescreve a outra.
        root = Path(outputDirectory) / (extractionMode or "disabled") if selectionMethod == "random_forest" else Path(outputDirectory)
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "experiment.json"
        configuration = {k: v for k, v in locals().copy().items() if k not in {"self", "root", "manifest_path"}}
        configuration["datasetPath"] = str(Path(datasetPath).resolve())
        configuration["outputDirectory"] = str(root.resolve())
        manifest = {"status": "running", "configuration": configuration}
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        try:
            prepared = SimilarityPreprocessor(
                nearConstantRatio=nearConstantRatio,
                correlationThreshold=correlationThreshold if selectionMethod == "correlation" else None,
                randomSeed=randomSeed, downsample=downsample, removeZeroColumns=removeZeroColumns,
                removeAllMissingColumns=removeAllMissingColumns, removeConstantColumns=removeConstantColumns,
                imputation=imputation, normalization=normalization,
                sampleUnit="flow" if representation == "stream" else "window",
            ).runFromFile(datasetPath, root, selectedFeatures)
            selection = None
            if selectionMethod == "random_forest":
                selection = SimilarityOneVsAll(featuresPerAttack, extractionMode, minAttackOccurrences,
                                               nEstimators, randomSeed, nJobs).runFromFile(
                    prepared["imputedDatasetPath"], prepared["featureSelectionPath"], root / "reports")
                features = selection["selectedFeatures"]
                frame = pd.read_csv(prepared["imputedDatasetPath"])
                processed = SimilarityPreprocessor(normalization=normalization).standardize(frame, features)
                processed.to_csv(prepared["standardizedDatasetPath"], index=False)
                payload = json.loads(Path(prepared["featureSelectionPath"]).read_text(encoding="utf-8"))
                payload.update(mahalanobisFeatures=features, mahalanobisFeatureCount=len(features),
                               oneVsAllSelectionPath=selection["selectionPath"])
                Path(prepared["featureSelectionPath"]).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            else:
                features = prepared["features"]
            analyzer = SimilarityAnalyzer()
            result = analyzer.runFromFile(prepared["standardizedDatasetPath"], prepared["featureSelectionPath"], root / "reports")
            levels_path, low, high = analyzer.classifyDistanceFile(
                result["pairwisePath"], root / "reports/pairwise_distance_levels.csv", similarQuantile, distinctQuantile)
            matrix = pd.read_csv(result["distanceMatrixPath"], index_col=0)
            heatmap = root / "plots/mahalanobis_distance_heatmap.png"
            dendrogram = root / "plots/attack_distance_dendrogram.png"
            SimilarityDistanceHeatmapPlot().plot(matrix, heatmap)
            SimilarityDendrogramPlot().plot(matrix, dendrogram)
            # Artefato antigo removido apenas dentro da saída deste cenário.
            (root / "reports/attack_feature_statistics.csv").unlink(missing_ok=True)
            sample_key = "selectedFlows" if representation == "stream" else "selectedWindows"
            sample_counts = {a: r[sample_key] for a, r in prepared["balanceSummary"].items()}
            manifest.update(status="complete", representation=representation, selectionMethod=selectionMethod,
                            sampleUnit="flow" if representation == "stream" else "window",
                            windowSeconds=None if representation == "stream" else 1,
                            extractionMode=extractionMode if selectionMethod == "random_forest" else None,
                            features=features, featureCount=len(features), sampleCounts=sample_counts,
                            sampleCount=sum(sample_counts.values()), shrinkage=result["shrinkage"],
                            relativeDistanceLimits={"similar": low, "distinct": high},
                            outputs={**result, "levelsPath": str(levels_path), "heatmapPath": str(heatmap),
                                     "dendrogramPath": str(dendrogram)},
                            note="Descriptive supervised feature selection; not held-out classification performance.")
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            return {**prepared, **result, "features": features, "selection": selection,
                    "levelsPath": str(levels_path), "heatmapPath": str(heatmap), "dendrogramPath": str(dendrogram),
                    "manifestPath": str(manifest_path)}
        except Exception as exc:
            manifest.update(status="failed", error=str(exc))
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            raise

    @staticmethod
    def compare(experimentsDirectory, outputDirectory):
        """Consolida somente execuções concluídas; ignora diagonais nas correlações de rankings."""
        root = Path(experimentsDirectory)
        comparisons, rows, feature_sets = [], [], {}
        for path in sorted(root.rglob("experiment.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                continue
            # Ignora os resultados antigos do Stream agregado incorretamente.
            if manifest.get("representation") == "stream" and manifest.get("sampleUnit") != "flow":
                continue
            scenario = path.parent.relative_to(root).as_posix()
            output = manifest["outputs"]
            pairs = pd.read_csv(output["levelsPath"])
            pairs["Rank"] = pairs["MahalanobisDistance"].rank(method="average")
            pairs = pairs.set_index(["AttackA", "AttackB"])[["MahalanobisDistance", "DistanceLevel", "Rank"]]
            pairs.columns = [f"{scenario}::{c}" for c in pairs.columns]
            comparisons.append(pairs)
            feature_sets[scenario] = set(manifest["features"])
            rows.append({"Scenario": scenario, "Representation": manifest["representation"],
                         "Selection": manifest["selectionMethod"], "SampleUnit": manifest.get("sampleUnit", "window"), "ExtractionMode": manifest["extractionMode"],
                         "FeatureCount": manifest["featureCount"], "SampleCount": manifest["sampleCount"],
                         "SamplesPerAttack": json.dumps(manifest["sampleCounts"], sort_keys=True),
                         "Downsample": manifest["configuration"]["downsample"], "Shrinkage": manifest["shrinkage"]})
        if not rows:
            raise ValueError("No completed experiments were found.")
        out = Path(outputDirectory)
        out.mkdir(parents=True, exist_ok=True)
        pairs = pd.concat(comparisons, axis=1)
        pairs.to_csv(out / "pairwise_scenario_comparison.csv")
        pd.DataFrame(rows).to_csv(out / "scenario_summary.csv", index=False)
        distances = pairs[[c for c in pairs if c.endswith("::MahalanobisDistance")]].copy()
        distances.columns = [c.rsplit("::", 1)[0] for c in distances]
        distances.corr(method="spearman").to_csv(out / "pairwise_rank_spearman.csv")
        overlap = pd.DataFrame(index=feature_sets, columns=feature_sets, dtype=float)
        for first, a in feature_sets.items():
            for second, b in feature_sets.items():
                overlap.loc[first, second] = len(a & b) / len(a | b) if a | b else np.nan
        overlap.to_csv(out / "feature_jaccard.csv")
        return {"summaryPath": str(out / "scenario_summary.csv"),
                "pairwiseComparisonPath": str(out / "pairwise_scenario_comparison.csv"),
                "rankCorrelationPath": str(out / "pairwise_rank_spearman.csv"),
                "featureOverlapPath": str(out / "feature_jaccard.csv")}
