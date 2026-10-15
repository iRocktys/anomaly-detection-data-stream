import json
from pathlib import Path

import pandas as pd


class SimilarityFeatureAnalysis:
    def characterizeFromFile(self, datasetPath, featureSelectionPath, outputPath):
        # Reabre a base imputada e calcula estatísticas descritivas das features para cada ataque.
        frame = pd.read_csv(datasetPath, low_memory=False)
        features = self.loadFeatures(featureSelectionPath)
        rows = []
        for attack, group in frame.groupby("Attack", sort=True):
            for feature in features:
                values = pd.to_numeric(group[feature], errors="coerce")
                rows.append({
                    "Attack": attack,
                    "Feature": feature,
                    "Count": int(values.notna().sum()),
                    "Mean": float(values.mean()),
                    "Median": float(values.median()),
                    "Std": float(values.std(ddof=0)),
                    "Q1": float(values.quantile(0.25)),
                    "Q3": float(values.quantile(0.75)),
                    "Minimum": float(values.min()),
                    "Maximum": float(values.max()),
                })
        result = pd.DataFrame(rows)
        output_path = Path(outputPath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        return str(output_path)

    @staticmethod
    def loadFeatures(featureSelectionPath):
        # Carrega do JSON somente as features efetivamente selecionadas para a análise de Mahalanobis.
        payload = json.loads(Path(featureSelectionPath).read_text(encoding="utf-8"))
        return list(payload["mahalanobisFeatures"])
