import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class SimilarityAnalyzer:
    def loadAnalysisData(self, standardizedDatasetPath, featureSelectionPath):
        # Reabre a base padronizada e a lista de features usada no espaço comum da análise.
        frame = pd.read_csv(standardizedDatasetPath, low_memory=False)
        payload = json.loads(Path(featureSelectionPath).read_text(encoding="utf-8"))
        features = list(payload["mahalanobisFeatures"])
        missing = [feature for feature in features if feature not in frame.columns]
        if missing:
            raise KeyError(f"Missing Mahalanobis features: {missing}")
        return frame, features

    def calculateCentroids(self, standardizedFrame, features):
        # Calcula o centroide multivariado padronizado de cada ataque.
        return standardizedFrame.groupby("Attack", sort=True)[features].mean()

    def buildWithinAttackResiduals(self, standardizedFrame, centroids, features):
        # Centraliza cada janela em torno do centroide do próprio ataque para representar variabilidade interna.
        blocks = []
        for attack, group in standardizedFrame.groupby("Attack", sort=True):
            values = group[features].to_numpy(dtype=float)
            center = centroids.loc[attack, features].to_numpy(dtype=float)
            blocks.append(values - center)
        return np.vstack(blocks)

    def estimateCommonCovariance(self, residuals, features):
        # Estima uma única covariância within-attack regularizada por Ledoit-Wolf para todos os ataques.
        estimator = LedoitWolf().fit(residuals)
        covariance = pd.DataFrame(estimator.covariance_, index=features, columns=features)
        inverse = np.linalg.pinv(estimator.covariance_)
        return covariance, inverse, float(estimator.shrinkage_)

    def calculateDistances(self, centroids, inverseCovariance):
        # Calcula a distância de Mahalanobis entre todos os pares de centroides usando a mesma covariância.
        attacks = list(centroids.index)
        matrix = pd.DataFrame(np.zeros((len(attacks), len(attacks))), index=attacks, columns=attacks, dtype=float)
        rows = []
        for first_index, first_attack in enumerate(attacks):
            first = centroids.loc[first_attack].to_numpy(dtype=float)
            for second_index in range(first_index + 1, len(attacks)):
                second_attack = attacks[second_index]
                second = centroids.loc[second_attack].to_numpy(dtype=float)
                difference = first - second
                squared = float(difference @ inverseCovariance @ difference.T)
                distance = float(np.sqrt(max(squared, 0.0)))
                matrix.loc[first_attack, second_attack] = distance
                matrix.loc[second_attack, first_attack] = distance
                rows.append({"AttackA": first_attack, "AttackB": second_attack, "MahalanobisDistance": distance})
        pairwise = pd.DataFrame(rows).sort_values("MahalanobisDistance", ascending=True).reset_index(drop=True)
        return pairwise, matrix

    def updateFeatureSelection(self, featureSelectionPath, shrinkage, attackCount, featureCount):
        # Acrescenta ao JSON de seleção os parâmetros finais da covariância usada pela Mahalanobis.
        path = Path(featureSelectionPath)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["mahalanobisAnalysis"] = {
            "covarianceEstimator": "LedoitWolf",
            "ledoitWolfShrinkage": float(shrinkage),
            "attackCount": int(attackCount),
            "featureCount": int(featureCount),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def runFromFile(self, standardizedDatasetPath, featureSelectionPath, reportDirectory):
        # Reabre os artefatos processados, calcula centroides, Ledoit-Wolf e a matriz de distância sem gerar metadata separado.
        frame, features = self.loadAnalysisData(standardizedDatasetPath, featureSelectionPath)
        report_directory = Path(reportDirectory)
        report_directory.mkdir(parents=True, exist_ok=True)
        centroids = self.calculateCentroids(frame, features)
        residuals = self.buildWithinAttackResiduals(frame, centroids, features)
        covariance, inverse, shrinkage = self.estimateCommonCovariance(residuals, features)
        pairwise, distance_matrix = self.calculateDistances(centroids, inverse)
        centroid_path = report_directory / "attack_centroids_standardized.csv"
        covariance_path = report_directory / "common_within_attack_covariance.csv"
        distance_path = report_directory / "mahalanobis_distance_matrix.csv"
        pairwise_path = report_directory / "pairwise_mahalanobis_distances.csv"
        centroids.to_csv(centroid_path)
        covariance.to_csv(covariance_path)
        distance_matrix.to_csv(distance_path)
        pairwise.to_csv(pairwise_path, index=False)
        self.updateFeatureSelection(featureSelectionPath, shrinkage, len(centroids), len(features))
        return {
            "distanceMatrixPath": str(distance_path),
            "pairwisePath": str(pairwise_path),
            "centroidPath": str(centroid_path),
            "covariancePath": str(covariance_path),
            "featureSelectionPath": str(featureSelectionPath),
            "shrinkage": shrinkage,
        }

    def classifyDistanceFile(self, pairwisePath, outputPath, similarQuantile=0.33, distinctQuantile=0.67):
        # Reabre o ranking de distâncias e classifica os pares em próximos, intermediários e distintos por quantis.
        pairwise = pd.read_csv(pairwisePath)
        if pairwise.empty:
            pairwise["DistanceLevel"] = []
            similar_limit = np.nan
            distinct_limit = np.nan
        else:
            similar_limit = float(pairwise["MahalanobisDistance"].quantile(float(similarQuantile)))
            distinct_limit = float(pairwise["MahalanobisDistance"].quantile(float(distinctQuantile)))
            pairwise["DistanceLevel"] = np.where(
                pairwise["MahalanobisDistance"] <= similar_limit,
                "similar",
                np.where(pairwise["MahalanobisDistance"] >= distinct_limit, "distinct", "intermediate"),
            )
        output_path = Path(outputPath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pairwise.to_csv(output_path, index=False)
        return str(output_path), similar_limit, distinct_limit
