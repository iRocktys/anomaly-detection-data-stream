import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import ProjectDefaults as defaults
from src.Plots.SimilarityDendrogramPlot import SimilarityDendrogramPlot
from src.Plots.SimilarityHeatmapPlot import SimilarityHeatmapPlot


class SimilarityAnalyzer:
    def __init__(self, outputRoot=None, randomSeed=42, maxSamplesPerAttack=None):
        # Inicializa a análise de similaridade estatística sem utilizar classificadores.
        self.outputRoot = Path(outputRoot or defaults.DEFAULT_SIMILARITY_OUTPUT_ROOT)
        self.randomSeed = int(randomSeed)
        self.maxSamplesPerAttack = None if maxSamplesPerAttack is None else int(maxSamplesPerAttack)

    def run(self, experimentName, csvPath, labelColumn="Label", selectedFeatures=None, normalLabels=None):
        # Calcula distâncias de Mahalanobis entre grupos de ataque e salva tabelas e gráficos do estudo.
        experiment_name = str(experimentName).strip()
        if not experiment_name:
            raise ValueError("experimentName must be provided explicitly.")
        frame = pd.read_csv(csvPath, low_memory=False)
        normal_labels = {str(value).strip().upper() for value in (normalLabels or defaults.DEFAULT_NORMAL_LABELS)}
        prepared, feature_names = self.prepareData(frame, labelColumn, selectedFeatures, normal_labels)
        pairwise_frame, distance_matrix, similarity_matrix = self.calculatePairwise(prepared, feature_names, labelColumn)
        output_directory = self.outputRoot / experiment_name
        plot_directory = output_directory / "plots"
        output_directory.mkdir(parents=True, exist_ok=True)
        plot_directory.mkdir(parents=True, exist_ok=True)
        pairwise_path = output_directory / "pairwise_distances.csv"
        distance_path = output_directory / "mahalanobis_distance_matrix.csv"
        similarity_path = output_directory / "similarity_matrix.csv"
        metadata_path = output_directory / "analysis_metadata.json"
        pairwise_frame.to_csv(pairwise_path, index=False)
        distance_matrix.to_csv(distance_path)
        similarity_matrix.to_csv(similarity_path)
        metadata = {
            "source": str(csvPath),
            "labelColumn": labelColumn,
            "features": list(feature_names),
            "attacks": list(distance_matrix.index),
            "distance": "Mahalanobis with Ledoit-Wolf covariance",
            "similarity": "1 / (1 + distance)",
            "maxSamplesPerAttack": self.maxSamplesPerAttack,
            "randomSeed": self.randomSeed,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        heatmap_path = SimilarityHeatmapPlot().plot(similarity_matrix, plot_directory / "similarity_heatmap.png")
        dendrogram_path = SimilarityDendrogramPlot().plot(distance_matrix, plot_directory / "similarity_dendrogram.png")
        return {
            "outputDirectory": str(output_directory),
            "pairwisePath": str(pairwise_path),
            "distanceMatrixPath": str(distance_path),
            "similarityMatrixPath": str(similarity_path),
            "metadataPath": str(metadata_path),
            "heatmapPath": str(heatmap_path),
            "dendrogramPath": str(dendrogram_path),
            "pairwiseFrame": pairwise_frame,
            "distanceMatrix": distance_matrix,
            "similarityMatrix": similarity_matrix,
        }

    def prepareData(self, frame, labelColumn, selectedFeatures, normalLabels):
        # Seleciona somente ataques e features numéricas finitas para a comparação estatística.
        if labelColumn not in frame.columns:
            raise KeyError(f"Missing label column: {labelColumn}.")
        data = frame.copy()
        labels = data[labelColumn].astype(str).str.strip()
        attack_mask = ~labels.str.upper().isin(normalLabels)
        data = data.loc[attack_mask].copy()
        data[labelColumn] = labels.loc[attack_mask]
        if data.empty:
            raise ValueError("No attack sample was found for similarity analysis.")
        if selectedFeatures is None:
            candidate_features = [column for column in data.columns if column != labelColumn]
        else:
            candidate_features = [str(column) for column in selectedFeatures]
        missing = [column for column in candidate_features if column not in data.columns]
        if missing:
            raise KeyError(f"Missing features in dataset: {missing}.")
        numeric = data[candidate_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        usable = [column for column in numeric.columns if numeric[column].notna().any()]
        if not usable:
            raise ValueError("No usable numeric feature was found.")
        numeric = numeric[usable]
        medians = numeric.median(axis=0).fillna(0.0)
        numeric = numeric.fillna(medians)
        prepared = numeric.copy()
        prepared[labelColumn] = data[labelColumn].to_numpy()
        if self.maxSamplesPerAttack is not None:
            prepared = self.sampleGroups(prepared, labelColumn)
        if prepared[labelColumn].nunique() < 2:
            raise ValueError("Similarity analysis requires at least two attack types.")
        return prepared.reset_index(drop=True), tuple(usable)

    def sampleGroups(self, frame, labelColumn):
        # Limita cada grupo de ataque de forma reproduzível quando uma amostragem máxima foi configurada.
        groups = []
        for _, group in frame.groupby(labelColumn, sort=True):
            if len(group) > self.maxSamplesPerAttack:
                group = group.sample(n=self.maxSamplesPerAttack, random_state=self.randomSeed)
            groups.append(group)
        return pd.concat(groups, ignore_index=True)

    def calculatePairwise(self, frame, featureNames, labelColumn):
        # Calcula a distância entre médias usando a covariância regularizada de cada par de ataques.
        attacks = sorted(frame[labelColumn].astype(str).unique())
        distances = pd.DataFrame(np.zeros((len(attacks), len(attacks))), index=attacks, columns=attacks, dtype=float)
        rows = []
        for first_index, first_attack in enumerate(attacks):
            first_values = frame.loc[frame[labelColumn] == first_attack, list(featureNames)].to_numpy(dtype=float)
            for second_index in range(first_index + 1, len(attacks)):
                second_attack = attacks[second_index]
                second_values = frame.loc[frame[labelColumn] == second_attack, list(featureNames)].to_numpy(dtype=float)
                distance = self.mahalanobisDistance(first_values, second_values)
                similarity = 1.0 / (1.0 + distance)
                distances.loc[first_attack, second_attack] = distance
                distances.loc[second_attack, first_attack] = distance
                rows.append({
                    "attackA": first_attack,
                    "attackB": second_attack,
                    "samplesA": int(len(first_values)),
                    "samplesB": int(len(second_values)),
                    "mahalanobisDistance": float(distance),
                    "similarity": float(similarity),
                })
        similarities = 1.0 / (1.0 + distances)
        np.fill_diagonal(similarities.values, 1.0)
        pairwise = pd.DataFrame(rows).sort_values("mahalanobisDistance", ascending=True).reset_index(drop=True)
        return pairwise, distances, similarities

    @staticmethod
    def mahalanobisDistance(firstValues, secondValues):
        # Mede a separação entre dois grupos com covariância Ledoit-Wolf para reduzir singularidades.
        combined = np.vstack([firstValues, secondValues])
        covariance = LedoitWolf().fit(combined).covariance_
        inverse = np.linalg.pinv(covariance)
        difference = np.mean(firstValues, axis=0) - np.mean(secondValues, axis=0)
        squared = float(difference @ inverse @ difference.T)
        return float(np.sqrt(max(squared, 0.0)))
