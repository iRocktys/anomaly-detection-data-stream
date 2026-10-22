from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from src.Data.FeatureAudit import buildFeatureRemovalReport
from src.Similarity.Preprocessing import SimilarityPreprocessor


class SimilarityAnalyzer:
    # Balanceia as classes somente para os cálculos estatísticos preservando o dataset usado pelo RF.
    def selectStatisticsSamples(self, frame, downsample="random", randomSeed=42):
        if downsample not in (None, "random"):
            raise ValueError("downsample deve ser 'random' ou None.")
        counts = frame.groupby("Label", sort=True).size()
        if counts.min() < 2:
            raise ValueError("Cada classe precisa ter pelo menos duas janelas para a covariância.")
        minimum, blocks, rows = int(counts.min()), [], []
        for label, group in frame.groupby("Label", sort=True):
            chosen = group.sample(n=minimum, random_state=randomSeed) if downsample == "random" else group
            blocks.append(chosen)
            rows.append({"Class": label, "RFWindows": len(group), "StatisticsWindows": len(chosen),
                         "DroppedForStatistics": len(group) - len(chosen)})
        selected = pd.concat(blocks).sort_index(kind="stable").reset_index(drop=True)
        return selected, pd.DataFrame(rows)

    # Calcula o centroide padronizado de cada classe incluindo o tráfego benigno.
    def calculateCentroids(self, frame, features):
        centers = frame.groupby("Label", sort=True)[features].mean()
        centers.index.name = "Class"
        return centers

    # Centraliza cada janela na média da própria classe para estimar a variabilidade interna comum.
    def buildWithinClassResiduals(self, frame, centroids, features):
        return np.vstack([group[features].to_numpy(dtype=float) - centroids.loc[label, features].to_numpy(dtype=float)
                          for label, group in frame.groupby("Label", sort=True)])

    # Estima a covariância interna regularizada de benignos e ataques pelo método Ledoit-Wolf.
    def estimateCommonCovariance(self, residuals, features):
        if not np.isfinite(residuals).all() or not np.any(residuals):
            raise ValueError("Não existe variabilidade interna suficiente para estimar a covariância.")
        estimator = LedoitWolf().fit(residuals)
        eigenvalues = np.linalg.eigvalsh(estimator.covariance_)
        if eigenvalues[-1] <= 0 or eigenvalues[0] <= eigenvalues[-1] * 1e-12:
            raise ValueError("Covariância degenerada; revise as características e as amostras selecionadas.")
        return pd.DataFrame(estimator.covariance_, index=features, columns=features), np.linalg.pinv(estimator.covariance_), float(estimator.shrinkage_)

    # Calcula a matriz de distâncias entre todas as classes utilizando uma covariância comum.
    def calculateDistances(self, centroids, inverseCovariance):
        classes = list(centroids.index)
        matrix = pd.DataFrame(np.zeros((len(classes), len(classes))), index=classes, columns=classes)
        for firstIndex, firstClass in enumerate(classes):
            for secondClass in classes[firstIndex + 1:]:
                difference = centroids.loc[firstClass].to_numpy() - centroids.loc[secondClass].to_numpy()
                distance = float(np.sqrt(max(float(difference @ inverseCovariance @ difference), 0)))
                matrix.loc[firstClass, secondClass] = matrix.loc[secondClass, firstClass] = distance
        return matrix

    # Executa os cálculos em memória e salva somente features selecionadas, centroides e matrizes.
    def run(self, selection, outputDirectory="output/Similarity", downsample="random", randomSeed=42,
            saveIntermediate=False, intermediateDirectory=None):
        frame, selectedFeatures = selection["frame"], list(selection["selectedFeatures"])
        if "BENIGN" not in set(frame.Label) or frame.Label.nunique() < 2:
            raise ValueError("A similaridade precisa incluir BENIGN e pelo menos um ataque.")
        if not selectedFeatures or not np.isfinite(frame[selectedFeatures].to_numpy(dtype=float)).all():
            raise ValueError("A análise precisa receber características selecionadas e finitas.")
        balanced, summary = self.selectStatisticsSamples(frame, downsample, randomSeed)
        if downsample == "random":
            SimilarityPreprocessor.saveIntermediateFrame(balanced, "04_Balanced", saveIntermediate, intermediateDirectory)
        constants = [f for f in selectedFeatures if balanced[f].nunique(dropna=False) <= 1]
        features = [f for f in selectedFeatures if f not in constants]
        removed = buildFeatureRemovalReport([{"Stage": "Covariance", "Feature": f,
                    "Reason": "Constante no subconjunto selecionado para os cálculos estatísticos"} for f in constants])
        if not features:
            raise ValueError(f"Todas as características ficaram constantes no subconjunto estatístico: {constants}")
        scaler = StandardScaler().fit(balanced[features]) if selection["normalization"] == "standard" else None
        standardized = balanced[["Timestamp"]].copy()
        standardized[features] = scaler.transform(balanced[features]) if scaler is not None else balanced[features].to_numpy()
        standardized["Label"] = balanced.Label
        if scaler is not None:
            SimilarityPreprocessor.saveIntermediateFrame(standardized, "05_Standardized", saveIntermediate, intermediateDirectory)
        centers = self.calculateCentroids(standardized, features)
        residuals = self.buildWithinClassResiduals(standardized, centers, features)
        covariance, inverse, shrinkage = self.estimateCommonCovariance(residuals, features)
        distances = self.calculateDistances(centers, inverse)
        selectedTable = pd.DataFrame([{
            "Feature": feature,
            "SelectedByClasses": ", ".join(label for label, values in selection["selectedFeaturesByClass"].items() if feature in values),
            "UsedForCovariance": feature in features,
        } for feature in selectedFeatures])
        output = Path(outputDirectory)
        output.mkdir(parents=True, exist_ok=True)
        selectedTable.to_csv(output / "selected_features.csv", index=False)
        centers.to_csv(output / "class_centroids_standardized.csv")
        covariance.to_csv(output / "common_within_class_covariance.csv")
        distances.to_csv(output / "mahalanobis_distance_matrix.csv")
        # Snapshot único da mesma execução, independente dos intermediários opcionais.
        np.savez_compressed(
            output / "mahalanobis_plot_data.npz",
            features=np.asarray(features, dtype=str),
            values=standardized[features].to_numpy(dtype=float),
            labels=standardized.Label.to_numpy(dtype=str),
            timestamps=standardized.Timestamp.astype(str).to_numpy(dtype=str),
            classes=centers.index.to_numpy(dtype=str),
            centroids=centers.to_numpy(dtype=float),
            covariance=covariance.to_numpy(dtype=float),
        )
        return {"centroids": centers, "covariance": covariance, "distanceMatrix": distances,
                "features": features, "removedFeatures": removed, "classSummary": summary,
                "selectedFeatures": selectedTable, "shrinkage": shrinkage,
                "statisticsFrame": standardized, "inverseCovariance": inverse}
