from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.Similarity.Preprocessing import SimilarityPreprocessor


class SimilaritySamples3DPlot:
    # Lê as janelas originais e trata as características sem usar centroides ou covariância externa.
    def loadSamples(self, datasetPath, selectedFeaturesPath=None):
        if not Path(datasetPath).is_file():
            raise FileNotFoundError(f"Dataset não encontrado: {datasetPath}")
        features = None
        if selectedFeaturesPath is not None:
            if not Path(selectedFeaturesPath).is_file():
                raise FileNotFoundError(f"Lista de características não encontrada: {selectedFeaturesPath}")
            selected = pd.read_csv(selectedFeaturesPath)
            if "Feature" not in selected:
                raise ValueError("A lista de características precisa conter a coluna Feature.")
            features = list(dict.fromkeys(selected["Feature"].dropna().astype(str)))
            if not features:
                raise ValueError("A lista de características está vazia.")
        prepared = SimilarityPreprocessor(nearConstantRatio=None, normalization=None,
                                          imputation="median").runFromFile(datasetPath, selectedFeatures=features)
        return prepared["frame"], prepared["features"]

    # Padroniza as características das janelas originais e ajusta o PCA diretamente a todas as amostras.
    def projectSamples(self, samples, features):
        values = samples[features].to_numpy(dtype=float)
        if len(samples) < 2 or not features or not np.isfinite(values).all():
            raise ValueError("O PCA precisa de amostras e características numéricas finitas.")
        standardized = StandardScaler().fit_transform(values)
        if not np.any(standardized):
            raise ValueError("As amostras não apresentam variabilidade para o PCA.")
        componentCount = min(3, len(samples) - 1, len(features))
        estimator = PCA(n_components=componentCount, svd_solver="full", whiten=False)
        coordinates = np.zeros((len(samples), 3))
        coordinates[:, :componentCount] = estimator.fit_transform(standardized)
        ratios = np.zeros(3)
        ratios[:componentCount] = estimator.explained_variance_ratio_
        projected = pd.DataFrame(coordinates, columns=["PC1", "PC2", "PC3"])
        projected["Label"] = samples["Label"].to_numpy()
        return projected, ratios

    # Gera a nuvem PCA das classes escolhidas com cores e legenda ajustadas à quantidade informada.
    def plotFromCsv(self, datasetPath, attacks, outputPath, selectedFeaturesPath=None,
                    samplesPerAttack=200, sampleRandomSeed=42,
                    pointSize=14, elevation=22, azimuth=40, dpi=300, title="PCA Projection of Attack Samples"):
        if attacks is None or isinstance(attacks, str):
            raise ValueError("attacks deve ser uma lista com os nomes das classes desejadas.")
        attacks = list(dict.fromkeys(str(label).strip().upper() for label in attacks))
        if not attacks or any(not label for label in attacks):
            raise ValueError("Informe pelo menos uma classe com nome válido em attacks.")
        if samplesPerAttack is not None and (isinstance(samplesPerAttack, bool)
                or not isinstance(samplesPerAttack, (int, np.integer)) or samplesPerAttack < 1):
            raise ValueError("samplesPerAttack deve ser um inteiro positivo ou None.")
        if not np.isfinite(pointSize) or pointSize <= 0:
            raise ValueError("pointSize deve ser positivo e finito.")
        samples, features = self.loadSamples(datasetPath, selectedFeaturesPath)
        missing = [label for label in attacks if label not in set(samples["Label"])]
        if missing:
            raise ValueError(f"Ataques ausentes no dataset: {missing}")
        projected, _ = self.projectSamples(samples, features)
        available = projected.loc[projected["Label"].isin(attacks)]
        count = None if samplesPerAttack is None else min(samplesPerAttack, int(available.groupby("Label").size().min()))
        selected = pd.concat([group if count is None else group.sample(n=count, random_state=sampleRandomSeed)
                              for _, group in available.groupby("Label", sort=True)], ignore_index=True)
        selected = selected.sample(frac=1, random_state=sampleRandomSeed)
        palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000", "#E69F00",
                   "#56B4E9", "#882255", "#332288", "#999933", "#44AA99", "#AA4499"]
        extraCount = max(len(attacks) - len(palette), 0)
        if extraCount <= 20:
            palette.extend(plt.get_cmap("tab20")(np.arange(extraCount)))
        else:
            palette.extend(plt.get_cmap("hsv")(np.linspace(0.0, 1.0, extraCount, endpoint=False)))
        colors = dict(zip(attacks, [to_rgba(color) for color in palette]))
        counts = selected.groupby("Label").size()
        legendColumns = min(3, len(attacks))
        legendRows = int(np.ceil(len(attacks) / legendColumns))
        figure = plt.figure(figsize=(7.2, 6.0 + (legendRows - 1) * 0.3))
        axis = figure.add_subplot(111, projection="3d")
        axis.scatter(selected.PC1, selected.PC2, selected.PC3, s=pointSize,
                     c=[colors[label] for label in selected["Label"]], marker="o", alpha=1.0,
                     edgecolors="none", linewidths=0, depthshade=False)
        for number, setter in enumerate([axis.set_xlabel, axis.set_ylabel, axis.set_zlabel]):
            setter(f"PC{number + 1}", labelpad=7, fontsize=10)
        values = selected[["PC1", "PC2", "PC3"]].to_numpy()
        midpoints = (values.min(axis=0) + values.max(axis=0)) / 2
        halfRange = max(float(np.ptp(values, axis=0).max()) * 0.56, 0.5)
        step = float(np.diff(MaxNLocator(nbins=5).tick_values(-halfRange, halfRange))[0])
        midpoints = np.round(midpoints / step) * step
        halfRange = max(float(np.ceil(np.abs(values - midpoints).max() * 1.03 / step) * step), step * 2)
        offsets = np.arange(-halfRange + step, halfRange - step * 0.1, step)
        for number, (setter, tickSetter) in enumerate(zip(
                [axis.set_xlim, axis.set_ylim, axis.set_zlim], [axis.set_xticks, axis.set_yticks, axis.set_zticks])):
            setter(midpoints[number] - halfRange, midpoints[number] + halfRange)
            tickSetter(midpoints[number] + offsets)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=elevation, azim=azimuth)
        for coordinateAxis in [axis.xaxis, axis.yaxis, axis.zaxis]:
            coordinateAxis.set_major_formatter(StrMethodFormatter("{x:.3g}"))
            coordinateAxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            coordinateAxis.pane.set_edgecolor("#cccccc")
        axis.tick_params(labelsize=9, pad=1)
        figure.suptitle(title, x=0.5, y=0.96, fontsize=12)
        handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=colors[label],
                          markeredgecolor="none", markersize=7,
                          label=f"{label}") for label in attacks]
        figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
                      ncol=legendColumns, frameon=False, fontsize=10, handletextpad=0.4, columnspacing=1.5)
        figure.subplots_adjust(left=0.12, right=0.97, bottom=min(0.45, 0.14 + (legendRows - 1) * 0.04), top=0.89)
        path = Path(outputPath)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(path, dpi=dpi, facecolor="white")
        finally:
            plt.close(figure)
        return str(path)
