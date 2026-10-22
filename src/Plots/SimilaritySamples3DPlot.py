from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class SimilaritySamples3DPlot:
    @staticmethod
    def loadAnalysis(dataPath):
        path = Path(dataPath)

        if not path.is_file():
            raise FileNotFoundError(
                f"Base do PCA ausente: {path}. Execute a etapa de "
                "covariância e centroides uma vez com o "
                "SimilarityAnalyzer atualizado."
            )

        with np.load(path, allow_pickle=False) as data:
            features = data["features"].tolist()
            samples = pd.DataFrame(data["values"], columns=features)
            samples["Timestamp"] = data["timestamps"]
            samples["Label"] = data["labels"]

            return {
                "features": features,
                "statisticsFrame": samples,
                "centroids": pd.DataFrame(
                    data["centroids"],
                    index=data["classes"],
                    columns=features,
                ),
                "covariance": pd.DataFrame(
                    data["covariance"],
                    index=features,
                    columns=features,
                ),
            }

    def plotFromFile(self, dataPath, outputPath, **kwargs):
        return self.plot(
            self.loadAnalysis(dataPath),
            outputPath,
            **kwargs,
        )

    def projectSamples(self, analysis):
        # Usa a mesma matriz final e a mesma covariância de Mahalanobis.
        features = list(analysis["features"])
        samples = analysis["statisticsFrame"]
        centers = analysis["centroids"].loc[:, features]

        covariance = analysis["covariance"].loc[
            features, features
        ].to_numpy(dtype=float)

        values = samples[features].to_numpy(dtype=float)

        if (
            len(samples) < 2
            or not features
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "O PCA precisa receber a matriz final finita "
                "da análise de Mahalanobis."
            )

        if (
            set(samples.Label) != set(centers.index)
            or not np.isfinite(centers.to_numpy()).all()
        ):
            raise ValueError(
                "As classes e os centroides precisam corresponder "
                "à matriz final."
            )

        if (
            not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T)
        ):
            raise ValueError(
                "A covariância precisa ser finita e simétrica."
            )

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        if (
            eigenvalues[-1] <= 0
            or eigenvalues[0] <= eigenvalues[-1] * 1e-12
        ):
            raise ValueError(
                "A covariância de Mahalanobis é degenerada."
            )

        metricTransform = (
            eigenvectors * (1.0 / np.sqrt(eigenvalues))
        ) @ eigenvectors.T

        origin = values.mean(axis=0)
        transformed = (values - origin) @ metricTransform
        transformedCenters = (
            centers.to_numpy(dtype=float) - origin
        ) @ metricTransform

        componentCount = min(3, len(features), len(samples) - 1)

        estimator = PCA(
            n_components=componentCount,
            svd_solver="full",
            whiten=False,
        )

        coordinates = np.zeros((len(samples), 3))
        centerCoordinates = np.zeros((len(centers), 3))

        coordinates[:, :componentCount] = (
            estimator.fit_transform(transformed)
        )
        centerCoordinates[:, :componentCount] = (
            estimator.transform(transformedCenters)
        )

        ratios = np.zeros(3)
        ratios[:componentCount] = estimator.explained_variance_ratio_

        projected = pd.DataFrame(
            coordinates,
            columns=["PC1", "PC2", "PC3"],
        )
        projected["Timestamp"] = samples.Timestamp.to_numpy()
        projected["Label"] = samples.Label.to_numpy()

        projectedCenters = pd.DataFrame(
            centerCoordinates,
            index=centers.index,
            columns=["PC1", "PC2", "PC3"],
        )

        return {
            "samples": projected,
            "centroids": projectedCenters,
            "explainedVarianceRatio": ratios,
        }

    def preparePlot(self, analysis, classes, colors=None):
        projection = self.projectSamples(analysis)
        samples = projection["samples"]
        centers = projection["centroids"]
        allClasses = list(centers.index)

        if classes is None:
            classes = allClasses
        elif isinstance(classes, str):
            raise ValueError(
                "classes deve ser uma lista ou None."
            )
        else:
            classes = list(
                dict.fromkeys(
                    str(label).strip().upper()
                    for label in classes
                )
            )

        if not classes or set(classes) - set(allClasses):
            raise ValueError(
                "Informe somente classes presentes na análise "
                f"de Mahalanobis. Disponíveis: {allClasses}"
            )

        if isinstance(colors, str):
            raise ValueError(
                "colors deve ser uma lista de cores ou None."
            )

        palette = (
            [
                "#70B5E0",
                "#F2AA76",
                "#76C7A4",
                "#CF99C9",
                "#AAA3DF",
                "#E5C65C",
                "#77CDD2",
                "#E893A0",
                "#A9C96E",
                "#C8AC86",
                "#91A9D7",
                "#CBBCDE",
            ]
            if colors is None
            else list(colors)
        )

        if len(palette) < len(classes):
            raise ValueError(
                f"Informe pelo menos {len(classes)} cores, "
                "uma para cada classe selecionada."
            )

        # A primeira cor corresponde à primeira classe selecionada.
        colorMap = {
            label: to_rgba(color, alpha=1.0)
            for label, color in zip(classes, palette)
        }

        selected = samples.loc[samples.Label.isin(classes)]
        selectedCenters = centers.loc[classes]

        return (
            projection,
            selected,
            selectedCenters,
            classes,
            colorMap,
        )

    def plot(
        self,
        analysis,
        outputPath,
        classes=None,
        pointSize=24,
        pointAlpha=0.18,
        elevation=22,
        azimuth=40,
        dpi=300,
        title="PCA in Mahalanobis Space",
        centroidSize=65,
        centroidEdgeWidth=0.8,
        colors=None,
    ):
        if not np.isfinite(pointSize) or pointSize <= 0:
            raise ValueError(
                "pointSize deve ser positivo e finito."
            )

        if not np.isfinite(pointAlpha) or not 0 < pointAlpha < 1:
            raise ValueError(
                "pointAlpha deve estar entre 0 e 1, exclusivamente."
            )

        if not np.isfinite(centroidSize) or centroidSize <= 0:
            raise ValueError(
                "centroidSize deve ser positivo e finito."
            )

        if (
            not np.isfinite(centroidEdgeWidth)
            or centroidEdgeWidth < 0
        ):
            raise ValueError(
                "centroidEdgeWidth deve ser não negativo e finito."
            )

        (
            projection,
            selected,
            selectedCenters,
            classes,
            colorMap,
        ) = self.preparePlot(analysis, classes, colors)

        legendColumns = min(3, len(classes))
        legendRows = int(np.ceil(len(classes) / legendColumns))

        with plt.rc_context({
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#888888",
            "axes.labelcolor": "#333333",
            "text.color": "#333333",
        }):
            figure = plt.figure(
                figsize=(6.0, 5.5 + (legendRows - 1) * 0.22),
                facecolor="white",
            )

            axis = figure.add_subplot(
                111,
                projection="3d",
                computed_zorder=False,
            )

            try:
                axis.set_facecolor("white")

                axis.scatter(
                    selected.PC1,
                    selected.PC2,
                    selected.PC3,
                    s=pointSize,
                    c=[colorMap[label] for label in selected.Label],
                    marker="o",
                    alpha=pointAlpha,
                    edgecolors="none",
                    linewidths=0,
                    depthshade=False,
                    zorder=2,
                )

                axis.scatter(
                    selectedCenters.PC1,
                    selectedCenters.PC2,
                    selectedCenters.PC3,
                    s=centroidSize,
                    c=[
                        colorMap[label]
                        for label in selectedCenters.index
                    ],
                    marker="o",
                    alpha=1.0,
                    edgecolors="#202020",
                    linewidths=centroidEdgeWidth,
                    depthshade=False,
                    zorder=3,
                )

                ratios = projection["explainedVarianceRatio"]

                for number, setter in enumerate([
                    axis.set_xlabel,
                    axis.set_ylabel,
                    axis.set_zlabel,
                ]):
                    setter(
                        f"PC{number + 1} ({ratios[number]:.1%})",
                        labelpad=5,
                        fontsize=10,
                    )

                values = np.vstack([
                    selected[["PC1", "PC2", "PC3"]].to_numpy(),
                    selectedCenters.to_numpy(),
                ])

                midpoint = (
                    values.min(axis=0) + values.max(axis=0)
                ) / 2

                halfRange = max(
                    float(np.ptp(values, axis=0).max()) * 0.56,
                    0.5,
                )

                for number, setter in enumerate([
                    axis.set_xlim,
                    axis.set_ylim,
                    axis.set_zlim,
                ]):
                    setter(
                        midpoint[number] - halfRange,
                        midpoint[number] + halfRange,
                    )

                axis.set_box_aspect((1, 1, 1), zoom=1.0)
                axis.view_init(elev=elevation, azim=azimuth)

                for coordinate in [
                    axis.xaxis,
                    axis.yaxis,
                    axis.zaxis,
                ]:
                    coordinate.pane.fill = False
                    coordinate.pane.set_edgecolor((1, 1, 1, 0))
                    coordinate.set_major_locator(
                        MaxNLocator(nbins=4)
                    )
                    coordinate.set_major_formatter(
                        StrMethodFormatter("{x:.3g}")
                    )
                    coordinate._axinfo["grid"].update(
                        color=(0.83, 0.85, 0.87, 0.55),
                        linewidth=0.5,
                    )

                axis.tick_params(labelsize=9, pad=1)
                figure.suptitle(title, y=0.98, fontsize=12)

                handles = [
                    Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="none",
                        markerfacecolor=colorMap[label],
                        markeredgecolor="#202020",
                        markeredgewidth=centroidEdgeWidth,
                        markersize=7,
                        label=label,
                    )
                    for label in classes
                ]

                figure.legend(
                    handles=handles,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.015),
                    ncol=legendColumns,
                    frameon=False,
                    fontsize=10,
                )

                figure.subplots_adjust(
                    left=0.09,
                    right=0.91,
                    bottom=min(
                        0.4,
                        0.18 + (legendRows - 1) * 0.04,
                    ),
                    top=0.94,
                )

                path = Path(outputPath)
                path.parent.mkdir(parents=True, exist_ok=True)

                figure.savefig(
                    path,
                    dpi=dpi,
                    facecolor="white",
                    bbox_inches="tight",
                    pad_inches=0.06,
                    bbox_extra_artists=(
                        axis.xaxis.label,
                        axis.yaxis.label,
                        axis.zaxis.label,
                        *figure.legends,
                        *figure.texts,
                    ),
                )
            finally:
                plt.close(figure)

        return str(path)