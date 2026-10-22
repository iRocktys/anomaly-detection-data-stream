from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.Plots.SimilaritySamples3DPlot import SimilaritySamples3DPlot


class SimilaritySamples2DPlot(SimilaritySamples3DPlot):
    def plot(
        self,
        analysis,
        outputPath,
        classes=None,
        pointSize=24,
        pointAlpha=0.18,
        dpi=300,
        title="PCA in Mahalanobis Space",
        centroidSize=65,
        centroidEdgeWidth=0.8,
        colors=None,
    ):
        # PC1 e PC2 correspondem aos mesmos eixos do gráfico 3D.
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
            samples,
            centers,
            classes,
            colorMap,
        ) = self.preparePlot(analysis, classes, colors)

        with plt.rc_context({
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#888888",
            "axes.labelcolor": "#333333",
            "text.color": "#333333",
        }):
            figure, axis = plt.subplots(
                figsize=(6.0, 5.0),
                facecolor="white",
            )

            try:
                axis.scatter(
                    samples.PC1,
                    samples.PC2,
                    s=pointSize,
                    c=[colorMap[label] for label in samples.Label],
                    alpha=pointAlpha,
                    edgecolors="none",
                    linewidths=0,
                    zorder=2,
                )

                axis.scatter(
                    centers.PC1,
                    centers.PC2,
                    s=centroidSize,
                    c=[colorMap[label] for label in centers.index],
                    alpha=1.0,
                    edgecolors="#202020",
                    linewidths=centroidEdgeWidth,
                    zorder=3,
                )

                ratios = projection["explainedVarianceRatio"]

                axis.set_xlabel(f"PC1 ({ratios[0]:.1%})")
                axis.set_ylabel(f"PC2 ({ratios[1]:.1%})")
                axis.set_title(title, pad=12)

                axis.set_aspect("equal", adjustable="datalim")
                axis.margins(0.08)
                axis.set_axisbelow(True)

                axis.grid(
                    color="#D4D9DE",
                    linewidth=0.5,
                    alpha=0.65,
                )

                axis.spines[["top", "right"]].set_visible(False)

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

                axis.legend(
                    handles=handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.14),
                    ncol=min(3, len(classes)),
                    frameon=False,
                )

                path = Path(outputPath)
                path.parent.mkdir(parents=True, exist_ok=True)

                figure.savefig(
                    path,
                    dpi=dpi,
                    facecolor="white",
                    bbox_inches="tight",
                    pad_inches=0.08,
                )
            finally:
                plt.close(figure)

        return str(path)