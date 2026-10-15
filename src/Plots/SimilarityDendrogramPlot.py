from pathlib import Path

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


class SimilarityDendrogramPlot:
    def plot(self, distanceMatrix, outputPath, dpi=160):
        # Gera o dendrograma dos ataques diretamente a partir da matriz de distância de Mahalanobis.
        output_path = Path(outputPath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(distanceMatrix) < 2:
            raise ValueError("The dendrogram requires at least two attacks.")
        condensed = squareform(distanceMatrix.to_numpy(dtype=float), checks=False)
        linked = linkage(condensed, method="average")
        figure_width = max(12.0, len(distanceMatrix.index) * 1.15)
        figure, axis = plt.subplots(figsize=(figure_width, 6.5))
        dendrogram(linked, labels=list(distanceMatrix.index), ax=axis)
        axis.set_title("Attack Distance Dendrogram")
        axis.set_xlabel("Attack")
        axis.set_ylabel("Mahalanobis Distance")
        for label in axis.get_xticklabels():
            label.set_rotation(35)
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")
        figure.subplots_adjust(bottom=0.24)
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return str(output_path)
