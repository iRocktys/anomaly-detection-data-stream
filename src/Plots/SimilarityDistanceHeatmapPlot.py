from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class SimilarityDistanceHeatmapPlot:
    def plot(self, distanceMatrix, outputPath, dpi=160):
        # Gera o mapa de calor da distância de Mahalanobis entre todos os ataques.
        output_path = Path(outputPath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        values = distanceMatrix.to_numpy(dtype=float)
        size = max(7.0, len(distanceMatrix) * 0.78)
        figure, axis = plt.subplots(figsize=(size, size * 0.85))
        image = axis.imshow(values, aspect="auto")
        axis.set_title("Attack Mahalanobis Distance Matrix")
        axis.set_xlabel("Attack")
        axis.set_ylabel("Attack")
        axis.set_xticks(np.arange(len(distanceMatrix.columns)), labels=distanceMatrix.columns, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(distanceMatrix.index)), labels=distanceMatrix.index)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, label="Mahalanobis Distance")
        figure.tight_layout()
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return str(output_path)
