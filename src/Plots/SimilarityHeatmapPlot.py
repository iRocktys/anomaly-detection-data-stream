from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class SimilarityHeatmapPlot:
    def plot(self, similarityMatrix, outputPath):
        # Gera o mapa de calor da similaridade entre ataques usando nomes e escala em inglês.
        output_path = Path(outputPath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        values = similarityMatrix.to_numpy(dtype=float)
        figure, axis = plt.subplots(figsize=(max(7, len(similarityMatrix) * 0.8), max(6, len(similarityMatrix) * 0.7)))
        image = axis.imshow(values, vmin=0.0, vmax=1.0, aspect="auto")
        axis.set_title("Attack Similarity Matrix")
        axis.set_xlabel("Attack")
        axis.set_ylabel("Attack")
        axis.set_xticks(np.arange(len(similarityMatrix.columns)), labels=similarityMatrix.columns, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(similarityMatrix.index)), labels=similarityMatrix.index)
        figure.colorbar(image, ax=axis, label="Similarity")
        figure.tight_layout()
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return output_path
