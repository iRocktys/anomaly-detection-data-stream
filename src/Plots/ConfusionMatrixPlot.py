from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.Plots.PlotBase import PlotBase


class ConfusionMatrixPlot(PlotBase):
    def plot(self, source, outputPath=None, title="Binary Confusion Matrix", dpi=160):
        # Gera a matriz de confusão binária usando somente instâncias liberadas para avaliação.
        frame = self.readFrame(source)
        if "evaluationReady" in frame.columns:
            frame = frame[frame["evaluationReady"].astype(bool)]
        if frame.empty:
            raise ValueError("No evaluated instances are available for the confusion matrix.")
        true_values = frame["isAttack"].astype(int).to_numpy()
        predicted_values = frame["predictedLabel"].astype(int).to_numpy()
        matrix = np.array([
            [np.sum((true_values == 0) & (predicted_values == 0)), np.sum((true_values == 0) & (predicted_values == 1))],
            [np.sum((true_values == 1) & (predicted_values == 0)), np.sum((true_values == 1) & (predicted_values == 1))],
        ], dtype=int)
        fig, axis = plt.subplots(figsize=(6.5, 5.5))
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        axis.set_title(title, fontsize=14, fontweight="bold")
        axis.set_xlabel("Predicted Label")
        axis.set_ylabel("True Label")
        axis.set_xticks([0, 1], labels=["BENIGN", "ATTACK"])
        axis.set_yticks([0, 1], labels=["BENIGN", "ATTACK"])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return self.finish(fig, source, outputPath, "confusion_matrix.png", dpi)
