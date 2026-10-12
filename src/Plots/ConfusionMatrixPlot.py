import matplotlib.pyplot as plt
import numpy as np

from src.Plots.PlotBase import PlotBase


class ConfusionMatrixPlot(PlotBase):
    def plot(self, source, outputPath=None, title="Binary confusion matrix", dpi=160):
        # Gera a matriz de confusão binária usando somente instâncias marcadas como avaliáveis.
        frame = self.readFrame(source)
        if "evaluationReady" in frame.columns:
            frame = frame[frame["evaluationReady"].astype(bool)]
        if frame.empty:
            raise ValueError("There are no evaluated instances for the confusion matrix.")
        true_values = frame["isAttack"].astype(int).to_numpy()
        predicted_values = frame["predictedLabel"].astype(int).to_numpy()
        matrix = np.array([
            [np.sum((true_values == 0) & (predicted_values == 0)), np.sum((true_values == 0) & (predicted_values == 1))],
            [np.sum((true_values == 1) & (predicted_values == 0)), np.sum((true_values == 1) & (predicted_values == 1))],
        ], dtype=int)
        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=12)
        axis.set_xticks([0, 1], ["BENIGN", "ATTACK"])
        axis.set_yticks([0, 1], ["BENIGN", "ATTACK"])
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_title(title, fontsize=14, fontweight="bold")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        return self.finish(figure, source, outputPath, "confusion_matrix.png", dpi)
