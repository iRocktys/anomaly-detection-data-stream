import matplotlib.pyplot as plt

from src.Plots.PlotBase import PlotBase


class PredictionPlot(PlotBase):
    def plot(self, source, outputPath=None, title="Binary predictions over the stream", dpi=160):
        # Gera a comparação temporal entre rótulos binários reais e preditos pelo classificador.
        frame = self.readFrame(source)
        x_values = self.getXAxis(frame)
        figure, axis = plt.subplots(figsize=(18, 5.5))
        handles = self.addAttackRegions(axis, frame)
        true_line, = axis.step(x_values, frame["isAttack"].astype(int), where="post", linewidth=1.5, label="True label", zorder=4)
        predicted_line, = axis.step(x_values, frame["predictedLabel"].astype(int), where="post", linewidth=1.1, alpha=0.8, label="Predicted label", zorder=5)
        handles.extend([true_line, predicted_line])
        axis.set_title(title, fontsize=15, fontweight="bold")
        axis.set_xlabel("Instance")
        axis.set_ylabel("Binary label")
        axis.set_yticks([0, 1], ["BENIGN", "ATTACK"])
        self.styleAxis(axis)
        self.applyLegend(axis, handles, 4)
        figure.tight_layout()
        return self.finish(figure, source, outputPath, "predictions.png", dpi)
