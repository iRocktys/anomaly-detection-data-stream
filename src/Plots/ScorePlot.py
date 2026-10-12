import matplotlib.pyplot as plt

from src.Plots.PlotBase import PlotBase


class ScorePlot(PlotBase):
    scoreColors = ["#5f86ad", "#f0a43a", "#6f2dbd", "#2a9d8f"]

    def plot(self, source, outputPath=None, title="Anomaly scores and causal threshold", movingAverageColumns=None, showWarmup=True, attackAlpha=0.22, legendColumns=6, dpi=160):
        # Gera scores brutos, médias móveis e threshold causal da detecção de anomalias.
        frame = self.readFrame(source)
        if frame.empty or "rawScore" not in frame.columns:
            raise ValueError("The source does not contain anomaly detection scores.")
        x_values = self.getXAxis(frame)
        figure, axis = plt.subplots(figsize=(18, 7.5))
        handles = []
        handles.append(self.addWarmup(axis, frame, x_values, showWarmup=showWarmup))
        handles.extend(self.addAttackRegions(axis, frame, attackAlpha=attackAlpha))
        raw_line, = axis.plot(x_values, self.numericSeries(frame["rawScore"]), color="#657380", linewidth=0.7, alpha=0.45, label="Raw anomaly score", zorder=2)
        handles.append(raw_line)
        for index, column_name in enumerate(list(movingAverageColumns or [])):
            if column_name not in frame.columns:
                continue
            label = f"Moving average ({column_name.replace('scoreMa', '')})"
            line, = axis.plot(x_values, self.numericSeries(frame[column_name]), color=self.scoreColors[index % len(self.scoreColors)], linewidth=1.5, label=label, zorder=4 + index)
            handles.append(line)
        if "threshold" in frame.columns:
            threshold_line, = axis.plot(x_values, self.numericSeries(frame["threshold"]), color="#b71c1c", linewidth=1.8, label="Anomaly threshold", zorder=8)
            handles.append(threshold_line)
        axis.set_title(title, fontsize=15, fontweight="bold")
        axis.set_xlabel("Instance")
        axis.set_ylabel("Score")
        self.styleAxis(axis)
        self.applyLegend(axis, handles, legendColumns)
        figure.tight_layout()
        return self.finish(figure, source, outputPath, "scores.png", dpi)
