import matplotlib.pyplot as plt

from src.Plots.PlotBase import PlotBase


class MetricsWindowPlot(PlotBase):
    defaultColors = ["#5f86ad", "#f0a43a", "#6f2dbd", "#2a9d8f"]

    def plot(self, source, attackSource=None, outputPath=None, windowSize=None, title=None, metricColumns=None, metricLabels=None, xColumn="windowEnd", attackAlpha=0.22, legendColumns=6, dpi=160):
        # Gera a evolução das métricas binárias acumuladas ao longo das janelas da stream.
        frame = self.readFrame(source)
        if frame.empty:
            raise ValueError("The windowed metrics file is empty.")
        metric_columns = list(metricColumns or ["cumulativeF1", "cumulativePrecision", "cumulativeRecall", "cumulativeAccuracy"])
        metric_labels = list(metricLabels or ["Cumulative F1-score", "Cumulative precision", "Cumulative recall", "Cumulative accuracy"])
        if len(metric_columns) != len(metric_labels):
            raise ValueError("The number of metrics must match the number of labels.")
        for column_name in metric_columns:
            if column_name not in frame.columns:
                raise ValueError(f"The column '{column_name}' was not found.")
        resolved_title = title or f"Cumulative binary metrics by window ({windowSize} instances)"
        x_values = frame[xColumn] if xColumn in frame.columns else frame.index
        figure, axis = plt.subplots(figsize=(16, 7))
        handles = self.addAttackRegions(axis, attackSource, attackAlpha=attackAlpha)
        for index, column_name in enumerate(metric_columns):
            line, = axis.plot(x_values, frame[column_name], color=self.defaultColors[index % len(self.defaultColors)], linewidth=2.0, marker="o", markersize=4.0, label=metric_labels[index], zorder=5)
            handles.append(line)
        axis.set_title(resolved_title, fontsize=15, fontweight="bold")
        axis.set_xlabel("Window end instance")
        axis.set_ylabel("Metric value")
        axis.set_ylim(-0.02, 1.02)
        self.styleAxis(axis)
        self.applyLegend(axis, handles, legendColumns)
        figure.tight_layout()
        return self.finish(figure, source, outputPath, "metrics_windows.png", dpi)
