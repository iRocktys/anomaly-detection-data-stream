import matplotlib.pyplot as plt

from src.Plots.PlotBase import PlotBase


class MetricsWindowPlot(PlotBase):
    defaultColors = ["#5f86ad", "#f0a43a", "#6f2dbd", "#2a9d8f", "#d1495b"]

    def plot(self, source, attackSource=None, outputPath=None, windowSize=None, title=None, metricColumns=None, metricLabels=None, metricColors=None, xColumn="windowEnd", attackNameColumn="labelName", attackFlagColumn="isAttack", attackAlpha=0.30, legendColumns=8, dpi=160):
        # Gera e salva o gráfico correspondente aos dados recebidos.
        frame = self.readFrame(source)

        if frame.empty:
            raise ValueError("The windowed-metrics file is empty.")

        metricColumns = list(metricColumns or ["cumulativeF1", "cumulativePrecision", "cumulativeRecall", "cumulativeAccuracy"])
        metricLabels = list(metricLabels or ["Cumulative F1-score", "Cumulative Precision", "Cumulative Recall", "Cumulative Accuracy"])
        metricColors = list(metricColors or self.defaultColors)

        if len(metricColumns) != len(metricLabels):
            raise ValueError("The number of metrics must match the number of labels.")

        for columnName in metricColumns:
            if columnName not in frame.columns:
                raise ValueError(f"Column '{columnName}' was not found.")

        if title is None:
            title = f"Cumulative Metrics by Window — {windowSize} Instances"

        xValues = frame[xColumn] if xColumn in frame.columns else frame.index

        fig, axis = plt.subplots(figsize=(16, 7))

        attackHandles = self.addAttackRegions(axis, attackSource, attackNameColumn, attackFlagColumn, "instanceId", attackAlpha)
        handles = []

        for index, columnName in enumerate(metricColumns):
            line, = axis.plot(xValues, frame[columnName], color=metricColors[index % len(metricColors)], linewidth=2.0, marker="o", markersize=4.5, label=metricLabels[index], zorder=5)
            handles.append(line)

        axis.set_title(title, fontsize=15, fontweight="bold")
        axis.set_xlabel(f"End of Each {windowSize}-Instance Window")
        axis.set_ylabel("Metric Value")
        axis.set_ylim(-0.02, 1.02)
        self.styleAxis(axis)

        handles.extend(attackHandles)
        self.applyLegend(axis, handles, legendColumns)

        fig.tight_layout()

        return self.finish(fig, source, outputPath, "metrics_windows.png", dpi)
