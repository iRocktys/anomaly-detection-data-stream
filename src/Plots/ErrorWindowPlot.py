import matplotlib.pyplot as plt

from src.Plots.PlotBase import PlotBase


class ErrorWindowPlot(PlotBase):
    def plot(self, source, attackSource=None, outputPath=None, windowSize=None, title=None, fpColumn="fp", fnColumn="fn", xColumn="windowEnd", attackNameColumn="labelName", attackFlagColumn="isAttack", attackAlpha=0.30, legendColumns=8, dpi=160):
        # Gera e salva o gráfico correspondente aos dados recebidos.
        frame = self.readFrame(source)

        if frame.empty:
            raise ValueError("The windowed-metrics file is empty.")

        for columnName in [fpColumn, fnColumn]:
            if columnName not in frame.columns:
                raise ValueError(f"Column '{columnName}' was not found.")

        if title is None:
            title = f"False Positives and False Negatives by Window — {windowSize} Instances"

        xValues = frame[xColumn] if xColumn in frame.columns else frame.index

        fig, axis = plt.subplots(figsize=(16, 6.5))

        attackHandles = self.addAttackRegions(axis, attackSource, attackNameColumn, attackFlagColumn, "instanceId", attackAlpha)

        fpLine, = axis.plot(xValues, frame[fpColumn], color="#d1495b", linewidth=2.0, marker="o", markersize=5, label="False Positives", zorder=5)
        fnLine, = axis.plot(xValues, frame[fnColumn], color="#3066be", linewidth=2.0, marker="o", markersize=5, label="False Negatives", zorder=5)

        axis.set_title(title, fontsize=15, fontweight="bold")
        axis.set_xlabel(f"End of Each {windowSize}-Instance Window")
        axis.set_ylabel("Error Count")
        self.styleAxis(axis)

        handles = [fpLine, fnLine]
        handles.extend(attackHandles)
        self.applyLegend(axis, handles, legendColumns)

        fig.tight_layout()

        return self.finish(fig, source, outputPath, "fp_fn_windows.png", dpi)
