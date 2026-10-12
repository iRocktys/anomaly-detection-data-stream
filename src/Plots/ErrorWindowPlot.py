import matplotlib.pyplot as plt

from src.Plots.PlotBase import PlotBase


class ErrorWindowPlot(PlotBase):
    def plot(self, source, attackSource=None, outputPath=None, windowSize=None, title=None, fpColumn="fp", fnColumn="fn", xColumn="windowEnd", attackAlpha=0.22, legendColumns=6, dpi=160):
        # Gera a quantidade de falsos positivos e falsos negativos em cada janela avaliada.
        frame = self.readFrame(source)
        if frame.empty:
            raise ValueError("The windowed metrics file is empty.")
        for column_name in (fpColumn, fnColumn):
            if column_name not in frame.columns:
                raise ValueError(f"The column '{column_name}' was not found.")
        resolved_title = title or f"False positives and false negatives by window ({windowSize} instances)"
        x_values = frame[xColumn] if xColumn in frame.columns else frame.index
        figure, axis = plt.subplots(figsize=(16, 6.5))
        handles = self.addAttackRegions(axis, attackSource, attackAlpha=attackAlpha)
        fp_line, = axis.plot(x_values, frame[fpColumn], color="#d1495b", linewidth=2.0, marker="o", markersize=4.5, label="False positives", zorder=5)
        fn_line, = axis.plot(x_values, frame[fnColumn], color="#3066be", linewidth=2.0, marker="o", markersize=4.5, label="False negatives", zorder=5)
        handles.extend([fp_line, fn_line])
        axis.set_title(resolved_title, fontsize=15, fontweight="bold")
        axis.set_xlabel("Window end instance")
        axis.set_ylabel("Error count")
        self.styleAxis(axis)
        self.applyLegend(axis, handles, legendColumns)
        figure.tight_layout()
        return self.finish(figure, source, outputPath, "fp_fn_windows.png", dpi)
