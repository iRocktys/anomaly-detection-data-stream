import matplotlib.pyplot as plt

from src.Plots.PlotBase import PlotBase


class ErrorWindowPlot(PlotBase):
    def plot(self, source, attackSource=None, outputPath=None, windowSize=None, title=None, fpColumn="fp", fnColumn="fn", xColumn="windowEnd", attackNameColumn="labelName", attackFlagColumn="isAttack", attackAlpha=0.30, legendColumns=8, dpi=160):
        frame = self.readFrame(source)

        if frame.empty:
            raise ValueError("O arquivo de métricas janeladas está vazio.")

        for columnName in [fpColumn, fnColumn]:
            if columnName not in frame.columns:
                raise ValueError(f"A coluna '{columnName}' não foi encontrada.")

        if title is None:
            title = f"Falsos positivos e falsos negativos por janela — janela de {windowSize} instâncias"

        xValues = frame[xColumn] if xColumn in frame.columns else frame.index

        fig, axis = plt.subplots(figsize=(16, 6.5))

        attackHandles = self.addAttackRegions(axis, attackSource, attackNameColumn, attackFlagColumn, "instanceId", attackAlpha)

        fpLine, = axis.plot(xValues, frame[fpColumn], color="#d1495b", linewidth=2.0, marker="o", markersize=5, label="Falsos positivos", zorder=5)
        fnLine, = axis.plot(xValues, frame[fnColumn], color="#3066be", linewidth=2.0, marker="o", markersize=5, label="Falsos negativos", zorder=5)

        axis.set_title(title, fontsize=15, fontweight="bold")
        axis.set_xlabel(f"Final de cada janela de {windowSize} instâncias")
        axis.set_ylabel("Quantidade de erros")
        self.styleAxis(axis)

        handles = [fpLine, fnLine]
        handles.extend(attackHandles)
        self.applyLegend(axis, handles, legendColumns)

        fig.tight_layout()

        return self.finish(fig, source, outputPath, "fp_fn_windows.png", dpi)