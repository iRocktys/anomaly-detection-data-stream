import matplotlib.pyplot as plt
import numpy as np

from src.Plots.PlotBase import PlotBase


class ScorePlot(PlotBase):
    scoreColors = ["#5f86ad", "#f0a43a", "#6f2dbd", "#2a9d8f"]

    def plot(self, source, outputPath=None, title="Scores, médias móveis e limiar causal", rawScoreColumn="rawScore", scoreColumn="score", thresholdColumn="threshold", movingAverageColumns=None, movingAverageLabels=None, movingAverageColors=None, showWarmup=True, warmupColumn="isWarmup", attackNameColumn="labelName", attackFlagColumn="isAttack", attackAlpha=0.30, legendColumns=8, dpi=160):
        frame = self.readFrame(source)

        if frame.empty:
            raise ValueError("O arquivo de scores está vazio.")

        if rawScoreColumn not in frame.columns:
            raise ValueError(f"A coluna '{rawScoreColumn}' não foi encontrada.")

        xValues = self.getXAxis(frame)
        rawScores = self.numericSeries(frame[rawScoreColumn])

        fig, axis = plt.subplots(figsize=(18, 7.5))

        warmupHandle = self.addWarmup(axis, frame, xValues, showWarmup, warmupColumn)
        attackHandles = self.addAttackRegions(axis, frame, attackNameColumn, attackFlagColumn, "instanceId", attackAlpha)

        rawLine, = axis.plot(xValues, rawScores, color="#657380", linewidth=0.70, alpha=0.46, label="Score bruto", zorder=2)
        handles = [rawLine]

        movingAverageColumns = list(movingAverageColumns or [])
        movingAverageLabels = list(movingAverageLabels or movingAverageColumns)
        movingAverageColors = list(movingAverageColors or self.scoreColors)

        if len(movingAverageLabels) != len(movingAverageColumns):
            raise ValueError("A quantidade de rótulos deve ser igual à quantidade de colunas de média móvel.")

        for index, columnName in enumerate(movingAverageColumns):
            if columnName not in frame.columns:
                raise ValueError(f"A coluna de média móvel '{columnName}' não foi encontrada.")

            values = self.numericSeries(frame[columnName])
            line, = axis.plot(xValues, values, color=movingAverageColors[index % len(movingAverageColors)], linewidth=1.35 + (0.22 * index), alpha=0.95, label=movingAverageLabels[index], zorder=4 + index)
            handles.append(line)

        if scoreColumn in frame.columns and scoreColumn != rawScoreColumn:
            evaluatedScores = self.numericSeries(frame[scoreColumn])

            if not np.allclose(rawScores, evaluatedScores, equal_nan=True):
                evaluatedLine, = axis.plot(xValues, evaluatedScores, color="#39424a", linewidth=1.15, linestyle="--", alpha=0.72, label="Score usado na decisão", zorder=7)
                handles.append(evaluatedLine)

        if thresholdColumn in frame.columns:
            thresholdValues = self.numericSeries(frame[thresholdColumn])
            thresholdLine, = axis.plot(xValues, thresholdValues, color="#b71c1c", linewidth=1.80, label="Threshold", zorder=8)
            handles.append(thresholdLine)

        axis.set_title(title, fontsize=15, fontweight="bold")
        axis.set_xlabel("Instância")
        axis.set_ylabel("Score")
        self.styleAxis(axis)

        handles.append(warmupHandle)
        handles.extend(attackHandles)
        self.applyLegend(axis, handles, legendColumns)

        fig.tight_layout()

        return self.finish(fig, source, outputPath, "scores.png", dpi)