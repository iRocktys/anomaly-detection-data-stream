from collections import OrderedDict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Plots.PlotBase import PlotBase


class ScorePlot(PlotBase):
    """Plota scores, médias móveis, limiares causais e regiões multiclasse.

    A avaliação continua binária por meio da coluna ``isAttack``. A coluna
    ``labelName`` é usada somente para nomear e colorir as regiões de ataque.
    """

    scoreColors = ["#5f86ad", "#f0a43a", "#6f2dbd", "#2a9d8f"]
    thresholdColors = ["#b71c1c", "#d32f2f", "#e53935", "#8e0000"]
    thresholdStyles = ["-", "--", "-.", ":"]
    attackColors = [
        "#f3aaaa",
        "#abc7ef",
        "#b9dfc1",
        "#efd39e",
        "#d3b8eb",
        "#efb8d0",
        "#a9d9d9",
        "#c8c8a8",
        "#e8bca5",
        "#b9c5dc",
    ]

    def __init__(self, targetNames=None):
        super().__init__(targetNames)

    def plot(
        self,
        source,
        outputPath=None,
        title="Scores, médias móveis e limiar causal",
        rawScoreColumn="rawScore",
        scoreColumn="score",
        thresholdColumn="threshold",
        thresholdColumns=None,
        movingAverageWindows=(10, 50, 100),
        movingAverageLabels=None,
        movingAverageColors=None,
        showEvaluatedScore=False,
        attackNameColumn="labelName",
        attackFlagColumn="isAttack",
        attackLabelColumn="trueLabel",
        normalClassIndex=0,
        maxNormalGap=0,
        attackAlpha=0.30,
        showAttackLabels=False,
        attackLegendColumns=4,
        legendColumns=None,
        dpi=160,
    ):
        frame = self.readFrame(source)
        if frame.empty:
            raise ValueError("O artefato de scores está vazio.")
        if rawScoreColumn not in frame.columns:
            raise ValueError(
                f"Coluna de score bruto ausente: {rawScoreColumn}. "
                f"Disponíveis: {list(frame.columns)}"
            )

        xValues = self.getXAxis(frame)
        rawScores = self.numericSeries(frame[rawScoreColumn])
        attackRegions = self.buildNamedAttackRegions(
            frame=frame,
            xValues=xValues,
            attackNameColumn=attackNameColumn,
            attackFlagColumn=attackFlagColumn,
            attackLabelColumn=attackLabelColumn,
            normalClassIndex=normalClassIndex,
            maxNormalGap=maxNormalGap,
        )
        attackColorMap = self.buildAttackColorMap(attackRegions)

        fig, ax = plt.subplots(figsize=(18, 7.5))
        self.addNamedAttackRegions(
            ax=ax,
            regions=attackRegions,
            colorMap=attackColorMap,
            alpha=attackAlpha,
            showLabels=showAttackLabels,
        )

        seriesHandles = []
        rawLine, = ax.plot(
            xValues,
            rawScores,
            color="#657380",
            linewidth=0.70,
            alpha=0.46,
            label="Score bruto",
            zorder=2,
        )
        seriesHandles.append(rawLine)

        resolvedWindows = self.resolveMovingAverageWindows(movingAverageWindows)
        resolvedLabels = self.resolveMovingAverageLabels(
            resolvedWindows,
            movingAverageLabels,
        )
        resolvedColors = self.resolveMovingAverageColors(
            resolvedWindows,
            movingAverageColors,
        )
        for index, windowSize in enumerate(resolvedWindows):
            movingAverage = self.getMovingAverage(
                frame=frame,
                rawScores=rawScores,
                windowSize=windowSize,
            )
            line, = ax.plot(
                xValues,
                movingAverage,
                color=resolvedColors[index],
                linewidth=1.35 + (0.22 * index),
                alpha=0.95,
                label=resolvedLabels[index],
                zorder=4 + index,
            )
            seriesHandles.append(line)

        if showEvaluatedScore and scoreColumn in frame.columns:
            evaluatedScores = self.numericSeries(frame[scoreColumn])
            if not np.allclose(rawScores, evaluatedScores, equal_nan=True):
                evaluatedLine, = ax.plot(
                    xValues,
                    evaluatedScores,
                    color="#39424a",
                    linewidth=1.15,
                    linestyle="--",
                    alpha=0.72,
                    label="Score usado na decisão",
                    zorder=6,
                )
                seriesHandles.append(evaluatedLine)

        thresholdSpecs = self.resolveThresholdSpecs(
            frame=frame,
            thresholdColumn=thresholdColumn,
            thresholdColumns=thresholdColumns,
        )
        for index, (columnName, displayName) in enumerate(thresholdSpecs):
            thresholdHandle = self.plotThreshold(
                ax=ax,
                frame=frame,
                xValues=xValues,
                columnName=columnName,
                displayName=displayName,
                color=self.thresholdColors[index % len(self.thresholdColors)],
                lineStyle=self.thresholdStyles[index % len(self.thresholdStyles)],
                readyColumn="thresholdReady" if columnName == thresholdColumn else None,
            )
            if thresholdHandle is not None:
                seriesHandles.append(thresholdHandle)

        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        ax.set_xlabel("Instância", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.grid(True, alpha=0.20, linestyle=":", zorder=0)
        ax.margins(x=0)
        self.expand_y_limits(ax, kind="generic")

        attackHandles = [
            mpatches.Patch(
                facecolor=color,
                edgecolor=color,
                alpha=min(0.85, attackAlpha + 0.35),
                label=attackName,
            )
            for attackName, color in attackColorMap.items()
        ]

        legendHandles = seriesHandles + attackHandles
        if legendHandles:
            if legendColumns is None:
                resolvedLegendColumns = min(7, len(legendHandles))
            else:
                resolvedLegendColumns = max(
                    1,
                    min(int(legendColumns), len(legendHandles)),
                )

            fig.subplots_adjust(bottom=0.24)
            fig.text(
                0.5,
                0.085,
                "Legenda",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
            fig.legend(
                handles=legendHandles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.015),
                ncol=resolvedLegendColumns,
                frameon=False,
                fontsize=10,
                handlelength=2.8,
                columnspacing=1.6,
            )
        else:
            fig.subplots_adjust(bottom=0.13)

        output = self.finish(
            fig=fig,
            source=source,
            outputPath=outputPath,
            suffix="scores.png",
            dpi=dpi,
        )
        return output

    def plotThreshold(
        self,
        ax,
        frame,
        xValues,
        columnName,
        displayName,
        color,
        lineStyle,
        readyColumn=None,
    ):
        if columnName not in frame.columns:
            return None

        thresholdValues = self.numericSeries(frame[columnName])
        thresholdValues[~np.isfinite(thresholdValues)] = np.nan
        if np.all(np.isnan(thresholdValues)):
            return None

        readyMask = None
        if readyColumn and readyColumn in frame.columns:
            readyMask = frame[readyColumn].fillna(False).astype(bool).to_numpy()

        if readyMask is not None and np.any(~readyMask):
            warmupValues = thresholdValues.copy()
            warmupValues[readyMask] = np.nan
            if np.any(np.isfinite(warmupValues)):
                ax.plot(
                    xValues,
                    warmupValues,
                    color=color,
                    linewidth=1.35,
                    linestyle="--",
                    alpha=0.38,
                    zorder=7,
                )

        visibleValues = thresholdValues.copy()
        if readyMask is not None:
            visibleValues[~readyMask] = np.nan
            if not np.any(np.isfinite(visibleValues)):
                visibleValues = thresholdValues

        line, = ax.plot(
            xValues,
            visibleValues,
            color=color,
            linewidth=2.55,
            linestyle=lineStyle,
            alpha=0.98,
            label=displayName,
            zorder=9,
        )
        return line

    def buildNamedAttackRegions(
        self,
        frame,
        xValues,
        attackNameColumn,
        attackFlagColumn,
        attackLabelColumn,
        normalClassIndex,
        maxNormalGap,
    ):
        attackFlags = self.resolveAttackFlags(
            frame,
            attackFlagColumn,
            attackLabelColumn,
            normalClassIndex,
        )
        labels = self.resolveAttackNames(
            frame,
            attackNameColumn,
            attackLabelColumn,
        )
        attackPositions = np.flatnonzero(attackFlags)
        if attackPositions.size == 0:
            return []

        maximumGap = max(0, int(maxNormalGap)) + 1
        regions = []
        startPosition = int(attackPositions[0])
        lastPosition = startPosition
        currentName = labels[startPosition]

        for positionValue in attackPositions[1:]:
            position = int(positionValue)
            attackName = labels[position]
            sameRegion = (
                attackName == currentName
                and position - lastPosition <= maximumGap
            )
            if not sameRegion:
                regions.append({
                    "start": self.regionStart(xValues, startPosition),
                    "end": self.regionEnd(xValues, lastPosition),
                    "name": currentName,
                })
                startPosition = position
                currentName = attackName
            lastPosition = position

        regions.append({
            "start": self.regionStart(xValues, startPosition),
            "end": self.regionEnd(xValues, lastPosition),
            "name": currentName,
        })
        return regions

    def resolveAttackFlags(
        self,
        frame,
        attackFlagColumn,
        attackLabelColumn,
        normalClassIndex,
    ):
        if attackFlagColumn in frame.columns:
            values = pd.to_numeric(frame[attackFlagColumn], errors="coerce").fillna(0)
            return values.astype(int).to_numpy() == 1
        if attackLabelColumn in frame.columns:
            values = pd.to_numeric(frame[attackLabelColumn], errors="coerce")
            return values.fillna(normalClassIndex).astype(int).to_numpy() != int(normalClassIndex)
        raise ValueError(
            f"O artefato precisa possuir {attackFlagColumn} ou {attackLabelColumn}."
        )

    def resolveAttackNames(self, frame, attackNameColumn, attackLabelColumn):
        names = []
        for position in range(len(frame)):
            name = None
            if attackNameColumn in frame.columns:
                value = frame.iloc[position][attackNameColumn]
                if pd.notna(value) and str(value).strip():
                    name = str(value).strip()
            if name is None and attackLabelColumn in frame.columns:
                rawLabel = frame.iloc[position][attackLabelColumn]
                try:
                    labelIndex = int(rawLabel)
                    if 0 <= labelIndex < len(self.target_names):
                        name = str(self.target_names[labelIndex]).strip()
                    else:
                        name = f"Classe {labelIndex}"
                except (TypeError, ValueError):
                    name = str(rawLabel).strip()
            names.append(name or "Ataque")
        return names

    def buildAttackColorMap(self, regions):
        colorMap = OrderedDict()
        for region in regions:
            attackName = region["name"]
            if attackName not in colorMap:
                colorMap[attackName] = self.attackColors[
                    len(colorMap) % len(self.attackColors)
                ]
        return colorMap

    def addNamedAttackRegions(self, ax, regions, colorMap, alpha, showLabels):
        for region in regions:
            attackName = region["name"]
            color = colorMap[attackName]
            ax.axvspan(
                region["start"],
                region["end"],
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=alpha,
                zorder=1,
            )
            if showLabels:
                middle = (region["start"] + region["end"]) / 2.0
                ax.text(
                    middle,
                    0.96,
                    attackName,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#4f4f4f",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": color,
                        "alpha": 0.82,
                        "pad": 1.5,
                    },
                    clip_on=True,
                    zorder=10,
                )

    def resolveThresholdSpecs(self, frame, thresholdColumn, thresholdColumns):
        if thresholdColumns is None:
            if thresholdColumn not in frame.columns:
                return []
            return [(thresholdColumn, self.thresholdDisplayName(frame))]

        if isinstance(thresholdColumns, dict):
            return [
                (str(columnName), str(displayName))
                for columnName, displayName in thresholdColumns.items()
                if str(columnName) in frame.columns
            ]

        specs = []
        for item in thresholdColumns:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                columnName, displayName = item
            else:
                columnName = str(item)
                displayName = f"Limiar {columnName}"
            if str(columnName) in frame.columns:
                specs.append((str(columnName), str(displayName)))
        return specs

    def thresholdDisplayName(self, frame):
        if "thresholdStrategy" not in frame.columns:
            return "Limiar"
        strategies = (
            frame["thresholdStrategy"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )
        if len(strategies) != 1:
            return "Limiar"
        strategy = strategies[0]
        names = {
            "fixed": "Limiar fixo",
            "incrementalMeanStd": "Limiar média/desvio incremental",
            "incrementalmeanstd": "Limiar média/desvio incremental",
            "spot": "Limiar SPOT",
            "dspot": "Limiar DSPOT",
        }
        return names.get(strategy, f"Limiar {strategy}")

    def getMovingAverage(self, frame, rawScores, windowSize):
        artifactColumn = f"scoreMa{int(windowSize)}"
        if artifactColumn in frame.columns:
            return self.numericSeries(frame[artifactColumn])
        return (
            pd.Series(rawScores)
            .rolling(window=int(windowSize), min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )

    @staticmethod
    def resolveMovingAverageWindows(windows):
        resolved = []
        for window in windows or []:
            value = int(window)
            if value < 1:
                raise ValueError("As janelas das médias móveis devem ser maiores que zero.")
            if value not in resolved:
                resolved.append(value)
        return resolved

    @staticmethod
    def resolveMovingAverageLabels(windows, labels):
        if labels is not None:
            if len(labels) != len(windows):
                raise ValueError(
                    "movingAverageLabels deve possuir o mesmo tamanho de movingAverageWindows."
                )
            return [str(label) for label in labels]
        descriptions = ["curta", "média", "longa"]
        return [
            f"Média móvel {descriptions[index] if index < len(descriptions) else index + 1} ({window})"
            for index, window in enumerate(windows)
        ]

    def resolveMovingAverageColors(self, windows, colors):
        if colors is not None:
            if len(colors) != len(windows):
                raise ValueError(
                    "movingAverageColors deve possuir o mesmo tamanho de movingAverageWindows."
                )
            return [str(color) for color in colors]
        return [
            self.scoreColors[index % len(self.scoreColors)]
            for index in range(len(windows))
        ]

    @staticmethod
    def getXAxis(frame):
        if "instanceId" in frame.columns:
            return pd.to_numeric(frame["instanceId"], errors="coerce").to_numpy(dtype=float)
        return np.arange(len(frame), dtype=float)

    @staticmethod
    def regionStart(xValues, position):
        if len(xValues) <= 1:
            return float(xValues[position]) - 0.5
        if position == 0:
            step = xValues[1] - xValues[0]
        else:
            step = xValues[position] - xValues[position - 1]
        return float(xValues[position] - (step / 2.0))

    @staticmethod
    def regionEnd(xValues, position):
        if len(xValues) <= 1:
            return float(xValues[position]) + 0.5
        if position == len(xValues) - 1:
            step = xValues[-1] - xValues[-2]
        else:
            step = xValues[position + 1] - xValues[position]
        return float(xValues[position] + (step / 2.0))

    @staticmethod
    def numericSeries(series):
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

    @staticmethod
    def readFrame(source):
        return source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)

    @staticmethod
    def finish(fig, source, outputPath, suffix, dpi=160):
        if outputPath is None:
            base = Path(source).with_suffix("") if not isinstance(source, pd.DataFrame) else Path("plot")
            outputPath = str(base) + f"-{suffix}"
        Path(outputPath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outputPath, bbox_inches="tight", dpi=int(dpi))
        plt.close(fig)
        return str(outputPath)