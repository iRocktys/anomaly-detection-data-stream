from collections import OrderedDict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlotBase:
    attackColors = ["#f3aaaa", "#abc7ef", "#b9dfc1", "#efd39e", "#d3b8eb", "#efb8d0", "#a9d9d9", "#c8c8a8"]
    warmupColor = "#c7c7c7"

    def readFrame(self, source):
        if isinstance(source, pd.DataFrame):
            return source.copy()

        return pd.read_csv(source)

    def getXAxis(self, frame, column="instanceId"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)

        return np.arange(len(frame), dtype=float)

    def numericSeries(self, values):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

    def addWarmup(self, axis, frame, xValues, showWarmup=True, warmupColumn="isWarmup", alpha=0.20):
        if not showWarmup or warmupColumn not in frame.columns:
            return None

        warmupMask = frame[warmupColumn].astype(bool).to_numpy()
        positions = np.flatnonzero(warmupMask)

        if positions.size == 0:
            return None

        start = xValues[positions[0]]
        end = xValues[positions[-1]]

        axis.axvspan(start, end, facecolor=self.warmupColor, edgecolor=self.warmupColor, alpha=alpha, zorder=0)
        axis.axvline(end, color="#8a8a8a", linewidth=0.9, linestyle=":", alpha=0.65, zorder=3)

        return mpatches.Patch(facecolor=self.warmupColor, edgecolor="#8a8a8a", alpha=alpha, label="Warmup")

    def addAttackRegions(self, axis, attackSource, attackNameColumn="labelName", attackFlagColumn="isAttack", xColumn="instanceId", alpha=0.30):
        if attackSource is None:
            return []

        frame = self.readFrame(attackSource)

        if frame.empty or attackFlagColumn not in frame.columns:
            return []

        xValues = self.getXAxis(frame, xColumn)
        attackFlags = frame[attackFlagColumn].astype(bool).to_numpy()

        if attackNameColumn in frame.columns:
            attackNames = frame[attackNameColumn].fillna("Ataque").astype(str).to_numpy()
        else:
            attackNames = np.full(len(frame), "Ataque")

        regions = []
        start = None
        currentName = None

        for position, isAttack in enumerate(attackFlags):
            attackName = attackNames[position]

            if isAttack and start is None:
                start = position
                currentName = attackName

            elif isAttack and attackName != currentName:
                regions.append((start, position - 1, currentName))
                start = position
                currentName = attackName

            elif not isAttack and start is not None:
                regions.append((start, position - 1, currentName))
                start = None
                currentName = None

        if start is not None:
            regions.append((start, len(frame) - 1, currentName))

        uniqueNames = list(dict.fromkeys(region[2] for region in regions))
        colorMap = {name: self.attackColors[index % len(self.attackColors)] for index, name in enumerate(uniqueNames)}

        for start, end, attackName in regions:
            axis.axvspan(xValues[start], xValues[end], facecolor=colorMap[attackName], edgecolor=colorMap[attackName], alpha=alpha, zorder=1)

        handles = []

        for attackName in uniqueNames:
            handles.append(
                mpatches.Patch(
                    facecolor=colorMap[attackName],
                    edgecolor=colorMap[attackName],
                    alpha=min(0.85, alpha + 0.35),
                    label=attackName,
                )
            )

        return handles

    def applyLegend(self, axis, handles, columns=8):
        uniqueHandles = OrderedDict()

        for handle in handles:
            if handle is not None:
                uniqueHandles[handle.get_label()] = handle

        axis.legend(
            list(uniqueHandles.values()),
            list(uniqueHandles.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=max(1, int(columns)),
            frameon=True,
            fontsize=10,
        )

    def finish(self, fig, source, outputPath, defaultName, dpi=160):
        if outputPath is None:
            if isinstance(source, pd.DataFrame):
                outputPath = Path(defaultName)
            else:
                outputPath = Path(source).parent / defaultName

        outputPath = Path(outputPath)
        outputPath.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(outputPath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        return str(outputPath)

    def styleAxis(self, axis):
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)