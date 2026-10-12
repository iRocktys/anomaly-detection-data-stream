from collections import OrderedDict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlotBase:
    attackColors = ["#f3aaaa", "#abc7ef", "#b9dfc1", "#efd39e"]
    warmupColor = "#c7c7c7"

    def readFrame(self, source):
        # Carrega uma cópia de um DataFrame ou de um CSV usado como fonte do gráfico.
        if isinstance(source, pd.DataFrame):
            return source.copy()
        return pd.read_csv(source)

    def getXAxis(self, frame, column="instanceId"):
        # Resolve o eixo X a partir da coluna de instância ou de uma sequência posicional.
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        return np.arange(len(frame), dtype=float)

    def numericSeries(self, values):
        # Converte uma série qualquer para vetor numérico tolerando valores inválidos.
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

    def addWarmup(self, axis, frame, xValues, showWarmup=True, warmupColumn="isWarmup", alpha=0.20):
        # Destaca visualmente a região de warm-up quando essa informação existe no resultado.
        if not showWarmup or warmupColumn not in frame.columns:
            return None
        warmup_mask = frame[warmupColumn].astype(bool).to_numpy()
        positions = np.flatnonzero(warmup_mask)
        if positions.size == 0:
            return None
        start = xValues[positions[0]]
        end = xValues[positions[-1]]
        axis.axvspan(start, end, facecolor=self.warmupColor, edgecolor=self.warmupColor, alpha=alpha, zorder=0)
        axis.axvline(end, color="#8a8a8a", linewidth=0.9, linestyle=":", alpha=0.65, zorder=3)
        return mpatches.Patch(facecolor=self.warmupColor, edgecolor="#8a8a8a", alpha=alpha, label="Warm-up")

    def addAttackRegions(self, axis, attackSource, attackFlagColumn="isAttack", xColumn="instanceId", attackAlpha=0.22):
        # Destaca blocos contíguos de ataque usando somente a verdade-terreno binária.
        if attackSource is None:
            return []
        frame = self.readFrame(attackSource)
        if frame.empty or attackFlagColumn not in frame.columns:
            return []
        x_values = self.getXAxis(frame, xColumn)
        attack_flags = frame[attackFlagColumn].astype(bool).to_numpy()
        regions = []
        start = None
        for position, is_attack in enumerate(attack_flags):
            if is_attack and start is None:
                start = position
            elif not is_attack and start is not None:
                regions.append((start, position - 1))
                start = None
        if start is not None:
            regions.append((start, len(frame) - 1))
        for region_start, region_end in regions:
            axis.axvspan(x_values[region_start], x_values[region_end], facecolor=self.attackColors[0], edgecolor=self.attackColors[0], alpha=attackAlpha, zorder=1)
        return [mpatches.Patch(facecolor=self.attackColors[0], edgecolor=self.attackColors[0], alpha=min(0.85, attackAlpha + 0.35), label="Attack region")] if regions else []

    def applyLegend(self, axis, handles, columns=6):
        # Remove rótulos repetidos e posiciona a legenda fora da área principal do gráfico.
        unique_handles = OrderedDict()
        for handle in handles:
            if handle is not None:
                unique_handles[handle.get_label()] = handle
        if not unique_handles:
            return
        axis.legend(
            list(unique_handles.values()),
            list(unique_handles.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=max(1, int(columns)),
            frameon=True,
            fontsize=10,
        )

    def finish(self, figure, source, outputPath, defaultName, dpi=160):
        # Salva e fecha a figura garantindo a criação do diretório de destino.
        if outputPath is None:
            outputPath = Path(defaultName) if isinstance(source, pd.DataFrame) else Path(source).parent / defaultName
        output_path = Path(outputPath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return str(output_path)

    def styleAxis(self, axis):
        # Aplica o acabamento visual comum a todos os gráficos do projeto.
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
