from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class ErrorPlot:
    def plot(self, source, outputPath=None, title="Erros por janela"):
        frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        xValues = frame["windowEnd"] if "windowEnd" in frame else frame.index
        axes[0].plot(xValues, frame["fp"], linewidth=2)
        axes[0].set_ylabel("Falsos positivos")
        axes[1].plot(xValues, frame["fn"], linewidth=2)
        axes[1].set_ylabel("Falsos negativos")
        axes[1].set_xlabel("Instância final da janela")
        axes[0].set_title(title)
        for axis in axes:
            axis.grid(True, alpha=0.25)
        fig.tight_layout()
        return self.finish(fig, source, outputPath, "errors.png")

    @staticmethod
    def finish(fig, source, outputPath, suffix):
        if outputPath is None:
            base = Path(source).with_suffix("") if not isinstance(source, pd.DataFrame) else Path("plot")
            outputPath = str(base) + f"-{suffix}"
        Path(outputPath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outputPath, bbox_inches="tight")
        plt.close(fig)
        return str(outputPath)
