from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class MetricsPlot:
    def plot(self, source, outputPath=None, title="Métricas por janela"):
        frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        xValues = frame["windowEnd"] if "windowEnd" in frame else frame.index
        for axis, metric, label in zip(
            axes,
            ["f1", "precision", "recall"],
            ["F1", "Precisão", "Recall"],
        ):
            axis.plot(xValues, frame[metric], linewidth=2)
            axis.set_ylabel(label)
            axis.set_ylim(-0.02, 1.02)
            axis.grid(True, alpha=0.25)
        axes[0].set_title(title)
        axes[-1].set_xlabel("Instância final da janela")
        fig.tight_layout()
        return self.finish(fig, source, outputPath, "metrics.png")

    @staticmethod
    def finish(fig, source, outputPath, suffix):
        if outputPath is None:
            base = Path(source).with_suffix("") if not isinstance(source, pd.DataFrame) else Path("plot")
            outputPath = str(base) + f"-{suffix}"
        Path(outputPath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outputPath, bbox_inches="tight")
        plt.close(fig)
        return str(outputPath)
