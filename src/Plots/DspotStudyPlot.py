from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class DspotStudyPlot:
    def plot(self, source, outputPath, warmupSize):
        # Plota scores, threshold incremental e limite da calibração do estudo DSPOT.
        frame = pd.read_csv(source) if isinstance(source, (str, Path)) else source.copy()
        output_path = Path(outputPath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(14, 6))
        axis.plot(frame["index"], frame["score"], linewidth=1.0, label="Score")
        threshold = frame["thresholdBefore"].where(frame["thresholdBefore"].notna(), frame["thresholdAfter"])
        axis.plot(frame["index"], threshold, linewidth=1.3, label="DSPOT Threshold")
        axis.axvline(int(warmupSize) - 1, linestyle="--", linewidth=1.0, label="Calibration End")
        anomalies = frame[frame["classification"] == "anomaly"]
        if not anomalies.empty:
            axis.scatter(anomalies["index"], anomalies["score"], s=18, label="Anomaly")
        axis.set_title("DSPOT Incremental Threshold")
        axis.set_xlabel("Instance")
        axis.set_ylabel("Score")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return output_path
