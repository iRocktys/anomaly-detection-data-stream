from pathlib import Path

import matplotlib.pyplot as plt

from src.Plots.PlotBase import PlotBase


class DspotStudyPlot(PlotBase):
    def plot(
        self,
        source,
        outputPath,
        warmupSize,
        attackNameColumn="labelName",
        attackFlagColumn="isAttack",
        attackAlpha=0.30,
        dpi=160,
    ):
        # Plota scores, threshold incremental, calibração, anomalias detectadas e regiões reais de ataque do estudo DSPOT.
        frame = self.readFrame(source)
        output_path = Path(outputPath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        x_values = self.getXAxis(frame, "index")
        figure, axis = plt.subplots(figsize=(14, 6))

        attack_handles = self.addAttackRegions(
            axis,
            frame,
            attackNameColumn=attackNameColumn,
            attackFlagColumn=attackFlagColumn,
            xColumn="index",
            alpha=attackAlpha,
        )

        score_line, = axis.plot(
            frame["index"],
            frame["score"],
            linewidth=1.0,
            color="#5f86ad",
            label="Score",
            zorder=3,
        )

        threshold = frame["thresholdBefore"].where(
            frame["thresholdBefore"].notna(),
            frame["thresholdAfter"],
        )

        threshold_line, = axis.plot(
            frame["index"],
            threshold,
            linewidth=1.4,
            color="#d95f02",
            label="DSPOT Threshold",
            zorder=4,
        )

        calibration_line = axis.axvline(
            int(warmupSize) - 1,
            linestyle="--",
            linewidth=1.0,
            color="#1f77b4",
            label="Calibration End",
            zorder=5,
        )

        handles = [score_line, threshold_line, calibration_line]

        anomalies = frame[frame["classification"] == "anomaly"]
        if not anomalies.empty:
            anomaly_points = axis.scatter(
                anomalies["index"],
                anomalies["score"],
                s=22,
                color="#1f77b4",
                label="Anomaly",
                zorder=6,
            )
            handles.append(anomaly_points)

        axis.set_title("DSPOT Incremental Threshold")
        axis.set_xlabel("Instance")
        axis.set_ylabel("Score")

        self.styleAxis(axis)
        handles.extend(attack_handles)
        self.applyLegend(axis, handles, columns=6)

        figure.tight_layout()
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)

        return output_path
