import os

import matplotlib.pyplot as plt
import numpy as np

from src.Plots.PlotBase import PlotBase


class MetricsPlot(PlotBase):
    def plot(
        self,
        results,
        attack_regions=None,
        title="Métricas",
        window_size=1000,
        target_class=None,
        scenario_name="General",
        discretization_strategy="fixed",
    ):
        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(15, 12),
            sharex=True,
        )

        has_std = False
        algo_name = list(results.keys())[0] if results else "General"

        for idx, (name, data) in enumerate(results.items()):
            color = self.colors[idx % len(self.colors)]
            x_axis = data["instances"]

            if "f1_mean" in data:
                f1_mean = self.clean_values(data["f1_mean"])
                f1_std = self.clean_values(data["f1_std"])

                precision_mean = self.clean_values(data["precision_mean"])
                precision_std = self.clean_values(data["precision_std"])

                recall_mean = self.clean_values(data["recall_mean"])
                recall_std = self.clean_values(data["recall_std"])

                self._plot_metric_line(ax1, x_axis, f1_mean, name, color)
                self._plot_metric_line(ax2, x_axis, precision_mean, name, color)
                self._plot_metric_line(ax3, x_axis, recall_mean, name, color)

                if np.sum(f1_std) > 0:
                    self._fill_std_area(ax1, x_axis, f1_mean, f1_std)
                    self._fill_std_area(ax2, x_axis, precision_mean, precision_std)
                    self._fill_std_area(ax3, x_axis, recall_mean, recall_std)

                    has_std = True

            else:
                f1_values = self.clean_values(data.get("f1", data.get("f1_score", [])))
                precision_values = self.clean_values(data["precision"])
                recall_values = self.clean_values(data["recall"])

                self._plot_metric_line(ax1, x_axis, f1_values, name, color)
                self._plot_metric_line(ax2, x_axis, precision_values, name, color)
                self._plot_metric_line(ax3, x_axis, recall_values, name, color)

        for ax in [ax1, ax2, ax3]:
            self.expand_y_limits(ax, kind="percent")

            self.add_attack_regions(
                ax=ax,
                attack_regions=attack_regions,
                alpha=0.55,
                show_legend=True,
                show_labels=True,
            )

            ax.grid(True, alpha=0.3, linestyle=":", zorder=0)
            ax.tick_params(axis="both", which="major", labelsize=12)

        ax1.set_title(
            f"{algo_name} - {title} (Métricas por janela de {window_size} instâncias)",
            fontsize=14,
            fontweight="bold",
        )

        ax1.set_ylabel("F1-Score por janela (%)", fontsize=14)
        ax2.set_ylabel("Precision por janela (%)", fontsize=14)
        ax3.set_ylabel("Recall por janela (%)", fontsize=14)
        ax3.set_xlabel("Instâncias", fontsize=14)

        handles, labels = ax1.get_legend_handles_labels()

        if has_std:
            handles, labels = self.add_std_patch_to_legend(handles, labels)

        legend = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(results) + 2,
            fontsize=12,
            frameon=False,
        )

        self.style_legend_patches(legend)

        fig.subplots_adjust(bottom=0.15, top=0.94, hspace=0.35)

        output_dir = self.resolve_output_dir(
            algo_name=algo_name,
            discretization_strategy=discretization_strategy,
            scenario_name=scenario_name,
        )

        output_path = os.path.join(
            output_dir,
            f"{algo_name}_{title}_Metricas.png",
        )

        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_metric_line(
        self,
        ax,
        x_axis,
        values,
        name,
        color,
    ):
        ax.plot(
            x_axis,
            values,
            label=f"{name}",
            color=color,
            linewidth=2.5,
            zorder=3,
            marker="o",
            markersize=5,
        )

    def _fill_std_area(
        self,
        ax,
        x_axis,
        mean_values,
        std_values,
    ):
        ax.fill_between(
            x_axis,
            mean_values - std_values,
            mean_values + std_values,
            color="gray",
            alpha=0.3,
            zorder=2,
        )