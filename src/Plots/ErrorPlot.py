import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from src.Plots.PlotBase import PlotBase

class ErrorPlot(PlotBase):
    def plot(
        self,
        results,
        attack_regions=None,
        title="Contagem de FP e FN",
        window_size=1000,
        scenario_name="General",
        discretization_strategy="fixed",
    ):
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(15, 9),
            sharex=True,
        )

        has_std = False
        algo_name = list(results.keys())[0] if results else "General"

        for idx, (name, data) in enumerate(results.items()):
            color = self.colors[idx % len(self.colors)]
            x_axis = data["instances"]

            if "fp_mean" in data:
                fp_mean = np.ceil(self.clean_values(data["fp_mean"]))
                fp_std = np.ceil(self.clean_values(data["fp_std"]))

                fn_mean = np.ceil(self.clean_values(data["fn_mean"]))
                fn_std = np.ceil(self.clean_values(data["fn_std"]))

                self._plot_error_line(ax1, x_axis, fp_mean, name, color)
                self._plot_error_line(ax2, x_axis, fn_mean, name, color)

                if np.sum(fp_std) > 0 or np.sum(fn_std) > 0:
                    self._fill_std_area(ax1, x_axis, fp_mean, fp_std)
                    self._fill_std_area(ax2, x_axis, fn_mean, fn_std)

                    has_std = True

            else:
                fp_values = np.ceil(self.clean_values(data.get("fp", [])))
                fn_values = np.ceil(self.clean_values(data.get("fn", [])))

                self._plot_error_line(ax1, x_axis, fp_values, name, color)
                self._plot_error_line(ax2, x_axis, fn_values, name, color)

        for ax in [ax1, ax2]:
            self.expand_y_limits(ax, kind="generic")

            self.add_attack_regions(
                ax=ax,
                attack_regions=attack_regions,
                alpha=0.55,
                show_legend=True,
                show_labels=True,
            )

            ax.grid(True, alpha=0.3, linestyle=":", zorder=0)
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax1.set_title(
            f"{algo_name} - {title} (FP/FN por janela de {window_size} instâncias)",
            fontsize=14,
            fontweight="bold",
        )

        ax1.set_ylabel("Falsos Positivos (FP)", fontsize=14)
        ax2.set_ylabel("Falsos Negativos (FN)", fontsize=14)
        ax2.set_xlabel("Instâncias", fontsize=14)

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
            f"{algo_name}_{title}_FP_FN.png",
        )

        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_error_line(
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