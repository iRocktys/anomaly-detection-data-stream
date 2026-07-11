import os

import matplotlib.pyplot as plt
import numpy as np

from src.Plots.PlotBase import PlotBase


class ScorePlot(PlotBase):
    def plot(
        self,
        results,
        attack_regions,
        title="Análise de Scores",
        discretization=0.5,
        scenario_name="General",
        discretization_strategy="fixed",
    ):
        fig, ax = plt.subplots(figsize=(15, 6))

        has_std = False
        algo_name = list(results.keys())[0] if results else "General"

        for idx, (name, data) in enumerate(results.items()):
            color = self.colors[idx % len(self.colors)]

            if "scores_mean" in data:
                scores_raw = np.asarray(data["scores_mean"], dtype=float)
                scores_std = np.asarray(
                    data.get("scores_std", np.zeros_like(scores_raw)),
                    dtype=float,
                )
                instances = np.arange(len(scores_raw))

            elif "scores" in data:
                scores_raw = np.asarray(data["scores"], dtype=float)
                scores_std = None
                instances = np.arange(len(scores_raw))

            else:
                continue

            trend_window = max(
                5,
                min(
                    25,
                    len(scores_raw) // 50 if len(scores_raw) >= 50 else 5,
                ),
            )

            trend_scores = self.moving_average(scores_raw, trend_window)

            ax.plot(
                instances,
                scores_raw,
                color=color,
                alpha=0.18,
                linewidth=0.8,
                zorder=2,
            )

            ax.plot(
                instances,
                trend_scores,
                color=color,
                alpha=0.95,
                linewidth=2.2,
                label=f"{name}",
                zorder=4,
            )

            if scores_std is not None and np.sum(scores_std) > 0:
                trend_std = self.moving_average(scores_std, trend_window)

                ax.fill_between(
                    instances,
                    trend_scores - trend_std,
                    trend_scores + trend_std,
                    color="gray",
                    alpha=0.22,
                    zorder=3,
                )

                has_std = True

        if str(discretization) != "params":
            ax.axhline(
                y=discretization,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"Threshold ({discretization})",
                zorder=5,
            )

        self.expand_y_limits(ax, kind="generic")

        self.add_attack_regions(
            ax=ax,
            attack_regions=attack_regions,
            alpha=0.58,
            show_legend=True,
            show_labels=True,
        )

        ax.set_title(f"{algo_name} - {title}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Score de Anomalia", fontsize=14)
        ax.set_xlabel("Instâncias", fontsize=14)

        handles, labels = ax.get_legend_handles_labels()

        if has_std:
            handles, labels = self.add_std_patch_to_legend(handles, labels)

        legend = ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(results) + 2,
            fontsize=12,
            frameon=False,
        )

        self.style_legend_patches(legend)

        ax.grid(True, alpha=0.3, linestyle=":", zorder=0)
        fig.subplots_adjust(bottom=0.2, top=0.94)

        output_dir = self.resolve_output_dir(
            algo_name=algo_name,
            discretization_strategy=discretization_strategy,
            scenario_name=scenario_name,
        )

        output_path = os.path.join(
            output_dir,
            f"{algo_name}_{title}_Scores.png",
        )

        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)