import re

import matplotlib.patches as mpatches
import numpy as np


class PlotBase:
    def __init__(self, target_names):
        self.target_names = target_names if target_names is not None else ["Normal", "Ataque"]

        self.colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
        ]

        self.bg_colors = [
            "#ff4d4d",
            "#4d88ff",
            "#2ecc71",
            "#ffb84d",
            "#b366ff",
            "#ff66b3",
            "#33cccc",
        ]

    def clean_attack_label(self, attack_idx):
        if attack_idx < len(self.target_names):
            label = str(self.target_names[attack_idx])
        else:
            label = f"Classe {attack_idx}"

        label = re.sub(r"(?i)^drdos[_\-\s]*", "", label)
        label = re.sub(r"(?i)^ddos[_\-\s]*", "", label)
        label = label.replace("_", " ").strip()

        return label if label else f"Classe {attack_idx}"

    def moving_average(self, values, window_size):
        values = np.asarray(values, dtype=float)

        if values.size == 0:
            return values

        if window_size is None or window_size <= 1:
            return values.copy()

        window_size = min(int(window_size), len(values))
        kernel = np.ones(window_size, dtype=float) / float(window_size)
        valid = np.convolve(values, kernel, mode="valid")
        prefix = [np.mean(values[:i + 1]) for i in range(window_size - 1)]

        return np.concatenate([np.asarray(prefix, dtype=float), valid])

    def clean_values(self, values):
        return np.asarray(
            [
                0.0 if value is None or np.isnan(value) else value
                for value in values
            ],
            dtype=float,
        )

    def expand_y_limits(self, ax, kind="generic"):
        ymin, ymax = ax.get_ylim()

        if ymin == ymax:
            delta = abs(ymax) * 0.1 if ymax != 0 else 1.0
            ax.set_ylim(ymin - delta, ymax + delta)
            return

        span = ymax - ymin
        pad_bottom = 0.03 * span

        if kind == "percent":
            new_top = ymax + max(5.0, 0.12 * max(abs(ymax), 100.0), 0.15 * span)
            new_bottom = ymin - max(1.0, pad_bottom)

            if ymax >= 95:
                new_top = max(new_top, 115.0)

            ax.set_ylim(new_bottom, new_top)

        else:
            new_bottom = ymin - pad_bottom
            new_top = ymax + max(0.15 * span, 0.08 * max(abs(ymax), 1.0))
            ax.set_ylim(new_bottom, new_top)

    def add_attack_regions(
        self,
        ax,
        attack_regions,
        alpha=0.55,
        show_legend=True,
        show_labels=True,
    ):
        if not attack_regions:
            return

        added_attack_labels = set()

        for start, end, attack_idx in attack_regions:
            attack_name = self.clean_attack_label(attack_idx)
            bg_color = self.bg_colors[attack_idx % len(self.bg_colors)]

            label_to_show = (
                attack_name
                if show_legend and attack_name not in added_attack_labels
                else ""
            )

            ax.axvspan(
                start,
                end,
                facecolor=bg_color,
                alpha=alpha,
                zorder=1,
                label=label_to_show,
            )

            mid = (start + end) / 2

            ax.axvline(
                mid,
                color=bg_color,
                alpha=0.95,
                linewidth=1.6,
                zorder=2,
            )

            if show_labels:
                ax.text(
                    mid,
                    0.89,
                    attack_name,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    color=bg_color,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.65,
                        "pad": 0.2,
                    },
                    clip_on=True,
                    zorder=10,
                )

            if label_to_show:
                added_attack_labels.add(attack_name)

    def add_std_patch_to_legend(self, handles, labels):
        if "Desvio Padrão" not in labels:
            handles.append(
                mpatches.Patch(
                    color="gray",
                    alpha=0.3,
                    label="Desvio Padrão",
                )
            )

            labels.append("Desvio Padrão")

        return handles, labels

    def style_legend_patches(self, legend):
        if legend is None:
            return

        for patch in legend.get_patches():
            patch.set_edgecolor("gray")
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)
