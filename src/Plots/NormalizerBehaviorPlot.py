import os
from collections import OrderedDict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Plots.PlotBase import PlotBase


class NormalizerBehaviorPlot(PlotBase):
    def plot_from_files(
        self,
        files_by_model,
        output_dir="output/ExpNormalizers/Plots",
        score_column="score",
        score_smoothing_column="score_ma_10",
        normal_class_idx=0,
        feature_columns=None,
    ):
        os.makedirs(output_dir, exist_ok=True)

        generated_paths = []

        for _, files_by_normalization in files_by_model.items():
            for _, csv_path in files_by_normalization.items():
                output_path = self.plot_single_file(
                    csv_path=csv_path,
                    output_dir=output_dir,
                    score_column=score_column,
                    score_smoothing_column=score_smoothing_column,
                    normal_class_idx=normal_class_idx,
                    feature_columns=feature_columns,
                )

                generated_paths.append(output_path)

        return generated_paths

    def plot_single_file(
        self,
        csv_path,
        output_dir="output/ExpNormalizers/Plots",
        score_column="score",
        score_smoothing_column="score_ma_10",
        normal_class_idx=0,
        feature_columns=None,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError(f"Arquivo vazio: {csv_path}")

        model_name = (
            str(df["model_name"].iloc[0])
            if "model_name" in df.columns
            else "UnknownModel"
        )

        model_code = (
            str(df["model_code"].iloc[0])
            if "model_code" in df.columns
            else model_name
        )

        normalization_strategy = (
            str(df["normalization_strategy"].iloc[0])
            if "normalization_strategy" in df.columns
            else "unknown"
        )

        dataset_name = (
            str(df["dataset"].iloc[0])
            if "dataset" in df.columns
            else "Dataset"
        )

        feature_columns = feature_columns or self._infer_feature_columns(df)

        if not feature_columns:
            raise ValueError(
                f"Nenhuma coluna de feature foi encontrada no arquivo: {csv_path}"
            )

        if len(feature_columns) > 33:
            feature_columns = feature_columns[:33]

        attack_regions = self._extract_attack_regions_named(
            df=df,
            normal_class_idx=normal_class_idx,
        )

        attack_color_map = self._build_attack_color_map(attack_regions)

        fig = plt.figure(figsize=(24, 14))

        grid = fig.add_gridspec(
            nrows=4,
            ncols=11,
            height_ratios=[1.0, 1.0, 1.0, 1.35],
            hspace=0.55,
            wspace=0.28,
        )

        for idx, feature_name in enumerate(feature_columns):
            row = idx // 11
            col = idx % 11

            ax = fig.add_subplot(grid[row, col])

            self._plot_single_feature_axis(
                ax=ax,
                df=df,
                feature_name=feature_name,
                attack_regions=attack_regions,
                attack_color_map=attack_color_map,
            )

        score_ax = fig.add_subplot(grid[3, :])

        self._plot_score_axis(
            ax=score_ax,
            df=df,
            attack_regions=attack_regions,
            attack_color_map=attack_color_map,
            score_column=score_column,
            score_smoothing_column=score_smoothing_column,
        )

        score_ax.set_xlabel("Instâncias", fontsize=13)

        fig.suptitle(
            f"{model_code} - {dataset_name} | {normalization_strategy}",
            fontsize=17,
            fontweight="bold",
            y=0.985,
        )

        handles, labels = self._build_figure_legend(
            attack_regions=attack_regions,
            attack_color_map=attack_color_map,
            score_ax=score_ax,
        )

        legend = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=6,
            fontsize=12,
            frameon=False,
        )

        self.style_legend_patches(legend)

        fig.subplots_adjust(
            top=0.935,
            bottom=0.11,
            left=0.045,
            right=0.99,
        )

        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")

        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        return output_path

    def _plot_single_feature_axis(
        self,
        ax,
        df,
        feature_name,
        attack_regions,
        attack_color_map,
    ):
        x_axis = self._get_x_axis(df)
        values = df[feature_name].astype(float).values

        ax.plot(
            x_axis,
            values,
            color=self.colors[0],
            linewidth=0.85,
            alpha=0.95,
            zorder=3,
        )

        self._add_named_attack_regions(
            ax=ax,
            attack_regions=attack_regions,
            attack_color_map=attack_color_map,
            alpha=0.23,
            show_legend=False,
            show_labels=False,
        )

        self.expand_y_limits(ax, kind="generic")

        ax.set_title(
            self._short_feature_name(feature_name),
            fontsize=8.5,
            fontweight="bold",
        )

        ax.grid(True, alpha=0.25, linestyle=":", zorder=0)
        ax.tick_params(axis="both", which="major", labelsize=7)

    def _plot_score_axis(
        self,
        ax,
        df,
        attack_regions,
        attack_color_map,
        score_column,
        score_smoothing_column,
    ):
        if score_column not in df.columns:
            raise ValueError(f"Coluna de score não encontrada: {score_column}")

        x_axis = self._get_x_axis(df)
        raw_score = df[score_column].astype(float).values

        ax.plot(
            x_axis,
            raw_score,
            color=self.colors[0],
            linewidth=0.85,
            alpha=0.28,
            label="Score bruto",
            zorder=2,
        )

        if score_smoothing_column in df.columns:
            smooth_score = df[score_smoothing_column].astype(float).values
            smooth_label = score_smoothing_column
        else:
            smooth_score = self.moving_average(raw_score, 10)
            smooth_label = "score_ma_10"

        ax.plot(
            x_axis,
            smooth_score,
            color=self.colors[1],
            linewidth=2.0,
            alpha=0.95,
            label=smooth_label,
            zorder=4,
        )

        self._add_named_attack_regions(
            ax=ax,
            attack_regions=attack_regions,
            attack_color_map=attack_color_map,
            alpha=0.24,
            show_legend=False,
            show_labels=True,
        )

        self.expand_y_limits(ax, kind="generic")

        ax.set_title(
            "Score do modelo",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )

        ax.set_ylabel("Score", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle=":", zorder=0)
        ax.tick_params(axis="both", which="major", labelsize=10)

    def _extract_attack_regions_named(
        self,
        df,
        normal_class_idx=0,
        max_gap_between_attacks=1000,
    ):
        if "label_original" not in df.columns:
            raise ValueError("Coluna obrigatória ausente: label_original")

        labels = df["label_original"].astype(int).values
        attack_indices = np.where(labels != int(normal_class_idx))[0]

        attack_regions = []

        if len(attack_indices) == 0:
            return attack_regions

        start_idx = attack_indices[0]
        last_idx = attack_indices[0]

        for idx in attack_indices[1:]:
            if idx - last_idx > max_gap_between_attacks:
                attack_label = self._resolve_attack_region_name(
                    df=df,
                    start_idx=start_idx,
                    end_idx=last_idx,
                    normal_class_idx=normal_class_idx,
                )

                attack_regions.append((start_idx, last_idx, attack_label))
                start_idx = idx

            last_idx = idx

        attack_label = self._resolve_attack_region_name(
            df=df,
            start_idx=start_idx,
            end_idx=last_idx,
            normal_class_idx=normal_class_idx,
        )

        attack_regions.append((start_idx, last_idx, attack_label))

        return attack_regions

    def _resolve_attack_region_name(
        self,
        df,
        start_idx,
        end_idx,
        normal_class_idx,
    ):
        region_df = df.iloc[start_idx:end_idx + 1]

        if "label_name" in region_df.columns:
            attack_names = (
                region_df.loc[
                    region_df["label_original"].astype(int) != int(normal_class_idx),
                    "label_name",
                ]
                .dropna()
                .astype(str)
                .values
            )

            if len(attack_names) > 0:
                values, counts = np.unique(attack_names, return_counts=True)
                return str(values[np.argmax(counts)])

        labels = region_df["label_original"].astype(int).values
        attack_labels = labels[labels != int(normal_class_idx)]

        if len(attack_labels) == 0:
            return "Ataque"

        values, counts = np.unique(attack_labels, return_counts=True)
        dominant_label = int(values[np.argmax(counts)])

        if self.target_names is not None and dominant_label < len(self.target_names):
            return str(self.target_names[dominant_label])

        return f"Ataque {dominant_label}"

    def _build_attack_color_map(self, attack_regions):
        attack_color_map = OrderedDict()

        for _, _, attack_name in attack_regions:
            if attack_name not in attack_color_map:
                color_idx = len(attack_color_map) % len(self.bg_colors)
                attack_color_map[attack_name] = self.bg_colors[color_idx]

        return attack_color_map

    def _add_named_attack_regions(
        self,
        ax,
        attack_regions,
        attack_color_map,
        alpha=0.25,
        show_legend=False,
        show_labels=False,
    ):
        if not attack_regions:
            return

        added_labels = set()

        for start, end, attack_name in attack_regions:
            color = attack_color_map.get(attack_name, self.bg_colors[0])

            label = None

            if show_legend and attack_name not in added_labels:
                label = attack_name
                added_labels.add(attack_name)

            ax.axvspan(
                start,
                end,
                facecolor=color,
                alpha=alpha,
                zorder=1,
                label=label,
            )

            if show_labels:
                mid = (start + end) / 2

                ax.axvline(
                    mid,
                    color=color,
                    alpha=0.90,
                    linewidth=1.2,
                    zorder=2,
                )

                ax.text(
                    mid,
                    0.88,
                    attack_name,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.70,
                        "pad": 0.2,
                    },
                    clip_on=True,
                    zorder=10,
                )

    def _build_figure_legend(
        self,
        attack_regions,
        attack_color_map,
        score_ax,
    ):
        legend_items = OrderedDict()

        for _, _, attack_name in attack_regions:
            if attack_name not in legend_items:
                legend_items[attack_name] = mpatches.Patch(
                    color=attack_color_map.get(attack_name, self.bg_colors[0]),
                    alpha=0.45,
                    label=attack_name,
                )

        handles, labels = score_ax.get_legend_handles_labels()

        for handle, label in zip(handles, labels):
            if label and label not in legend_items:
                legend_items[label] = handle

        return list(legend_items.values()), list(legend_items.keys())

    def _infer_feature_columns(self, df):
        ignored_columns = {
            "exec_id",
            "dataset",
            "instance_id",
            "label_original",
            "label_name",
            "is_attack",
            "model_code",
            "model_name",
            "normalization_strategy",
            "normalizer_update_policy",
            "train_model",
            "score",
        }

        feature_columns = []

        for column in df.columns:
            if column in ignored_columns:
                continue

            if column.startswith("score_ma_"):
                continue

            if pd.api.types.is_numeric_dtype(df[column]):
                feature_columns.append(column)

        return feature_columns

    def _get_x_axis(self, df):
        if "instance_id" in df.columns:
            return df["instance_id"].astype(int).values

        return np.arange(len(df))

    def _short_feature_name(self, feature_name, max_len=24):
        feature_name = str(feature_name)

        if len(feature_name) <= max_len:
            return feature_name

        return feature_name[:max_len - 3] + "..."