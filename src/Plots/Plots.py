from src.Plots.ErrorPlot import ErrorPlot
from src.Plots.MetricsPlot import MetricsPlot
from src.Plots.NormalizerBehaviorPlot import NormalizerBehaviorPlot
from src.Plots.ScorePlot import ScorePlot


class Plots:
    def __init__(self, target_names):
        self.target_names = target_names

        self.metrics_plot = MetricsPlot(target_names)
        self.error_plot = ErrorPlot(target_names)
        self.score_plot = ScorePlot(target_names)
        self.normalizer_behavior_plot = NormalizerBehaviorPlot(target_names)

    def plot_metrics(
        self,
        results,
        attack_regions=None,
        title="Métricas",
        window_size=1000,
        target_class=None,
        scenario_name="General",
        discretization_strategy="fixed",
    ):
        return self.metrics_plot.plot(
            results=results,
            attack_regions=attack_regions,
            title=title,
            window_size=window_size,
            target_class=target_class,
            scenario_name=scenario_name,
            discretization_strategy=discretization_strategy,
        )

    def plot_fp_fn(
        self,
        results,
        attack_regions=None,
        title="Contagem de FP e FN",
        window_size=1000,
        scenario_name="General",
        discretization_strategy="fixed",
    ):
        return self.error_plot.plot(
            results=results,
            attack_regions=attack_regions,
            title=title,
            window_size=window_size,
            scenario_name=scenario_name,
            discretization_strategy=discretization_strategy,
        )

    def plot_score(
        self,
        results,
        attack_regions,
        title="Análise de Scores",
        discretization=0.5,
        scenario_name="General",
        discretization_strategy="fixed",
    ):
        return self.score_plot.plot(
            results=results,
            attack_regions=attack_regions,
            title=title,
            discretization=discretization,
            scenario_name=scenario_name,
            discretization_strategy=discretization_strategy,
        )

    def plot_normalizer_behavior(
        self,
        files_by_model,
        output_dir="output/ExpNormalizers/Plots",
        score_column="score",
        score_smoothing_column="score_ma_10",
        normal_class_idx=0,
        feature_columns=None,
    ):
        return self.normalizer_behavior_plot.plot_from_files(
            files_by_model=files_by_model,
            output_dir=output_dir,
            score_column=score_column,
            score_smoothing_column=score_smoothing_column,
            normal_class_idx=normal_class_idx,
            feature_columns=feature_columns,
        )

    def plot_normalizer_behavior_file(
        self,
        csv_path,
        output_dir="output/ExpNormalizers/Plots",
        score_column="score",
        score_smoothing_column="score_ma_10",
        normal_class_idx=0,
        feature_columns=None,
    ):
        return self.normalizer_behavior_plot.plot_single_file(
            csv_path=csv_path,
            output_dir=output_dir,
            score_column=score_column,
            score_smoothing_column=score_smoothing_column,
            normal_class_idx=normal_class_idx,
            feature_columns=feature_columns,
        )