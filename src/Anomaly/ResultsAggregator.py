import numpy as np


class ResultsAggregator:
    def __init__(self, metrics, n_runs):
        self.metrics = metrics
        self.n_runs = n_runs

    def aggregate_algorithm_results(
        self,
        runs_data,
        exec_times,
        warmup_instances,
    ):
        f1_matrix = self._build_matrix(runs_data, "f1")
        precision_matrix = self._build_matrix(runs_data, "precision")
        recall_matrix = self._build_matrix(runs_data, "recall")
        fp_matrix = self._build_matrix(runs_data, "fp")
        fn_matrix = self._build_matrix(runs_data, "fn")
        scores_matrix = self._build_matrix(runs_data, "scores")

        cumulative_matrix = self._build_cumulative_matrix(
            runs_data=runs_data,
            warmup_instances=warmup_instances,
        )

        return {
            "instances": runs_data[0]["instances"],

            "f1_mean": self._mean_by_window(f1_matrix),
            "f1_std": self._std_by_window(f1_matrix),

            "precision_mean": self._mean_by_window(precision_matrix),
            "precision_std": self._std_by_window(precision_matrix),

            "recall_mean": self._mean_by_window(recall_matrix),
            "recall_std": self._std_by_window(recall_matrix),

            "fp_mean": self._mean_by_window(fp_matrix),
            "fp_std": self._std_by_window(fp_matrix),

            "fn_mean": self._mean_by_window(fn_matrix),
            "fn_std": self._std_by_window(fn_matrix),

            "scores_mean": self._mean_by_window(scores_matrix),
            "scores_std": self._std_by_window(scores_matrix),

            "exec_time_mean": np.mean(exec_times),
            "exec_time_std": np.std(exec_times) if self.n_runs > 1 else 0.0,

            "cumulative": self._aggregate_cumulative_metrics(cumulative_matrix),

            "true_labels_multi": runs_data[0]["true_labels_multi"],
        }

    def _build_matrix(self, runs_data, key):
        return np.array([run_result[key] for run_result in runs_data])

    def _mean_by_window(self, matrix):
        if len(matrix) == 0:
            return []

        if len(matrix[0]) == 0:
            return []

        return np.mean(matrix, axis=0)

    def _std_by_window(self, matrix):
        if len(matrix) == 0:
            return []

        if len(matrix[0]) == 0:
            return []

        if self.n_runs <= 1:
            return np.zeros_like(matrix[0])

        return np.std(matrix, axis=0)

    def _build_cumulative_matrix(
        self,
        runs_data,
        warmup_instances,
    ):
        cumulative_metrics = []

        for run_result in runs_data:
            y_true = self._remove_warmup(
                values=run_result["y_true"],
                warmup_instances=warmup_instances,
            )

            y_pred = self._remove_warmup(
                values=run_result["y_pred"],
                warmup_instances=warmup_instances,
            )

            cumulative_metrics.append(
                self.metrics.calc_sklearn_metrics(
                    y_true,
                    y_pred,
                )
            )

        return np.array(cumulative_metrics)

    def _remove_warmup(self, values, warmup_instances):
        values = np.array(values)

        if len(values) > warmup_instances:
            return values[warmup_instances:]

        return values

    def _aggregate_cumulative_metrics(self, cumulative_matrix):
        return {
            "f1": self._mean_std_tuple(cumulative_matrix, 0),
            "prec": self._mean_std_tuple(cumulative_matrix, 1),
            "rec": self._mean_std_tuple(cumulative_matrix, 2),
            "mcc": self._mean_std_tuple(cumulative_matrix, 3),
            "fp": self._mean_std_tuple(cumulative_matrix, 4),
            "fn": self._mean_std_tuple(cumulative_matrix, 5),
        }

    def _mean_std_tuple(self, matrix, column_idx):
        mean_value = np.mean(matrix[:, column_idx])
        std_value = np.std(matrix[:, column_idx]) if self.n_runs > 1 else 0.0

        return mean_value, std_value