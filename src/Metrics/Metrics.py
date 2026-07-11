import csv
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

warnings.filterwarnings(
    "ignore",
    message="A single label was found in 'y_true' and 'y_pred'.*",
)


class Metrics:
    COMMON_PARAM_NAMES = {
        "threshold",
        "anomaly_threshold",
        "dynamic_threshold",
        "train_only_normal_samples",
        "normalization_strategy",
        "normalizer_update_policy",
    }

    AUXILIARY_PARAM_NAMES = {
        "z",
        "u",
        "std",
        "mu",
        "warmup_mean",
        "warmup_std",
        "calculated_threshold",
    }

    MODEL_PARAM_MAP = {
        "HalfSpaceTrees": [
            "window_size",
            "number_of_trees",
            "max_depth",
            "size_limit",
            "anomaly_threshold",
            "threshold",
        ],
        "OnlineIsolationForest": [
            "num_trees",
            "max_leaf_samples",
            "growth_criterion",
            "subsample",
            "window_size",
            "branching_factor",
            "split",
            "n_jobs",
            "threshold",
        ],
        "Autoencoder": [
            "hidden_layer",
            "learning_rate",
            "threshold",
        ],
        "RobustRandomCutForest": [
            "tree_size",
            "n_trees",
            "random_state",
            "threshold",
        ],
        "AdaptiveIsolationForest": [
            "window_size",
            "n_trees",
            "height",
            "seed",
            "m_trees",
            "weights",
            "threshold",
        ],
    }

    MODEL_ALIASES = {
        "HST": "HalfSpaceTrees",
        "OIF": "OnlineIsolationForest",
        "AE": "Autoencoder",
        "RRCF": "RobustRandomCutForest",
        "AIF": "AdaptiveIsolationForest",
    }

    def __init__(self):
        self.all_possible_params = sorted(
            {
                param
                for params in self.MODEL_PARAM_MAP.values()
                for param in params
            }
            | self.COMMON_PARAM_NAMES
            | self.AUXILIARY_PARAM_NAMES
        )

    def calc_sklearn_metrics(self, y_true, y_pred, target_class=None):
        if len(y_true) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        f1 = f1_score(
            y_true,
            y_pred,
            pos_label=1,
            average="binary",
            zero_division=0,
        )

        precision = precision_score(
            y_true,
            y_pred,
            pos_label=1,
            average="binary",
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            pos_label=1,
            average="binary",
            zero_division=0,
        )

        mcc = matthews_corrcoef(y_true, y_pred)

        y_true_array = np.asarray(y_true)
        y_pred_array = np.asarray(y_pred)

        false_positive = float(
            np.sum((y_pred_array == 1) & (y_true_array == 0))
        )

        false_negative = float(
            np.sum((y_pred_array == 0) & (y_true_array == 1))
        )

        return (
            f1 * 100.0,
            precision * 100.0,
            recall * 100.0,
            mcc,
            false_positive,
            false_negative,
        )

    def display_cumulative_metrics(
        self,
        predictions_history,
        warmup_instances=0,
        target_class=None,
        n_runs=1,
        params_dict=None,
        experiment_name="General",
        scenario_name="Default_Full",
        discretization=None,
        window_evaluation=None,
        exec_id="N/A",
        discretization_strategy="fixed",
        task_type="anomaly",
        save_prequential=True,
        threshold_strategy=None,
        decision_strategy=None,
        decision_window=None,
        persistence_k=None,
        persistence_n=None,
        verbose=False,
    ):
        discretization_str = (
            self._format_value(discretization, decimal_comma=False)
            if discretization is not None
            else "N/A"
        )

        cumulative_paths = []
        prequential_paths = []

        if verbose:
            self._print_metrics_header(
                scenario_name=scenario_name,
                window_evaluation=window_evaluation,
            )

        for model_name, data in predictions_history.items():
            if verbose:
                self._print_model_metrics(
                    model_name=model_name,
                    data=data,
                    task_type=task_type,
                    strategy_name=discretization_strategy,
                    window_evaluation=window_evaluation,
                )

            cumulative_path = self.save_cumulative_metrics_csv(
                model_name=model_name,
                data=data,
                params_dict=params_dict,
                experiment_name=experiment_name,
                scenario_name=scenario_name,
                discretization=discretization_str,
                window_evaluation=window_evaluation,
                exec_id=exec_id,
                warmup_instances=warmup_instances,
                strategy_name=discretization_strategy,
                task_type=task_type,
                threshold_strategy=threshold_strategy,
                decision_strategy=decision_strategy,
                decision_window=decision_window,
                persistence_k=persistence_k,
                persistence_n=persistence_n,
            )

            cumulative_paths.append(cumulative_path)

            if save_prequential:
                prequential_path = self.save_prequential_metrics_csv(
                    model_name=model_name,
                    data=data,
                    experiment_name=experiment_name,
                    scenario_name=scenario_name,
                    window_evaluation=window_evaluation,
                    exec_id=exec_id,
                    warmup_instances=warmup_instances,
                    strategy_name=discretization_strategy,
                    task_type=task_type,
                    threshold_strategy=threshold_strategy,
                    decision_strategy=decision_strategy,
                    decision_window=decision_window,
                    persistence_k=persistence_k,
                    persistence_n=persistence_n,
                )

                if prequential_path:
                    prequential_paths.append(prequential_path)

        if verbose:
            self._print_output_paths(
                cumulative_paths=cumulative_paths,
                prequential_paths=prequential_paths,
            )

    def save_cumulative_metrics_csv(
        self,
        model_name: str,
        data: Dict[str, Any],
        params_dict: Optional[Dict[str, Any]],
        experiment_name: str,
        scenario_name: str,
        discretization: Any,
        window_evaluation: Any,
        exec_id: Any,
        warmup_instances: int,
        strategy_name: str,
        task_type: str,
        threshold_strategy: Optional[str] = None,
        decision_strategy: Optional[str] = None,
        decision_window: Any = None,
        persistence_k: Any = None,
        persistence_n: Any = None,
    ) -> str:
        category, contamination_block = self._split_experiment_name(experiment_name)

        clean_params = self._clean_params(params_dict)
        model_params = self._filter_model_params(
            model_name=model_name,
            params_dict=clean_params,
        )

        z_value, warmup_mean, warmup_std, calculated_threshold = (
            self._extract_threshold_metadata(
                params_dict=clean_params,
                discretization=discretization,
            )
        )

        output_dir = self._resolve_output_dir(
            model_name=model_name,
            strategy_name=strategy_name,
        )

        csv_file_path = os.path.join(
            output_dir,
            f"{self._normalize_model_name(model_name)}_{scenario_name}.csv",
        )

        headers = [
            "Exec_ID",
            "Dataset",
            "Category",
            "Contamination_Block",
            "Task_Type",
            "Model",
            "Scenario",
            "Strategy",
            "Threshold_Strategy",
            "Decision_Strategy",
            "Decision_Window",
            "Persistence_K",
            "Persistence_N",
            "Warmup",
            "Win_Evaluation",
            "Discretization",
            "Z",
            "Score_Mean",
            "Score_Std",
            "Calculated_Threshold",
            "F1_avg",
            "F1_std",
            "Prec_avg",
            "Prec_std",
            "Rec_avg",
            "Rec_std",
            "MCC_avg",
            "MCC_std",
            "FP_avg",
            "FP_std",
            "FN_avg",
            "FN_std",
            "Time_avg",
            "Time_std",
        ]

        param_headers = [f"param_{name}" for name in model_params.keys()]
        headers = headers + param_headers

        cumulative = data.get("cumulative")

        if cumulative:
            f1_m, f1_s = cumulative.get("f1", (0.0, 0.0))
            prec_m, prec_s = cumulative.get("prec", (0.0, 0.0))
            rec_m, rec_s = cumulative.get("rec", (0.0, 0.0))
            mcc_m, mcc_s = cumulative.get("mcc", (0.0, 0.0))
            fp_m, fp_s = cumulative.get("fp", (0.0, 0.0))
            fn_m, fn_s = cumulative.get("fn", (0.0, 0.0))
            time_m = data.get("exec_time_mean", data.get("exec_time", 0.0))
            time_s = data.get("exec_time_std", 0.0)
        else:
            f1_m, prec_m, rec_m, mcc_m, fp_m, fn_m = self.calc_sklearn_metrics(
                data.get("y_true", []),
                data.get("y_pred", []),
            )

            f1_s = prec_s = rec_s = mcc_s = fp_s = fn_s = 0.0
            time_m = data.get("exec_time", 0.0)
            time_s = 0.0

        row = [
            exec_id,
            experiment_name,
            category,
            contamination_block,
            task_type,
            self._normalize_model_name(model_name),
            scenario_name,
            strategy_name,
            threshold_strategy if threshold_strategy is not None else strategy_name,
            decision_strategy if decision_strategy is not None else "raw",
            decision_window if decision_window is not None else "N/A",
            persistence_k if persistence_k is not None else "N/A",
            persistence_n if persistence_n is not None else "N/A",
            warmup_instances,
            window_evaluation if window_evaluation is not None else "N/A",
            discretization if discretization is not None else "N/A",
            z_value,
            warmup_mean,
            warmup_std,
            calculated_threshold if calculated_threshold is not None else "N/A",
            f1_m,
            f1_s,
            prec_m,
            prec_s,
            rec_m,
            rec_s,
            mcc_m,
            mcc_s,
            self._safe_ceil_int(fp_m),
            self._safe_ceil_int(fp_s),
            self._safe_ceil_int(fn_m),
            self._safe_ceil_int(fn_s),
            time_m,
            time_s,
        ] + list(model_params.values())

        self._write_csv_row(
            csv_file_path=csv_file_path,
            headers=headers,
            row=row,
        )

        return csv_file_path

    def save_prequential_metrics_csv(
        self,
        model_name: str,
        data: Dict[str, Any],
        experiment_name: str,
        scenario_name: str,
        window_evaluation: Any,
        exec_id: Any,
        warmup_instances: int,
        strategy_name: str,
        task_type: str,
        threshold_strategy: Optional[str] = None,
        decision_strategy: Optional[str] = None,
        decision_window: Any = None,
        persistence_k: Any = None,
        persistence_n: Any = None,
    ) -> Optional[str]:
        if window_evaluation is None:
            return None

        instances = list(data.get("instances", []))

        if not instances:
            return None

        output_dir = self._resolve_output_dir(
            model_name=model_name,
            strategy_name=strategy_name,
        )

        prequential_dir = os.path.join(output_dir, "prequential")
        os.makedirs(prequential_dir, exist_ok=True)

        normalized_model_name = self._normalize_model_name(model_name)

        csv_file_path = os.path.join(
            prequential_dir,
            f"{normalized_model_name}_{scenario_name}_window_{window_evaluation}.csv",
        )

        headers = [
            "Exec_ID",
            "Dataset",
            "Task_Type",
            "Model",
            "Scenario",
            "Strategy",
            "Threshold_Strategy",
            "Decision_Strategy",
            "Decision_Window",
            "Persistence_K",
            "Persistence_N",
            "Warmup",
            "Window_Size",
            "Window_Index",
            "Instance",
            "F1_avg",
            "F1_std",
            "Prec_avg",
            "Prec_std",
            "Rec_avg",
            "Rec_std",
            "FP_avg",
            "FP_std",
            "FN_avg",
            "FN_std",
        ]

        rows = []

        for idx, instance in enumerate(instances):
            f1_m, f1_s = self._get_metric_pair(data, "f1_mean", "f1_std", idx)
            prec_m, prec_s = self._get_metric_pair(data, "precision_mean", "precision_std", idx)
            rec_m, rec_s = self._get_metric_pair(data, "recall_mean", "recall_std", idx)
            fp_m, fp_s = self._get_metric_pair(data, "fp_mean", "fp_std", idx)
            fn_m, fn_s = self._get_metric_pair(data, "fn_mean", "fn_std", idx)

            rows.append(
                [
                    exec_id,
                    experiment_name,
                    task_type,
                    normalized_model_name,
                    scenario_name,
                    strategy_name,
                    threshold_strategy if threshold_strategy is not None else strategy_name,
                    decision_strategy if decision_strategy is not None else "raw",
                    decision_window if decision_window is not None else "N/A",
                    persistence_k if persistence_k is not None else "N/A",
                    persistence_n if persistence_n is not None else "N/A",
                    warmup_instances,
                    window_evaluation,
                    idx + 1,
                    instance,
                    f1_m,
                    f1_s,
                    prec_m,
                    prec_s,
                    rec_m,
                    rec_s,
                    self._safe_ceil_int(fp_m),
                    self._safe_ceil_int(fp_s),
                    self._safe_ceil_int(fn_m),
                    self._safe_ceil_int(fn_s),
                ]
            )

        self._write_csv_rows(
            csv_file_path=csv_file_path,
            headers=headers,
            rows=rows,
        )

        return csv_file_path

    def _normalize_model_name(self, model_name: str) -> str:
        return self.MODEL_ALIASES.get(str(model_name), str(model_name))

    def _is_simple_value(self, value: Any) -> bool:
        return value is None or isinstance(
            value,
            (
                int,
                float,
                str,
                bool,
                np.integer,
                np.floating,
                np.bool_,
            ),
        )

    def _format_value(self, value: Any, decimal_comma: bool = True) -> str:
        if value is None:
            text = "N/A"
        elif isinstance(value, np.integer):
            text = str(int(value))
        elif isinstance(value, (np.floating, float)):
            if np.isnan(value):
                text = "N/A"
            else:
                text = f"{float(value):.4f}"
        elif isinstance(value, bool):
            text = str(value)
        else:
            text = str(value)

        return text.replace(".", ",") if decimal_comma else text

    def _safe_ceil_int(self, value: Any) -> int:
        try:
            if value is None or np.isnan(value):
                return 0
        except TypeError:
            pass

        return int(np.ceil(float(value)))

    def _split_experiment_name(self, experiment_name: str) -> Tuple[str, str]:
        parts = str(experiment_name).split("_")

        if len(parts) >= 2:
            return parts[0], parts[1]

        return str(experiment_name), "N/A"

    def _resolve_output_dir(self, model_name: str, strategy_name: str) -> str:
        output_dir = os.path.join(
            "output",
            self._normalize_model_name(model_name),
            str(strategy_name),
        )

        os.makedirs(output_dir, exist_ok=True)

        return output_dir

    def _clean_params(self, params_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not params_dict:
            return {}

        return {
            str(key): value
            for key, value in params_dict.items()
            if self._is_simple_value(value)
        }

    def _extract_threshold_metadata(
        self,
        params_dict: Dict[str, Any],
        discretization: Any,
    ) -> Tuple[Any, Any, Any, Any]:
        z_value = params_dict.get("z", "N/A")

        warmup_mean = params_dict.get(
            "u",
            params_dict.get(
                "mu",
                params_dict.get("warmup_mean", "N/A"),
            ),
        )

        warmup_std = params_dict.get(
            "std",
            params_dict.get("warmup_std", "N/A"),
        )

        calculated_threshold = params_dict.get(
            "calculated_threshold",
            discretization,
        )

        return z_value, warmup_mean, warmup_std, calculated_threshold

    def _filter_model_params(
        self,
        model_name: str,
        params_dict: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        clean_params = self._clean_params(params_dict)

        if not clean_params:
            return {}

        normalized_model_name = self._normalize_model_name(model_name)
        allowed_params = self.MODEL_PARAM_MAP.get(normalized_model_name)

        excluded_params = set(self.AUXILIARY_PARAM_NAMES)

        if allowed_params is None:
            return {
                key: value
                for key, value in clean_params.items()
                if key not in excluded_params
            }

        allowed_set = set(allowed_params) | self.COMMON_PARAM_NAMES

        return {
            key: value
            for key, value in clean_params.items()
            if key in allowed_set and key not in excluded_params
        }

    def _write_csv_row(
        self,
        csv_file_path: str,
        headers: List[str],
        row: List[Any],
    ) -> None:
        file_exists = (
            os.path.isfile(csv_file_path)
            and os.path.getsize(csv_file_path) > 0
        )

        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        with open(csv_file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")

            if not file_exists:
                writer.writerow(headers)

            writer.writerow([self._format_value(value) for value in row])

    def _write_csv_rows(
        self,
        csv_file_path: str,
        headers: List[str],
        rows: Iterable[List[Any]],
    ) -> None:
        rows = list(rows)

        if not rows:
            return

        file_exists = (
            os.path.isfile(csv_file_path)
            and os.path.getsize(csv_file_path) > 0
        )

        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        with open(csv_file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")

            if not file_exists:
                writer.writerow(headers)

            for row in rows:
                writer.writerow([self._format_value(value) for value in row])

    def _get_metric_pair(
        self,
        data: Dict[str, Any],
        key_mean: str,
        key_std: str,
        idx: int,
    ) -> Tuple[Any, Any]:
        mean_values = data.get(key_mean, [])
        std_values = data.get(key_std, [])

        mean_value = mean_values[idx] if idx < len(mean_values) else 0.0
        std_value = std_values[idx] if idx < len(std_values) else 0.0

        return mean_value, std_value

    def _print_metrics_header(
        self,
        scenario_name,
        window_evaluation,
    ):
        title = f"ANOMALY METRICS | {scenario_name.upper()}"
        window_text = str(window_evaluation) if window_evaluation is not None else "N/A"

        print("\n" + "=" * 110)
        print(title)
        print("=" * 110)
        print(
            f"{'Algorithm':<24} | {'F1 (%)':<17} | {'Prec (%)':<17} | "
            f"{'Rec (%)':<17} | {'MCC':<17} | {'FP':<8} | "
            f"{'FN':<8} | {'Time (s)':<11} | {'Win':<8}"
        )
        print("-" * 110)
        print(f"Window evaluation: {window_text}")

    def _print_model_metrics(
        self,
        model_name,
        data,
        task_type,
        strategy_name,
        window_evaluation,
    ):
        if "cumulative" in data:
            f1_m, f1_s = data["cumulative"].get("f1", (0.0, 0.0))
            prec_m, prec_s = data["cumulative"].get("prec", (0.0, 0.0))
            rec_m, rec_s = data["cumulative"].get("rec", (0.0, 0.0))
            mcc_m, mcc_s = data["cumulative"].get("mcc", (0.0, 0.0))
            fp_m, _ = data["cumulative"].get("fp", (0.0, 0.0))
            fn_m, _ = data["cumulative"].get("fn", (0.0, 0.0))
            time_m = data.get("exec_time_mean", data.get("exec_time", 0.0))

            print(
                f"{model_name:<24} | "
                f"{f1_m:>7.4f} ± {f1_s:<6.4f} | "
                f"{prec_m:>7.4f} ± {prec_s:<6.4f} | "
                f"{rec_m:>7.4f} ± {rec_s:<6.4f} | "
                f"{mcc_m:>7.4f} ± {mcc_s:<6.4f} | "
                f"{self._safe_ceil_int(fp_m):>4} | "
                f"{self._safe_ceil_int(fn_m):>4} | "
                f"{float(time_m):>9.4f} | "
                f"{str(window_evaluation):<8}"
            )

    def _print_output_paths(
        self,
        cumulative_paths,
        prequential_paths,
    ):
        print("-" * 110)

        if cumulative_paths:
            print("Cumulative CSV:")
            for path in cumulative_paths:
                print(f" - {path}")

        if prequential_paths:
            print("Prequential CSV:")
            for path in prequential_paths:
                print(f" - {path}")

        print()