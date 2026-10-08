import numpy as np
import pandas as pd

from src.feature_rules import FEATURE_RULES


def sum_values(values):
    return pd.to_numeric(
        values,
        errors="coerce",
    ).sum(min_count=1)


def mean_values(values):
    return pd.to_numeric(
        values,
        errors="coerce",
    ).mean()


def min_values(values):
    return pd.to_numeric(
        values,
        errors="coerce",
    ).min()


def max_values(values):
    return pd.to_numeric(
        values,
        errors="coerce",
    ).max()


def weighted_mean(
    values,
    weights,
):
    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    weights = pd.to_numeric(
        weights,
        errors="coerce",
    )

    valid = (
        values.notna()
        & weights.notna()
        & (weights > 0)
    )

    if not valid.any():
        return np.nan

    return float(
        np.average(
            values[valid],
            weights=weights[valid],
        )
    )


def pooled_std(
    std_values,
    mean_values_series,
    counts,
):
    std_values = pd.to_numeric(
        std_values,
        errors="coerce",
    )

    mean_values_series = pd.to_numeric(
        mean_values_series,
        errors="coerce",
    )

    counts = pd.to_numeric(
        counts,
        errors="coerce",
    )

    valid = (
        std_values.notna()
        & mean_values_series.notna()
        & counts.notna()
        & (counts > 0)
    )

    if not valid.any():
        return np.nan

    s = std_values[valid].to_numpy(dtype=float)
    m = mean_values_series[valid].to_numpy(dtype=float)
    n = counts[valid].to_numpy(dtype=float)

    total_n = n.sum()

    if total_n <= 1:
        return 0.0

    global_mean = (
        np.sum(n * m)
        / total_n
    )

    within = np.sum(
        np.maximum(
            n - 1,
            0,
        )
        * (s ** 2)
    )

    between = np.sum(
        n
        * (
            (m - global_mean)
            ** 2
        )
    )

    variance = (
        within + between
    ) / max(
        total_n - 1,
        1,
    )

    return float(
        np.sqrt(
            max(
                variance,
                0.0,
            )
        )
    )


def add_helper_columns(data):
    data = data.copy()

    fwd = pd.to_numeric(
        data.get(
            "Total_Fwd_Packets",
            pd.Series(
                0,
                index=data.index,
            ),
        ),
        errors="coerce",
    ).fillna(0)

    bwd = pd.to_numeric(
        data.get(
            "Total_Backward_Packets",
            pd.Series(
                0,
                index=data.index,
            ),
        ),
        errors="coerce",
    ).fillna(0)

    data["__TotalPackets"] = (
        fwd + bwd
    )

    data["__FlowIATCount"] = np.maximum(
        data["__TotalPackets"] - 1,
        0,
    )

    data["__FwdIATCount"] = np.maximum(
        fwd - 1,
        0,
    )

    data["__BwdIATCount"] = np.maximum(
        bwd - 1,
        0,
    )

    return data


def aggregate_window(
    started_flows,
):
    data = add_helper_columns(
        started_flows
    )

    output = {}

    for feature, rule in FEATURE_RULES.items():
        if feature not in data.columns:
            continue

        method = rule["method"]

        if method == "sum":
            output[feature] = sum_values(
                data[feature]
            )

        elif method == "mean":
            output[feature] = mean_values(
                data[feature]
            )

        elif method == "min":
            output[feature] = min_values(
                data[feature]
            )

        elif method == "max":
            output[feature] = max_values(
                data[feature]
            )

        elif method == "weighted_mean":
            weight = rule["weight"]

            if weight in data.columns:
                output[feature] = weighted_mean(
                    data[feature],
                    data[weight],
                )

        elif method == "pooled_std":
            count_column = rule["count"]
            mean_column = rule["mean"]

            if (
                count_column in data.columns
                and mean_column in data.columns
            ):
                output[feature] = pooled_std(
                    data[feature],
                    data[mean_column],
                    data[count_column],
                )

        elif method == "variance":
            std_feature = rule["std"]

            if std_feature in output:
                output[feature] = (
                    output[std_feature]
                    ** 2
                )

    return output


def rules_table(data):
    rows = []

    for feature, rule in FEATURE_RULES.items():
        rows.append({
            "Feature": feature,
            "ExistsInCSV": (
                feature in data.columns
            ),
            "Method": rule["method"],
            "Weight": rule.get(
                "weight",
                "",
            ),
            "Count": rule.get(
                "count",
                "",
            ),
            "Mean": rule.get(
                "mean",
                "",
            ),
            "Description": rule[
                "description"
            ],
        })

    return pd.DataFrame(rows)
