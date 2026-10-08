import re

import numpy as np
import pandas as pd


def clean_column_name(name):
    name = str(name).strip()
    name = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        name,
    )
    return name.strip("_")


def canonical_label(series):
    labels = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )

    for prefix in [
        "DRDOS_",
        "DRDOS-",
        "DDOS_",
        "DDOS-",
    ]:
        labels = labels.str.replace(
            prefix,
            "",
            regex=False,
        )

    labels = labels.str.replace(
        "UDP_LAG",
        "UDP-LAG",
        regex=False,
    )

    return labels


def normalize_columns(data):
    data = data.copy()

    data.columns = [
        clean_column_name(column)
        for column in data.columns
    ]

    return data


def prepare_flows(
    data,
    window,
):
    data = normalize_columns(
        data
    )

    data["FlowStart"] = pd.to_datetime(
        data["Timestamp"],
        errors="coerce",
    )

    data = (
        data
        .dropna(
            subset=["FlowStart"]
        )
        .copy()
    )

    duration_us = pd.to_numeric(
        data["Flow_Duration"],
        errors="coerce",
    ).fillna(0).clip(lower=0)

    duration_us = np.maximum(
        duration_us.to_numpy(
            dtype=float
        ),
        1,
    )

    data["FlowEnd"] = (
        data["FlowStart"]
        + pd.to_timedelta(
            duration_us,
            unit="us",
        )
    )

    data["WindowStart"] = (
        data["FlowStart"]
        .dt.floor(window)
    )

    data["CanonicalLabel"] = (
        canonical_label(
            data["Label"]
        )
    )

    return data


def load_sample(
    csv_path,
    start_row,
    sample_rows,
    window,
):
    skip_rows = (
        range(
            1,
            start_row + 1,
        )
        if start_row > 0
        else None
    )

    data = pd.read_csv(
        csv_path,
        skiprows=skip_rows,
        nrows=sample_rows,
        low_memory=False,
    )

    data = prepare_flows(
        data,
        window,
    )

    return (
        data
        .sort_values(
            "FlowStart"
        )
        .reset_index(
            drop=True
        )
    )


def iter_prepared_chunks(
    csv_path,
    window,
    chunk_size=100_000,
):
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        low_memory=False,
    ):
        yield prepare_flows(
            chunk,
            window,
        )
