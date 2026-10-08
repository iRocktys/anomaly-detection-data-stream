import pandas as pd

from src.aggregation import aggregate_window
from src.entropy import calculate_entropies


def attack_ratio(
    started_flows,
):
    attack_count = (
        started_flows[
            "CanonicalLabel"
        ]
        .ne("BENIGN")
        .sum()
    )

    return float(
        attack_count
        / len(started_flows)
    )


def dominant_attack(
    started_flows,
):
    attacks = (
        started_flows.loc[
            started_flows[
                "CanonicalLabel"
            ]
            != "BENIGN",
            "CanonicalLabel",
        ]
        .value_counts()
    )

    if len(attacks) == 0:
        return None

    return attacks.index[0]


def aggregate_started_flows(
    started_flows,
    window_start,
    attack_ratio_threshold,
):
    ratio = attack_ratio(
        started_flows
    )

    attack_name = dominant_attack(
        started_flows
    )

    label = "BENIGN"

    if (
        attack_name is not None
        and ratio
        >= attack_ratio_threshold
    ):
        label = attack_name

    row = {
        "Timestamp": window_start,
        "Label": label,
        "AttackRatio": ratio,
    }

    # Entropias calculadas usando IPs, portas e protocolo,
    # mas os valores originais dessas colunas NÃO são exportados.
    row.update(
        calculate_entropies(
            started_flows
        )
    )

    row.update(
        aggregate_window(
            started_flows
        )
    )

    return row


def create_windows(
    flows,
    attack_ratio_threshold,
):
    rows = []

    for window_start, started_flows in (
        flows.groupby(
            "WindowStart",
            sort=True,
        )
    ):
        rows.append(
            aggregate_started_flows(
                started_flows,
                window_start,
                attack_ratio_threshold,
            )
        )

    return pd.DataFrame(rows)


def create_windows_chunked(
    prepared_chunks,
    attack_ratio_threshold,
):
    rows = []
    carry = None

    for chunk in prepared_chunks:
        if len(chunk) == 0:
            continue

        if carry is not None:
            chunk = pd.concat(
                [
                    carry,
                    chunk,
                ],
                ignore_index=True,
            )

        chunk = (
            chunk
            .sort_values(
                "FlowStart"
            )
            .reset_index(
                drop=True
            )
        )

        last_window = (
            chunk[
                "WindowStart"
            ]
            .max()
        )

        complete = chunk[
            chunk[
                "WindowStart"
            ]
            != last_window
        ]

        carry = chunk[
            chunk[
                "WindowStart"
            ]
            == last_window
        ].copy()

        if len(complete):
            rows.append(
                create_windows(
                    complete,
                    attack_ratio_threshold,
                )
            )

    if carry is not None and len(carry):
        rows.append(
            create_windows(
                carry,
                attack_ratio_threshold,
            )
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.concat(
            rows,
            ignore_index=True,
        )
        .sort_values(
            "Timestamp"
        )
        .reset_index(
            drop=True
        )
    )
