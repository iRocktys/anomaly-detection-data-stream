import numpy as np


def shannon_entropy(values):
    counts = (
        values
        .dropna()
        .value_counts()
        .to_numpy(
            dtype=float
        )
    )

    if len(counts) == 0:
        return 0.0

    probabilities = (
        counts
        / counts.sum()
    )

    return float(
        -np.sum(
            probabilities
            * np.log2(
                probabilities
            )
        )
    )


def calculate_entropies(
    started_flows,
):
    columns = {
        "Source_IP":
            "SourceIPEntropy",
        "Destination_IP":
            "DestinationIPEntropy",
        "Source_Port":
            "SourcePortEntropy",
        "Destination_Port":
            "DestinationPortEntropy",
        "Protocol":
            "ProtocolEntropy",
    }

    result = {}

    for source, target in columns.items():
        if source in started_flows.columns:
            result[target] = shannon_entropy(
                started_flows[
                    source
                ]
            )

    return result
