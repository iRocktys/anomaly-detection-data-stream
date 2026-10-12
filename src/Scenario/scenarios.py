from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.attack_blocks import prepare_attack_segments


def split_baseline(
    baseline,
    parts,
):
    indices = np.array_split(
        np.arange(
            len(baseline)
        ),
        parts,
    )

    return [
        baseline.iloc[
            index
        ].copy()
        for index in indices
    ]


def required_attack_segments(
    scenarios,
):
    required = Counter()

    for sequence in scenarios.values():
        counts = Counter(
            sequence
        )

        for attack, amount in counts.items():
            required[attack] = max(
                required[attack],
                amount,
            )

    return required


def prepare_scenario_attack_segments(
    attack_blocks,
    scenarios,
    preferred_duration,
):
    required = required_attack_segments(
        scenarios
    )

    prepared = {}
    summary = []

    for attack, count in required.items():
        segments, metadata = prepare_attack_segments(
            attack_block=attack_blocks[
                attack
            ],
            attack_name=attack,
            required_count=count,
            preferred_duration=preferred_duration,
        )

        prepared[
            attack
        ] = segments

        summary.append(
            metadata
        )

    return (
        prepared,
        pd.DataFrame(
            summary
        ),
    )


def build_scenario(
    scenario_name,
    sequence,
    baseline,
    attack_segments,
    output_path,
    window,
):
    benign_parts = split_baseline(
        baseline,
        len(sequence) + 1,
    )

    attack_use = Counter()
    pieces = []

    for index, benign in enumerate(
        benign_parts
    ):
        benign = benign.copy()

        benign[
            "ScenarioPart"
        ] = "BENIGN"

        pieces.append(
            benign
        )

        if index < len(sequence):
            attack = sequence[
                index
            ]

            piece = attack_segments[
                attack
            ][
                attack_use[
                    attack
                ]
            ].copy()

            attack_use[
                attack
            ] += 1

            piece[
                "ScenarioPart"
            ] = attack

            pieces.append(
                piece
            )

    scenario = pd.concat(
        pieces,
        ignore_index=True,
    )

    # Nova linha temporal contínua do cenário.
    scenario[
        "Timestamp"
    ] = pd.date_range(
        start="2021-01-01 00:00:00",
        periods=len(
            scenario
        ),
        freq=window,
    )

    metadata_columns = [
        "OriginalTimestamp",
        "SourceFile",
        "AttackBlock",
        "ScenarioPart",
    ]

    final_columns = [
        column
        for column in scenario.columns
        if column not in metadata_columns
    ]

    final = scenario[
        final_columns
    ].copy()

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        output_path,
        index=False,
    )

    manifest_columns = [
        column
        for column in [
            "Timestamp",
            "ScenarioPart",
            "OriginalTimestamp",
            "SourceFile",
            "AttackBlock",
        ]
        if column in scenario.columns
    ]

    manifest = scenario[
        manifest_columns
    ].copy()

    manifest.to_csv(
        output_path.with_name(
            output_path.stem
            + "_manifest.csv"
        ),
        index=False,
    )

    return final
