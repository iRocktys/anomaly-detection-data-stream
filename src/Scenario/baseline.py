from pathlib import Path

import pandas as pd

from src.Scenario.data_loader import iter_prepared_chunks
from src.Scenario.windowing import create_windows_chunked


def build_benign_baseline(
    data_dir,
    source_files,
    output_path,
    window,
    attack_ratio_threshold,
    chunk_size=100_000,
):
    # Constrói o baseline com todas as janelas benignas disponíveis nos arquivos informados.
    collected = []

    for file_name in source_files:
        path = (
            Path(data_dir)
            / file_name
        )

        def benign_chunks():
            # Filtra cada lote preparado para manter somente fluxos originalmente benignos.
            for chunk in iter_prepared_chunks(
                path,
                window,
                chunk_size,
            ):
                yield chunk[
                    chunk[
                        "CanonicalLabel"
                    ]
                    == "BENIGN"
                ].copy()

        windows = create_windows_chunked(
            benign_chunks(),
            attack_ratio_threshold,
        )

        if len(windows) == 0:
            continue

        windows[
            "OriginalTimestamp"
        ] = windows[
            "Timestamp"
        ]

        windows[
            "SourceFile"
        ] = file_name

        collected.append(
            windows
        )

    if not collected:
        raise RuntimeError(
            "No BENIGN window was found."
        )

    baseline = pd.concat(
        collected,
        ignore_index=True,
    )

    baseline[
        "Timestamp"
    ] = pd.date_range(
        start="2020-01-01 00:00:00",
        periods=len(
            baseline
        ),
        freq=window,
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline.to_csv(
        output_path,
        index=False,
    )

    return baseline
