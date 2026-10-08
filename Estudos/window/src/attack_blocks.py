from pathlib import Path

import pandas as pd

from src.data_loader import iter_prepared_chunks
from src.windowing import create_windows_chunked


def find_attack_interval(
    csv_path,
    window,
    chunk_size=100_000,
):
    first_attack = None
    last_attack = None

    for chunk in iter_prepared_chunks(
        csv_path,
        window,
        chunk_size,
    ):
        attacks = chunk[
            chunk["CanonicalLabel"] != "BENIGN"
        ]

        if len(attacks) == 0:
            continue

        current_first = attacks["FlowStart"].min()
        current_last = attacks["FlowStart"].max()

        first_attack = (
            current_first
            if first_attack is None
            else min(
                first_attack,
                current_first,
            )
        )

        last_attack = (
            current_last
            if last_attack is None
            else max(
                last_attack,
                current_last,
            )
        )

    if first_attack is None:
        raise RuntimeError(
            f"Nenhum ataque encontrado em {csv_path}."
        )

    return (
        first_attack,
        last_attack,
    )


def build_attack_block(
    csv_path,
    attack_name,
    output_path,
    window,
    attack_ratio_threshold,
    chunk_size=100_000,
):
    first_attack, last_attack = find_attack_interval(
        csv_path,
        window,
        chunk_size,
    )

    def interval_chunks():
        for chunk in iter_prepared_chunks(
            csv_path,
            window,
            chunk_size,
        ):
            selected = chunk[
                (
                    chunk["FlowStart"]
                    >= first_attack
                )
                & (
                    chunk["FlowStart"]
                    <= last_attack
                )
            ].copy()

            yield selected

    block = create_windows_chunked(
        interval_chunks(),
        attack_ratio_threshold,
    )

    block["OriginalTimestamp"] = block["Timestamp"]
    block["AttackBlock"] = attack_name

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    block.to_csv(
        output_path,
        index=False,
    )

    return block


def attack_block_info(
    attack_block,
    attack_name,
):
    block = attack_block.copy()

    block["OriginalTimestamp"] = pd.to_datetime(
        block["OriginalTimestamp"]
    )

    first_timestamp = block[
        "OriginalTimestamp"
    ].min()

    last_timestamp = block[
        "OriginalTimestamp"
    ].max()

    real_duration = (
        last_timestamp
        - first_timestamp
    )

    attack_windows = int(
        (
            block["Label"]
            != "BENIGN"
        ).sum()
    )

    benign_windows = int(
        (
            block["Label"]
            == "BENIGN"
        ).sum()
    )

    return {
        "Attack": attack_name,
        "FirstTimestamp": first_timestamp,
        "LastTimestamp": last_timestamp,
        "RealDuration": real_duration,
        "TotalWindows": len(block),
        "AttackWindows": attack_windows,
        "BenignWindows": benign_windows,
    }


def _extract_segment(
    attack_block,
    start,
    duration,
):
    end = (
        start + duration
    )

    piece = attack_block[
        (
            attack_block["OriginalTimestamp"]
            >= start
        )
        & (
            attack_block["OriginalTimestamp"]
            < end
        )
    ].copy()

    return piece


def prepare_attack_segments(
    attack_block,
    attack_name,
    required_count,
    preferred_duration,
):
    block = attack_block.copy()

    block["OriginalTimestamp"] = pd.to_datetime(
        block["OriginalTimestamp"]
    )

    block = (
        block
        .sort_values(
            "OriginalTimestamp"
        )
        .reset_index(
            drop=True
        )
    )

    first_timestamp = block[
        "OriginalTimestamp"
    ].min()

    last_timestamp = block[
        "OriginalTimestamp"
    ].max()

    # + 1 segundo para representar a extensão da última janela.
    available_duration = (
        last_timestamp
        - first_timestamp
        + pd.Timedelta("1s")
    )

    preferred_duration = pd.Timedelta(
        preferred_duration
    )

    # O bloco canônico nunca é maior que o tempo realmente disponível.
    canonical_duration = min(
        preferred_duration,
        available_duration,
    )

    required_total_duration = (
        canonical_duration
        * required_count
    )

    segments = []
    strategy = None

    # Caso 1:
    # há duração suficiente para gerar todos os blocos sem repetição.
    if (
        available_duration
        >= required_total_duration
    ):
        strategy = "distinct"

        cursor = first_timestamp

        for index in range(
            required_count
        ):
            piece = _extract_segment(
                block,
                cursor,
                canonical_duration,
            )

            if len(piece) == 0:
                raise RuntimeError(
                    f"{attack_name}: segmento {index + 1} ficou vazio."
                )

            segments.append(
                piece
            )

            cursor = (
                cursor
                + canonical_duration
            )

    # Caso 2:
    # o ataque é curto. Um único bloco real é criado e reutilizado.
    else:
        strategy = "repeat"

        canonical = _extract_segment(
            block,
            first_timestamp,
            canonical_duration,
        )

        if len(canonical) == 0:
            raise RuntimeError(
                f"{attack_name}: bloco canônico vazio."
            )

        for _ in range(
            required_count
        ):
            segments.append(
                canonical.copy()
            )

    metadata = {
        "Attack": attack_name,
        "RequiredOccurrences": required_count,
        "AvailableDuration": available_duration,
        "PreferredDuration": preferred_duration,
        "SelectedBlockDuration": canonical_duration,
        "Strategy": strategy,
        "SegmentsCreated": len(segments),
    }

    return (
        segments,
        metadata,
    )
