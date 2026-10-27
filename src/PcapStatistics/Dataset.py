"""Extração única de features e contagens de referência por janela."""
from collections import Counter
import csv
from dataclasses import dataclass, asdict
from decimal import Decimal
import json
from pathlib import Path
import time

from .Features import FEATURE_NAMES, METADATA_NAMES, WindowAccumulator
from .Reader import iter_merged_packets, utc_timestamp

COUNT_COLUMNS = ['WindowStart', 'Timestamp', 'WindowComplete', 'Packet_Count',
                 'Benign_Packet_Count', 'Attack_Packet_Count', 'Unknown_Packet_Count',
                 'Ambiguous_Packet_Count', 'Unmatched_Packet_Count', 'Unsupported_Packet_Count',
                 'Attack_Packet_Ratio', 'Label_Coverage', 'Attack_Class_Counts']


@dataclass
class DatasetResult:
    featuresPath: str
    countsPath: str
    packetsProcessed: int
    ipv4Packets: int
    windowsWritten: int
    packetLimitReached: bool
    elapsedSeconds: float
    packetsPerSecond: float
    labelCoverage: float
    matchedPackets: int
    unknownPackets: int
    ambiguousPackets: int
    truncatedPackets: int
    fragmentedPackets: int
    firstTimestamp: str
    lastTimestamp: str

    def as_dict(self):
        return asdict(self)


class PcapStatisticsDataset:
    def __init__(self, windowSeconds=1.0, maxPackets=1_000_000, keepEmptyWindows=True,
                 progressEvery=100_000):
        duration = Decimal(str(windowSeconds)) * 1_000_000_000
        if not duration.is_finite() or duration <= 0 or duration != duration.to_integral_value():
            raise ValueError('windowSeconds deve ser positivo e representável em ns.')
        for name, value in [('maxPackets', maxPackets), ('progressEvery', progressEvery)]:
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f'{name} deve ser inteiro positivo ou None.')
        self.duration_ns = int(duration)
        self.maxPackets, self.keepEmptyWindows, self.progressEvery = maxPackets, keepEmptyWindows, progressEvery

    def runFiles(self, pcapFiles, outputDirectory, labelIndex, cutoffNs=None):
        files = [Path(p).resolve() for p in pcapFiles]
        if not files or len(set(files)) != len(files):
            raise ValueError('Informe PCAPs distintos e uma lista não vazia.')
        output = Path(outputDirectory).resolve()
        output.mkdir(parents=True, exist_ok=True)
        feature_path, count_path = output / 'window_features.csv', output / 'window_label_counts.csv'
        temp_features = feature_path.with_suffix('.csv.partial')
        temp_counts = count_path.with_suffix('.csv.partial')
        if set(files) & {feature_path, count_path, temp_features, temp_counts}:
            raise ValueError('Saídas não podem coincidir com entradas.')
        records = iter_merged_packets(files, cutoff_ns=cutoffNs, max_packets=self.maxPackets)
        matcher = None
        total = ipv4 = windows = matched = unknown = ambiguous = truncated = fragmented = 0
        first_ns = last_ns = current_start = None
        accumulator, labels, reasons = WindowAccumulator(), Counter(), Counter()
        started = time.perf_counter()
        try:
            with temp_features.open('w', newline='', encoding='utf-8') as fh, temp_counts.open('w', newline='', encoding='utf-8') as ch:
                fw = csv.DictWriter(fh, fieldnames=METADATA_NAMES + FEATURE_NAMES)
                cw = csv.DictWriter(ch, fieldnames=COUNT_COLUMNS)
                fw.writeheader()
                cw.writeheader()

                def emit(complete):
                    nonlocal windows
                    meta = dict(WindowStart=utc_timestamp(current_start),
                                Timestamp=utc_timestamp(current_start + self.duration_ns),
                                WindowComplete=int(complete))
                    features = accumulator.finish()
                    fw.writerow({**meta, **features})
                    benign = labels.get('BENIGN', 0)
                    attacks = {k: v for k, v in labels.items() if k != 'BENIGN'}
                    attack = sum(attacks.values())
                    missing = sum(reasons.values())
                    n = accumulator.packets
                    assert benign + attack + missing == n
                    cw.writerow({**meta, 'Packet_Count': n, 'Benign_Packet_Count': benign,
                                 'Attack_Packet_Count': attack, 'Unknown_Packet_Count': missing,
                                 'Ambiguous_Packet_Count': reasons['ambiguous'],
                                 'Unmatched_Packet_Count': reasons['unmatched'],
                                 'Unsupported_Packet_Count': reasons['unsupported'],
                                 'Attack_Packet_Ratio': attack / n if n else 0.0,
                                 'Label_Coverage': (benign + attack) / n if n else 0.0,
                                 'Attack_Class_Counts': json.dumps(attacks, sort_keys=True)})
                    windows += 1

                for packet in records:
                    ns = packet['TimestampNs']
                    start = ns // self.duration_ns * self.duration_ns
                    if first_ns is None:
                        first_ns = ns
                        current_start = start
                        matcher = labelIndex.matcher(ns, cutoffNs)
                    elif start != current_start:
                        emit(current_start >= first_ns)
                        current_start += self.duration_ns
                        accumulator, labels, reasons = WindowAccumulator(), Counter(), Counter()
                        if self.keepEmptyWindows:
                            while current_start < start:
                                emit(True)
                                current_start += self.duration_ns
                        current_start = start
                    total += 1
                    last_ns = ns
                    accumulator.update(packet)
                    if packet['IsIPv4']:
                        ipv4 += 1
                        truncated += packet['Truncated']
                        fragmented += packet['Fragmented']
                        label, status = matcher.match(packet)
                        if status == 'matched':
                            labels[label] += 1
                            matched += 1
                        else:
                            reasons[status] += 1
                            unknown += 1
                            ambiguous += status == 'ambiguous'
                    if self.progressEvery and total % self.progressEvery == 0:
                        elapsed = time.perf_counter() - started
                        print(f'{total:,} pacotes | {windows:,} janelas | {total / elapsed:,.0f} pacotes/s | '
                              f'cobertura IPv4: {matched / ipv4:.2%}' if ipv4 else f'{total:,} pacotes | sem IPv4', flush=True)
                if first_ns is None:
                    raise ValueError('Nenhum pacote antes do corte; confira data/fuso e entradas.')
                # Conservador: última janela é parcial, pois limite/EOF pode cortá-la.
                emit(False)
            temp_features.replace(feature_path)
            temp_counts.replace(count_path)
        finally:
            records.close()
            if matcher is not None:
                matcher.close()
            temp_features.unlink(missing_ok=True)
            temp_counts.unlink(missing_ok=True)
        elapsed = time.perf_counter() - started
        return DatasetResult(str(feature_path), str(count_path), total, ipv4, windows,
                             self.maxPackets is not None and total >= self.maxPackets,
                             elapsed, total / elapsed, matched / ipv4 if ipv4 else 0.0,
                             matched, unknown, ambiguous, truncated, fragmented,
                             utc_timestamp(first_ns), utc_timestamp(last_ns))
