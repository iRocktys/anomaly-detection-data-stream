"""Geração contínua de um CSV global, sem rótulos, cenários ou relatórios extras."""

import csv
from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral
from pathlib import Path
import time

from .Features import FEATURE_NAMES, METADATA_NAMES, WindowAccumulator, feature_dictionary
from .Flows import FlowTracker
from .Reader import discover_pcaps, iter_merged_packets, utc_timestamp


def seconds_ns(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} deve ser positivo e representável em nanossegundos.')
    try:
        decimal = Decimal(str(value))
        nanos = decimal * 1_000_000_000
        if not decimal.is_finite() or decimal <= 0 or nanos != nanos.to_integral_value():
            raise ValueError
        return int(nanos)
    except (ValueError, ArithmeticError):
        raise ValueError(f'{name} deve ser positivo e representável em nanossegundos.') from None


@dataclass(frozen=True)
class DatasetResult:
    datasetPath: str
    featureReportPath: str
    packetsProcessed: int
    ipv4Packets: int
    windowsWritten: int
    packetLimitReached: bool
    elapsedSeconds: float


class PcapStatisticsDataset:
    """O limite é global entre arquivos e inclui pacotes não IPv4.

    maxPackets=None processa a captura inteira. A intercalação cronológica
    mantém um pacote de antecipação por arquivo; esse buffer não é uma
    observação processada nem uma linha adicional no dataset.
    """
    def __init__(self, windowSeconds=1.0, maxPackets=None,
                 flowIdleTimeoutSeconds=120.0, subflowGapSeconds=1.0,
                 packetLengthMode='transport_payload', keepEmptyWindows=True,
                 progressEvery=None):
        self.duration_ns = seconds_ns(windowSeconds, 'windowSeconds')
        self.idle_ns = seconds_ns(flowIdleTimeoutSeconds, 'flowIdleTimeoutSeconds')
        self.subflow_ns = seconds_ns(subflowGapSeconds, 'subflowGapSeconds')
        for name, value in [('maxPackets', maxPackets), ('progressEvery', progressEvery)]:
            if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 1):
                raise ValueError(f'{name} deve ser inteiro positivo ou None.')
        if packetLengthMode not in ('transport_payload', 'ip'):
            raise ValueError('packetLengthMode deve ser transport_payload ou ip.')
        if not isinstance(keepEmptyWindows, bool):
            raise ValueError('keepEmptyWindows deve ser booleano.')
        self.maxPackets = maxPackets
        self.packetLengthMode = packetLengthMode
        self.keepEmptyWindows = keepEmptyWindows
        self.progressEvery = progressEvery

    def run(self, pcapDirectory, outputPath, featureReportPath, recursive=True):
        return self.runFiles(discover_pcaps(pcapDirectory, recursive), outputPath, featureReportPath)

    def runFiles(self, pcapFiles, outputPath, featureReportPath):
        files = [Path(path).resolve() for path in pcapFiles]
        if not files or len(set(files)) != len(files):
            raise ValueError('Informe arquivos PCAP distintos e uma lista não vazia.')
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
        output, report = Path(outputPath).resolve(), Path(featureReportPath).resolve()
        if output == report or output in files or report in files:
            raise ValueError('Dataset, relatório e arquivos de entrada devem ter caminhos distintos.')
        output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        # Publica somente resultados concluídos; falhas preservam a execução anterior.
        temporary = output.with_name(output.name + '.partial')
        temporary_report = report.with_name(report.name + '.partial')
        if temporary in files or temporary_report in files:
            raise ValueError('Arquivo temporário coincide com uma entrada.')
        started = time.perf_counter()
        tracker = FlowTracker(self.idle_ns)
        accumulator = WindowAccumulator(self.packetLengthMode, self.subflow_ns)
        current_start = None
        total = ipv4 = windows = 0
        limited = False
        records = iter_merged_packets(files)
        try:
            with temporary.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=METADATA_NAMES + FEATURE_NAMES)
                writer.writeheader()

                def emit():
                    nonlocal windows
                    row = dict(Timestamp=utc_timestamp(current_start + self.duration_ns))
                    row.update(accumulator.finish(self.duration_ns / 1_000_000_000))
                    writer.writerow(row)
                    windows += 1

                for packet in records:
                    timestamp = packet['TimestampNs']
                    window_start = timestamp // self.duration_ns * self.duration_ns
                    if current_start is None:
                        current_start = window_start
                    elif window_start != current_start:
                        emit()
                        current_start += self.duration_ns
                        accumulator = WindowAccumulator(self.packetLengthMode, self.subflow_ns)
                        if self.keepEmptyWindows:
                            while current_start < window_start:
                                emit()
                                current_start += self.duration_ns
                        current_start = window_start
                        accumulator = WindowAccumulator(self.packetLengthMode, self.subflow_ns)
                    assignment = tracker.observe(packet) if packet['IsIPv4'] and packet['Assignable'] else None
                    accumulator.update(packet, assignment)
                    total += 1
                    ipv4 += int(packet['IsIPv4'])
                    if self.progressEvery and total % self.progressEvery == 0:
                        print(f'{total:,} pacotes processados; {windows:,} janelas gravadas.')
                    if self.maxPackets is not None and total >= self.maxPackets:
                        limited = True
                        break
                if current_start is None:
                    raise ValueError('Os arquivos não contêm pacotes.')
                emit()
            with temporary_report.open('w', newline='', encoding='utf-8') as handle:
                rows = feature_dictionary(self.packetLengthMode, self.idle_ns / 1e9, self.subflow_ns / 1e9,
                                          window_seconds=self.duration_ns / 1e9)
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(output)
            temporary_report.replace(report)
        finally:
            records.close()
            temporary.unlink(missing_ok=True)
            temporary_report.unlink(missing_ok=True)
        return DatasetResult(str(output), str(report), total, ipv4, windows, limited,
                             time.perf_counter() - started)
