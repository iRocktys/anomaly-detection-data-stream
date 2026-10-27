"""Índice persistente de fluxos e associação temporal durante a leitura."""
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
import hashlib
import heapq
import json
import sqlite3

import pandas as pd

from .Reader import flow_key, utc_timestamp

REQUIRED_COLUMNS = ['Source IP', 'Source Port', 'Destination IP', 'Destination Port',
                    'Protocol', 'Timestamp', 'Flow Duration', 'Label']


def local_timestamp_ns(value, utc_offset_hours):
    """Horário local explícito -> epoch; nunca usa o fuso do computador."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        raise ValueError('Use data/hora sem fuso e informe csvUtcOffsetHours separadamente.')
    offset = Decimal(str(utc_offset_hours)) * 3_600_000_000_000
    if offset != offset.to_integral_value() or abs(offset) > 24 * 3_600_000_000_000:
        raise ValueError('Offset UTC inválido.')
    return int(stamp.value) - int(offset)


def discover_csvs(directory):
    files = sorted(p.resolve() for p in Path(directory).rglob('*.csv') if p.is_file())
    if not files:
        raise FileNotFoundError(f'Nenhum CSV original encontrado em {directory}')
    return files


def inspect_csvs(paths, chunksize=100_000):
    """Inspeção leve: cabeçalho e três linhas; não lê todos os CSVs."""
    rows = []
    for path in paths:
        preview = pd.read_csv(path, nrows=3, encoding='utf-8-sig')
        preview.columns = preview.columns.str.strip()
        missing = sorted(set(REQUIRED_COLUMNS) - set(preview.columns))
        if missing:
            raise ValueError(f'{path}: faltam colunas {missing}. Use CSVs originais.')
        rows.append(dict(File=str(path), SampleTimestamp=preview['Timestamp'].iloc[0] if len(preview) else None,
                         SampleLabels=', '.join(preview['Label'].astype(str).unique()),
                         Bytes=Path(path).stat().st_size))
    return rows


class FlowLabelIndex:
    def __init__(self, path):
        self.path = Path(path)
        with sqlite3.connect(self.path) as connection:
            self.summary = json.loads(connection.execute('SELECT value FROM metadata WHERE name="summary"').fetchone()[0])

    @classmethod
    def build(cls, csv_paths, output_path, csv_utc_offset_hours=0,
              start_ns=None, cutoff_ns=None, chunksize=100_000, rebuild=False):
        paths = sorted(Path(p).resolve() for p in csv_paths)
        if not paths or len(set(paths)) != len(paths):
            raise ValueError('Informe uma lista não vazia de CSVs distintos.')
        signature_data = dict(version=1, files=[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths],
                              offset=csv_utc_offset_hours, start=start_ns, cutoff=cutoff_ns)
        signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
        output = Path(output_path).resolve()
        if output in paths:
            raise ValueError('O índice não pode sobrescrever um CSV de entrada.')
        if output.exists() and not rebuild:
            existing = cls(output)
            if existing.summary['Signature'] == signature:
                print('Índice existente reutilizado; entradas e configuração coincidem.')
                return existing
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + '.partial')
        temporary.unlink(missing_ok=True)
        counts, total, retained, unsupported = Counter(), 0, 0, 0
        offset_ns = local_timestamp_ns('1970-01-01', csv_utc_offset_hours)
        # offset_ns já é o negativo do deslocamento local.
        connection = sqlite3.connect(temporary)
        try:
            connection.execute('CREATE TABLE flows (start_ns INTEGER, end_ns INTEGER, key TEXT, label TEXT)')
            connection.execute('CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT)')
            for path in paths:
                header = pd.read_csv(path, nrows=0, encoding='utf-8-sig').columns
                names = {name.strip(): name for name in header}
                missing = sorted(set(REQUIRED_COLUMNS) - set(names))
                if missing:
                    raise ValueError(f'{path}: faltam {missing}')
                columns = [names[name] for name in REQUIRED_COLUMNS]
                for chunk in pd.read_csv(path, usecols=columns, dtype=str, chunksize=chunksize,
                                         encoding='utf-8-sig', keep_default_na=False):
                    chunk.columns = chunk.columns.str.strip()
                    total += len(chunk)
                    # O exemplo enviado usa ISO com microssegundos, sem fuso.
                    starts = pd.to_datetime(chunk['Timestamp'], format='ISO8601', errors='raise')
                    if starts.dt.tz is not None or starts.isna().any():
                        raise ValueError(f'{path}: timestamp deve ser ISO sem fuso e não vazio.')
                    starts_ns = starts.astype('datetime64[ns]').astype('int64') + offset_ns
                    values = chunk[REQUIRED_COLUMNS].itertuples(index=False, name=None)
                    batch = []
                    for start, row in zip(starts_ns, values):
                        src, sport, dst, dport, proto, stamp, duration, label = row
                        duration = Decimal(duration)
                        if not duration.is_finite() or duration < 0 or duration != duration.to_integral_value():
                            raise ValueError(f'{path}: Flow Duration inválida em {stamp}')
                        start = int(start)
                        end = start + int(duration) * 1000
                        if end > 9_223_372_036_854_775_807:
                            raise ValueError('Duração excede limite temporal.')
                        if (start_ns is not None and end < start_ns) or (cutoff_ns is not None and start >= cutoff_ns):
                            continue
                        label = label.strip()
                        if not label:
                            raise ValueError(f'{path}: Label vazio.')
                        if label.upper() == 'BENIGN':
                            label = 'BENIGN'
                        protocol = Decimal(proto)
                        if not protocol.is_finite() or protocol != protocol.to_integral_value():
                            raise ValueError(f'{path}: protocolo inválido.')
                        if int(protocol) not in (6, 17):
                            unsupported += 1
                            continue
                        ports = [Decimal(sport), Decimal(dport)]
                        if any(not p.is_finite() or p != p.to_integral_value() or not 0 <= p <= 65535 for p in ports):
                            raise ValueError(f'{path}: porta inválida.')
                        key = flow_key(src, int(ports[0]), dst, int(ports[1]), int(protocol))
                        batch.append((start, end, key, label))
                        counts[label] += 1
                    connection.executemany('INSERT INTO flows VALUES (?,?,?,?)', batch)
                    retained += len(batch)
                    connection.commit()
                print(f'CSV indexado: {path.name} | linhas lidas acumuladas: {total:,}', flush=True)
            connection.execute('CREATE INDEX by_start ON flows(start_ns)')
            bounds = connection.execute('SELECT min(start_ns), max(end_ns) FROM flows').fetchone()
            summary = dict(Signature=signature, CsvFiles=len(paths), RowsRead=total,
                           IndexedFlows=retained, UnsupportedProtocolFlows=unsupported,
                           FlowLabels=dict(counts), CsvUtcOffsetHours=csv_utc_offset_hours,
                           FirstFlowUTC=utc_timestamp(bounds[0]) if bounds[0] is not None else None,
                           LastFlowEndUTC=utc_timestamp(bounds[1]) if bounds[1] is not None else None)
            connection.execute('INSERT INTO metadata VALUES (?,?)', ('summary', json.dumps(summary)))
            connection.commit()
            connection.close()
            temporary.replace(output)
        except BaseException:
            connection.close()
            temporary.unlink(missing_ok=True)
            raise
        return cls(output)

    def matcher(self, first_ns, cutoff_ns=None):
        return FlowLabelMatcher(self.path, first_ns, cutoff_ns)


class FlowLabelMatcher:
    """Varredura temporal: SQLite entrega fluxos uma vez, sem SQL por pacote.

    Intervalos fechados [início,fim], tolerância zero. Múltiplos fluxos com
    o mesmo rótulo permitem associação; rótulos diferentes são conflito.
    Não exclui nem duplica pacotes ao encontrar referências repetidas.
    """
    def __init__(self, path, first_ns, cutoff_ns=None):
        self.connection = sqlite3.connect(path)
        self.cursor = self.connection.execute(
            'SELECT start_ns,end_ns,key,label FROM flows WHERE end_ns >= ? AND start_ns < ? ORDER BY start_ns',
            (int(first_ns), int(cutoff_ns) if cutoff_ns is not None else 9_223_372_036_854_775_807))
        self.next = self.cursor.fetchone()
        self.active = defaultdict(Counter)
        self.ends = []
        self.previous = None
        self.serial = 0

    def match(self, packet):
        ns = packet['TimestampNs']
        if self.previous is not None and ns < self.previous:
            raise ValueError('O índice temporal exige pacotes em ordem.')
        self.previous = ns
        while self.next is not None and self.next[0] <= ns:
            start, end, key, label = self.next
            if end >= ns:
                self.active[key][label] += 1
                self.serial += 1
                heapq.heappush(self.ends, (end, self.serial, key, label))
            self.next = self.cursor.fetchone()
        while self.ends and self.ends[0][0] < ns:
            end, serial, key, label = heapq.heappop(self.ends)
            self.active[key][label] -= 1
            if self.active[key][label] == 0:
                del self.active[key][label]
            if not self.active[key]:
                del self.active[key]
        if packet['MatchKey'] is None:
            return None, 'unsupported'
        labels = self.active.get(packet['MatchKey'])
        if not labels:
            return None, 'unmatched'
        if len(labels) != 1:
            return None, 'ambiguous'
        return next(iter(labels)), 'matched'

    def close(self):
        self.connection.close()
