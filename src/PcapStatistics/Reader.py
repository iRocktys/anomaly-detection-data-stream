"""Leitura Scapy e intercalação de capturas internamente ordenadas."""
from contextlib import ExitStack
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import heapq
import ipaddress

from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.utils import PcapReader

PCAP_MAGIC = {b'\xd4\xc3\xb2\xa1', b'\xa1\xb2\xc3\xd4', b'\x4d\x3c\xb2\xa1',
              b'\xa1\xb2\x3c\x4d', b'\x0a\x0d\x0d\x0a'}


def utc_timestamp(ns):
    seconds, nanos = divmod(int(ns), 1_000_000_000)
    return datetime.fromtimestamp(seconds, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + f'.{nanos:09d}Z'


def packet_ns(packet):
    return int(Decimal(str(packet.time)) * 1_000_000_000)


def flow_key(src, sport, dst, dport, protocol):
    """Chave bidirecional; não identifica por si só uma ocorrência temporal."""
    a = (str(ipaddress.IPv4Address(src)), int(sport))
    b = (str(ipaddress.IPv4Address(dst)), int(dport))
    a, b = sorted((a, b))
    return f'{int(protocol)}|{a[0]}|{a[1]}|{b[0]}|{b[1]}'


def discover_pcaps(directory, recursive=True):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Pasta PCAP não encontrada: {directory}')
    found = []
    for path in (directory.rglob('*') if recursive else directory.iterdir()):
        if path.is_file():
            with path.open('rb') as handle:
                if handle.read(4) in PCAP_MAGIC:
                    found.append(path.resolve())
    if not found:
        raise FileNotFoundError(f'Nenhum PCAP/PCAPNG em {directory}')
    return sorted(found, key=lambda p: str(p).lower())


def inspect_inputs(paths):
    """Somente primeiro registro por arquivo; NÃO certifica ordem integral."""
    rows = []
    for path in paths:
        with PcapReader(str(path)) as reader:
            packet = next(reader, None)
            ns = packet_ns(packet) if packet is not None else None
        rows.append(dict(File=str(path), Bytes=Path(path).stat().st_size,
                         FirstTimestamp=utc_timestamp(ns) if ns is not None else None,
                         FirstTimestampNs=ns))
    return rows


def packet_record(packet):
    row = dict(TimestampNs=packet_ns(packet), IsIPv4=False, Truncated=False,
               Fragmented=False, SourcePort=None, DestinationPort=None,
               TCPFlags=None, MatchKey=None)
    network = packet
    while network and not isinstance(network, (IP, IPv6)):
        network = network.payload
    if not isinstance(network, IP):
        return row
    ip = network
    header = int(ip.ihl or 0) * 4
    declared = int(ip.len or 0)
    available = len(ip.original) if ip.original else len(bytes(ip))
    if header < 20 or declared < header or available < header:
        raise ValueError(f'IPv4 inválido em {utc_timestamp(row["TimestampNs"])}')
    fragmented = bool(int(ip.frag) or (int(ip.flags) & 1))
    row.update(IsIPv4=True, SourceIP=ip.src, DestinationIP=ip.dst,
               Protocol=int(ip.proto), IPLength=declared, TTL=int(ip.ttl),
               Truncated=available < declared, Fragmented=fragmented)
    # Comprimento IP declarado permite contar bytes mesmo com snaplen reduzido.
    # Sem remontagem, fragmentos ficam sem portas/flags/associação.
    if fragmented:
        return row
    transport = ip.payload
    if row['Protocol'] == 6 and isinstance(transport, TCP):
        size = int(transport.dataofs or 0) * 4
        if size < 20 or min(declared, available) < header + size:
            return row
        row.update(SourcePort=int(transport.sport), DestinationPort=int(transport.dport),
                   TCPFlags=int(transport.flags))
    elif row['Protocol'] == 17 and isinstance(transport, UDP):
        if min(declared, available) < header + 8:
            return row
        row.update(SourcePort=int(transport.sport), DestinationPort=int(transport.dport))
    else:
        # ICMP/outros exigem regra específica do exportador; não inferir portas.
        return row
    row['MatchKey'] = flow_key(ip.src, row['SourcePort'], ip.dst,
                               row['DestinationPort'], row['Protocol'])
    return row


def iter_merged_packets(paths, cutoff_ns=None, max_packets=None):
    """Uma antecipação por arquivo. Limite conta registros, inclusive não IPv4.

    A interrupção pelo corte pressupõe ordem interna em TODOS os arquivos.
    Regressões no prefixo lido causam erro; o sufixo não lido não é validado.
    """
    with ExitStack() as stack:
        readers, previous, heap = [], {}, []
        for index, path in enumerate(paths):
            reader = stack.enter_context(PcapReader(str(path)))
            readers.append(reader)
            packet = next(reader, None)
            if packet is not None:
                ns = packet_ns(packet)
                previous[index] = ns
                heapq.heappush(heap, (ns, index, 1, packet))
        processed = 0
        while heap:
            ns, index, number, packet = heapq.heappop(heap)
            if cutoff_ns is not None and ns >= cutoff_ns:
                return
            yield packet_record(packet)
            processed += 1
            if max_packets is not None and processed >= max_packets:
                return
            packet = next(readers[index], None)
            if packet is not None:
                nxt = packet_ns(packet)
                if nxt < previous[index]:
                    raise ValueError(f'Timestamp fora de ordem: {paths[index]}, pacote {number + 1}')
                previous[index] = nxt
                heapq.heappush(heap, (nxt, index, number + 1, packet))
