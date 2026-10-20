"""PCAP/PCAPNG, inclusive sem extensão; intercalação cronológica entre arquivos."""

from contextlib import ExitStack
from datetime import datetime, timezone
from decimal import Decimal
import heapq
from pathlib import Path

from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6
from scapy.utils import PcapReader

PCAP_MAGIC = {b'\xd4\xc3\xb2\xa1', b'\xa1\xb2\xc3\xd4', b'\x4d\x3c\xb2\xa1',
              b'\xa1\xb2\x3c\x4d', b'\x0a\x0d\x0d\x0a'}


def utc_timestamp(timestamp_ns):
    seconds, nanos = divmod(timestamp_ns, 1_000_000_000)
    base = datetime.fromtimestamp(seconds, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    return f'{base}.{nanos:09d}Z'


def discover_pcaps(directory, recursive=True):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Pasta PCAP não encontrada: {directory}')
    paths = directory.rglob('*') if recursive else directory.iterdir()
    found = []
    for path in paths:
        if path.is_file():
            with path.open('rb') as handle:
                if handle.read(4) in PCAP_MAGIC:
                    found.append(path.resolve())
    if not found:
        raise FileNotFoundError(f'Nenhum PCAP/PCAPNG identificado pelo cabeçalho em {directory}.')
    return sorted(found, key=lambda path: str(path).lower())


def packet_record(packet):
    timestamp_ns = int(Decimal(str(packet.time)) * 1_000_000_000)
    row = dict(TimestampNs=timestamp_ns, IsIPv4=False, Truncated=False)
    network = packet
    while network and not isinstance(network, (IP, IPv6)):
        network = network.payload
    if not isinstance(network, IP):
        return row
    ip = network  # Primeiro cabeçalho IP; IPv4 encapsulado em IPv6 não entra.
    header = int(ip.ihl or 0) * 4
    declared = int(ip.len or 0)
    available = len(ip.original) if ip.original else len(bytes(ip))
    if header < 20 or declared < header or available < header:
        raise ValueError(f'Cabeçalho IPv4 inválido no timestamp {utc_timestamp(timestamp_ns)}.')
    length = min(declared, available)
    protocol = int(ip.proto)
    fragmented = bool(int(ip.frag) or (int(ip.flags) & 1))
    row.update(IsIPv4=True, SourceIP=ip.src, DestinationIP=ip.dst, Protocol=protocol,
               IPLength=length, Truncated=available < declared, Fragmented=fragmented,
               SourcePort=None, DestinationPort=None, TCPFlags=None, TCPWindow=None,
               TCPSequence=None, TransportHeaderLength=None, PayloadLength=None,
               Assignable=False)
    transport = ip.payload
    # Fragmentos permanecem nas contagens IP/entropias. Sem remontagem não se
    # atribuem seus comprimentos/direções às conexões TCP/UDP.
    if fragmented:
        return row
    if protocol == 6 and isinstance(transport, TCP):
        transport_header = int(transport.dataofs or 0) * 4
        if transport_header < 20 or length < header + transport_header:
            return row
        row.update(SourcePort=int(transport.sport), DestinationPort=int(transport.dport),
                   TCPFlags=int(transport.flags), TCPWindow=int(transport.window),
                   TCPSequence=int(transport.seq), TransportHeaderLength=transport_header)
    elif protocol == 17 and isinstance(transport, UDP) and length >= header + 8:
        row.update(SourcePort=int(transport.sport), DestinationPort=int(transport.dport),
                   TransportHeaderLength=8)
    elif protocol not in (6, 17):
        # Pseudofluxos por par de IPs e protocolo; portas não se aplicam.
        transport_header = 8 if protocol == 1 and isinstance(transport, ICMP) and length >= header + 8 else 0
        row['TransportHeaderLength'] = transport_header
    else:
        return row
    row['PayloadLength'] = max(0, length - header - row['TransportHeaderLength'])
    row['Assignable'] = True
    return row


def iter_merged_packets(paths):
    """Intercala arquivos ordenados internamente, sem carregar capturas na RAM.

    Mantém um pacote de antecipação por arquivo. Sobreposições temporais são
    intercaladas; pacotes iguais não são deduplicados. Regressões internas de
    timestamp interrompem a execução para evitar IATs e janelas incorretos.
    """
    with ExitStack() as stack:
        readers, previous, heap = [], {}, []
        for file_index, path in enumerate(paths):
            reader = stack.enter_context(PcapReader(str(path)))
            readers.append(reader)
            try:
                packet = next(reader)
            except (StopIteration, EOFError):
                continue
            timestamp = int(Decimal(str(packet.time)) * 1_000_000_000)
            previous[file_index] = timestamp
            heapq.heappush(heap, (timestamp, file_index, 1, packet))
        while heap:
            timestamp, file_index, index, packet = heapq.heappop(heap)
            yield packet_record(packet)
            try:
                packet = next(readers[file_index])
            except (StopIteration, EOFError):
                continue
            next_timestamp = int(Decimal(str(packet.time)) * 1_000_000_000)
            if next_timestamp < previous[file_index]:
                raise ValueError(f'Timestamp fora de ordem em {paths[file_index]}, pacote {index + 1}.')
            previous[file_index] = next_timestamp
            heapq.heappush(heap, (next_timestamp, file_index, index + 1, packet))
